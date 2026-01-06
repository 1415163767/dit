import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from random import shuffle
from threading import Thread
from transformers import AutoProcessor

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, VideoReader_contextmanager,
                    get_random_mask, get_video_reader_batch, resize_frame)


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


def clean_caption(caption: str):
    caption = caption.strip()
    removed = False
    sentences = caption.split(". ")

    NEGATIVE_KEYWORDS = [
        "no", "not", "unreadable", "invisible", "illegible", "missing", "absent"
    ]
    TARGET_KEYWORDS = [
        "text", "caption", "subtitle", "word", "writing"
    ]

    if sentences:
        last = sentences[-1].rstrip(".").lower()
        if any(neg in last for neg in NEGATIVE_KEYWORDS) and any(
            tgt in last for tgt in TARGET_KEYWORDS
        ):
            sentences = sentences[:-1]
            removed = True

    cleaned_caption = ". ".join(s.rstrip(".") for s in sentences).strip()
    if cleaned_caption and not cleaned_caption.endswith("."):
        cleaned_caption += "."

    return cleaned_caption, removed


class ImageVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        video_sample_stride=1, video_sample_n_frames=33, vit_sample_stride=2,
        resolution_list=[(384, 384), (288, 512), (512, 288)],
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):  
        list_data_dict = []
        for root, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if d != 'videos']
            if 'video_captions_all_long_short.json' in files:
                removed_count = 0
                too_short_count = 0
                json_path = os.path.join(root, 'video_captions_all_long_short.json')
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    if "caption" in item and isinstance(item["caption"], str):
                        if len(item["caption"]) <= 10:
                            too_short_count += 1
                            continue
                        cleaned, removed = clean_caption(item["caption"])
                        item["caption"] = cleaned
                        if removed:
                            removed_count += 1

                list_data_dict.extend(data)
                print(
                    f"[OK] {json_path} | "
                    f"entries: {len(data)}, "
                    f"cleaned: {removed_count}, "
                    f"too short: {too_short_count}"
                )
        
        random.shuffle(list_data_dict)  # Randomly shuffle the data for training
        self.dataset = list_data_dict
        self.length = len(self.dataset)
        print(f"total data scale: {self.length}")

        # Enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.resolution_list        = resolution_list
        self.video_transforms = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Qwen3-VL-2B processor
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
        )

        # ViT params
        self.vit_sample_stride = vit_sample_stride
        self.processor.video_processor.size['longest_edge'] = 83886080

    def sample_indexes(self, video_length, sample_n_frames):
        clip_length  = min(video_length, (sample_n_frames - 1) * self.video_sample_stride + 1)
        start_idx    = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
        batch_index  = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)
        remainder = (len(batch_index) - 1) % 4
        if remainder != 0:
            pad_len = 4 - remainder
            batch_index = np.pad(batch_index, (0, pad_len), mode='edge')
        return batch_index

    def read_video_frames(self, video_file_path, idx):
        if isinstance(video_file_path, np.ndarray):
            video_length = int(self.video_length_drop_end * video_file_path.shape[0])
            min_sample_n_frames = min(
                self.video_sample_n_frames,
                int(video_file_path.shape[0] * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
            )
            batch_index = self.sample_indexes(video_length, min_sample_n_frames)
            pixel_values = video_file_path[batch_index]

        elif isinstance(video_file_path, str):
            with VideoReader_contextmanager(video_file_path, num_threads=4) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                video_length = int(self.video_length_drop_end * len(video_reader))
                batch_index = self.sample_indexes(video_length, min_sample_n_frames)

            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")
            
        else:
            raise ValueError(f"Unsupported video_file_path type: {type(video_file_path)}")
        
        processed_frames = []
        h, w, _ = pixel_values[0].shape
        tgt_h, tgt_w = min(self.resolution_list, key=lambda r: abs((r[1] / r[0]) - (w / h)))
        scale = max(tgt_h / h, tgt_w / w)
        new_h, new_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        new_h = max(new_h, tgt_h)
        new_w = max(new_w, tgt_w)
        sh = max(0, (new_h - tgt_h) // 2)
        sw = max(0, (new_w - tgt_w) // 2)
        for i in range(len(pixel_values)):
            frame = pixel_values[i]
            resized_frame = cv2.resize(frame, (new_w, new_h))
            cropped_frame = resized_frame[sh:sh + tgt_h, sw:sw + tgt_w]
            processed_frames.append(cropped_frame)
        pixel_values = torch.from_numpy(np.stack(processed_frames, axis=0))  # (T, H, W, C)
        pixel_values_for_vit = pixel_values[::self.vit_sample_stride].permute(0, 3, 1, 2).contiguous()
        vit_values, gird_thw = self.processor.video_processor.preprocess(pixel_values_for_vit, do_sample_frames=False).values()

        if not self.enable_bucket:
            pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 127.5 - 1.0  # [-1, 1]
            del video_reader
        else:
            pixel_values = self.video_transforms(pixel_values)

        return pixel_values, vit_values, gird_thw
        

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        if isinstance(data_info, dict):
            if 'video_path' in data_info:
                video_path = data_info['video_path'].replace('/mnt/yifanyang/', '/blob/')
                pixel_values, vit_values, grid_thw = self.read_video_frames(video_path, idx)
                text = random.choice([data_info['caption'], data_info['short_caption']])
            elif 'image_path' in data_info:
                print("Image Data Not Implemented Yet ...")
                return None, None, 'image', None
            else:
                print("Unsupported data_info dict type:", data_info)

        elif isinstance(data_info, (str, np.str_)):
            if data_info.endswith('.json'):
                # with open(data_info, 'r') as f:
                with open(data_info.replace('/blob', '/mnt/yifanyang'), 'r') as f:
                    zeyuan_data = json.load(f)
                video_path = zeyuan_data["video_path"].replace('/blob', '/mnt/yifanyang')
                # video_path = zeyuan_data["video_path"]
                pixel_values, vit_values, grid_thw = self.read_video_frames(video_path, idx)
                text = zeyuan_data['caption']
            elif data_info.endswith('.npy'):
                zehui_data = np.load(data_info, allow_pickle=True).item()
                video_path = zehui_data["video_id"]
                video_data = zehui_data["video"]
                pixel_values, vit_values, grid_thw = self.read_video_frames(video_data, idx)
                text = zehui_data["instruction"]
            else:
                print("Unsupported data_info string type:", data_info)

        return pixel_values, vit_values, grid_thw, text, 'video', video_path


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                pixel_values, vit_values, grid_thw, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["vit_values"] = vit_values
                sample["grid_thw"] = grid_thw
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


if __name__ == "__main__":
    from tqdm import tqdm
    dataset = ImageVideoDataset(
        add_zehui_data=True, add_zeyuan_data=True, add_llava_video_data=True, add_image_data=True, show_data_structure=True,
        video_sample_stride=1, video_sample_n_frames=33, vit_sample_stride=2, enable_bucket=False,
    )
    for i in tqdm(range(len(dataset))):
        if i > 100:
            break
        sample = dataset[i]
        print(
            f"[Sample]\n"
            f"  data_type:    {sample['data_type']}\n"
            f"  pixel_values: shape={sample['pixel_values'].shape}, "
            f"min={sample['pixel_values'].min():.3f}, max={sample['pixel_values'].max():.3f}\n"
            f"  vit_values:   shape={sample['vit_values'].shape}, "
            f"min={sample['vit_values'].min():.3f}, max={sample['vit_values'].max():.3f}\n"
            f"  grid_thw:     {sample['grid_thw']}\n"
            f"  text:         {sample['text']}\n"
        )
