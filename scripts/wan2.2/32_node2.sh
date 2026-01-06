export MODEL_NAME="/blob/dyb/pretrained_ckpts/Wan2.2-TI2V-5B"
# export MODEL_NAME="/home/v-yanboding/dit_training/ckpt/Wan2.2"
export VQ_PATH="/blob/dyb_output/icml2026/single_codebook_ema/checkpoint-55399/model.safetensors"
export DATA_PATH="/blob/dyb/processed_data"
export OUTPUT="/blob/dyb_output/icml2026/dit_continuous"
NCCL_DEBUG=INFO

export WANDB_PROJECT="icml_2026_dit_ablation"

accelerate launch \
  --use_deepspeed \
  --zero_stage 2 \
  --num_machines 4 \
  --machine_rank 2 \
  --main_process_ip 100.64.34.183 \
  --main_process_port 29500 \
  --num_processes 32 \
  --deepspeed_config_file config/zero_stage2_config.json \
  --deepspeed_multinode_launcher standard \
  scripts/wan2.2/train.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_PATH \
  --vq_model_path=$VQ_PATH \
  --video_sample_stride=1 \
  --vit_sample_stride=2 \
  --video_sample_n_frames=121 \
  --resolution_list "(384,384)" "(288,512)" "(512,288)" \
  --train_batch_size=1 \
  --resume_from_checkpoint "latest" \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=1 \
  --checkpointing_steps=5000 \
  --learning_rate=2e-05 \
  --lr_scheduler="cosine" \
  --lr_warmup_ratio=0.03 \
  --seed=42 \
  --output_dir=$OUTPUT \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="normal" \
  --trainable_modules "." \
  --report_to wandb

python /blob/thinking.py
