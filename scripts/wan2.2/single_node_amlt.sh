export MODEL_NAME="/blob/dyb/pretrained_ckpts/Wan2.2-TI2V-5B"
export VQ_PATH="/blob/dyb_output/qwen3_vl_vq_eam/checkpoint-180000/model.safetensors"
export OUTPUT="/blob/dyb_output/test"
NCCL_DEBUG=INFO

export WANDB_PROJECT="train_qwen3_vl_dit_1127"

accelerate launch \
  --use_deepspeed \
  --zero_stage 3 \
  --deepspeed_config_file config/zero_stage3_config_cpu_offload.json \
  --deepspeed_multinode_launcher standard \
  scripts/wan2.2/train.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --vq_model_path=$VQ_PATH \
  --add_zehui_data \
  --add_zeyuan_data \
  --show_data_structure \
  --video_sample_stride=1 \
  --vit_sample_stride=2 \
  --video_sample_n_frames=33 \
  --resolution_list "(480,704)" "(704,480)" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=10 \
  --checkpointing_steps=5000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=300 \
  --seed=42 \
  --output_dir=$OUTPUT \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --low_vram \
  --boundary_type="full" \
  --train_mode="normal" \
  --trainable_modules "." \
  --report_to wandb