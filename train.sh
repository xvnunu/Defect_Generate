#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 \
# python examples/controlnet/train_controlnet.py \
#   --train_data_dir="defect" \
#   --image_column="image" \
#   --conditioning_image_column="mask" \
#   --caption_column="captions" \
#   --output_dir="./output/controlnet-defect" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=2 \
#   --learning_rate=1e-5 \
#   --lr_scheduler="constant" \
#   --num_train_epochs=10 \
#   --report_to="tensorboard" \
#   --checkpointing_steps=500 \
#   --validation_image="./validation/example.png" \
#   --validation_prompt="a surface with spot defects" \
#   --controlnet_model_name_or_path="lllyasviel/sd-controlnet-canny" \
#   --tracker_project_name="defect-controlnet" \
#   --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5"


# CUDA_VISIBLE_DEVICES=0 \
# export CUDA_VISIBLE_DEVICES=0
# python examples/controlnet/train_controlnet.py \
#   --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
#   --output_dir="./output/controlnet-defect" \
#   --dataset_name=fusing/fill50k \
#   --resolution=512 \
#   --learning_rate=1e-5 \
#   --validation_image "./validation/example.png" \
#   --validation_prompt "a surface with spot defects" \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --enable_xformers_memory_efficient_attention \
#   --set_grads_to_none \
#   --mixed_precision fp16

# accelerate launch examples/controlnet/train_controlnet.py \
#   --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
#   --output_dir="./output/controlnet-defect-train" \
#   --dataset_name=defect \
#   --resolution=512 \
#   --learning_rate=1e-5 \
#   --validation_image "./validation/example.png" \
#   --validation_prompt "a surface with spot defects" \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --enable_xformers_memory_efficient_attention \
#   --set_grads_to_none \
#   --use_8bit_adam \
#   --mixed_precision fp16

# accelerate launch custom_controlnet/train_inpainting.py \
#   --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
#   --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_inpaint" \
#   --output_dir="./output/controlnet-inpainting-train-new" \
#   --dataset_name=defect \
#   --resolution=512 \
#   --learning_rate=1e-5 \
#   --validation_image "./output/test/control_image.png" \
#   --validation_prompt "a surface with dent defects" \
#   --train_batch_size=1 \
#   --num_train_epochs=30 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --enable_xformers_memory_efficient_attention \
#   --set_grads_to_none \
#   --mixed_precision fp16 \
#   --report_to wandb


CUDA_VISIBLE_DEVICES=0 \
python xr_src/train_seg.py \
  --pretrained_model_name_or_path="pretrained_model/stable-diffusion-v1-5/stable-diffusion-inpainting" \
  --controlnet_model_name_or_path="pretrained_model/lllyasviel/control_v11p_sd15_inpaint" \
  --output_dir="./output/controlnet-inpainting-segment-only-scratch" \
  --dataset_name=croped_scratch_defect \
  --resolution=512 \
  --learning_rate=1e-5 \
  --validation_image "validation/defect (19).jpg" \
  --validation_mask "validation/example.png" \
  --validation_prompt "a surface with dent defects" \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --mixed_precision fp16 \
  --validation_steps 50 \
  --report_to wandb