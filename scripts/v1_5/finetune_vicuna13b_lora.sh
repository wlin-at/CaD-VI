#!/bin/bash

bz=32
output_folder_name=job_name
output_dir=/output/dir/$output_folder_name
gpu_list=0,1,2,3

mkdir -p $output_dir


WANDB_API_KEY=xxx \
WANDB_PROJECT=xxx \
WANDB_ENTITY=xxx \
WANDB_DIR=$output_dir \
WANDB_NAME=$output_folder_name \
deepspeed --include localhost:$gpu_list llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /path/to/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /path/to/llava_v1_5_mix665k.json,/path/to/phase1_instruct_data_278k.json,/path/to/phase1_instruct_data_278k.json,/path/to/phase2_instruct_data_71k.json\
    --image_folder /path/to/llava_imgs,/path/to/localized_narratives_imgs,/path/to/COCO_train_2017_imgs \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /path/to/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${bz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --loss_type clm
