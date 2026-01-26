#!/bin/bash
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python finetune_match.py \
--name finetune_match \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id+mlm+match' \
--num_epoch 60 \
--root_dir /home/wangrui/code/MLLM4Text-ReID-main/data \
--finetune /home/wangrui/code/MLLM4Text-ReID-main/checkpoint/best2.pth