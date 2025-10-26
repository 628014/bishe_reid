#!/bin/bash
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python finetune.py \
--name finetune \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id+mlm' \
--num_epoch 60 \
--root_dir /home/wangrui/code/MLLM4Text-ReID-main/data \
--finetune /home/wangrui/code/MLLM4Text-ReID-main/checkpoint/best2.pth
