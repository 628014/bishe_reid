#!/bin/bash
# 脚本功能：依次执行两个训练脚本，第一个中断不影响第二个
# 作者：自定义
# 日期：2026-01-26

# 定义脚本路径（避免手动输入错误）
# SCRIPT1="/home/wangrui/code/MLLM4Text-ReID-main/finetune_wr_icfg.sh"
SCRIPT2="/home/wangrui/code/MLLM4Text-ReID-main/finetune_wr_cuhk.sh"

# 执行第一个脚本，即使失败也返回true（确保后续执行）
echo "===== 开始执行第一个训练脚本：finetune_wr_icfg.sh ====="
bash $SCRIPT1 || true  # || true 是核心：第一个脚本失败时强制返回成功

# 执行第二个脚本（不受第一个脚本影响）
echo -e "\n===== 开始执行第二个训练脚本：finetune_wr_cuhk.sh ====="
bash $SCRIPT2

# 输出执行完成提示
echo -e "\n===== 所有训练脚本执行完成 ====="