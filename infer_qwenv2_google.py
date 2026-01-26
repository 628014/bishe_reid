import os
import argparse
import torch
from utils.logger import setup_logger
from datasets.build import build_dataloader_qwen
from model.build_finetune_qwen_google import build_finetune_model_qwen
from utils.metrics_qwen import Evaluator
import os.path as op


def main():
    parser = argparse.ArgumentParser(description="Infer using finetuned Qwen2.5-VL-7B as backbone for ReID")
    parser.add_argument("--qwen_model_path", default='/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_with_score')
    parser.add_argument("--checkpoint", default=None, help='optional checkpoint for additional weights')
    parser.add_argument("--dataset_name", default='RSTPReid', help='dataset name: CUHK-PEDES, ICFG-PEDES, RSTPReid')
    parser.add_argument("--root_dir", default='/home/wangrui/code/MLLM4Text-ReID-main/data', help='root directory for datasets')
    parser.add_argument("--test_batch_size", type=int, default=8, help="Reduce this if OOM occurs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default='./logs/qwen_infer')
    
    # IRRA config compatibility
    parser.add_argument("--img_size", default=(384, 128)) 
    parser.add_argument("--text_length", default=77) # Not used by Qwen processor but required by dataset args
    
    args = parser.parse_args()

    # Args setup for compat
    args.training = False
    
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(f"Arguments: {vars(args)}")

    # 1. Build DataLoaders
    # 返回: Raw PIL Images 和 Raw Text Strings
    # Qwen 需要原始数据进行自己的 Processor 处理
    test_img_loader, test_txt_loader, num_classes = build_dataloader_qwen(args)

    # 2. Build Model
    logger.info(f"Building Model from {args.qwen_model_path} ...")
    # 注意：确保 build_finetune_model_qwen.py 已经应用了我们之前讨论的 NoneType 修复
    model = build_finetune_model_qwen(args, num_classes=num_classes)

    # 3. Load Checkpoint (Optional)
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        try:
            ck = torch.load(args.checkpoint, map_location='cpu')
            msg = model.load_state_dict(ck, strict=False)
            logger.info(f"Checkpoint loaded: {msg}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {args.checkpoint}: {e}")

    # Move model to device (Qwen usually handles this with device_map="auto", but just in case)
    if torch.cuda.is_available():
        # 如果模型没有自动分配，我们手动分配。
        # 通常 QwenVLReID 里的 device_map="auto" 已经把 LLM 放到了 GPU
        pass 

    # 4. Evaluation
    logger.info("Start Evaluation...")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    
    # 执行评估
    # 结果是一个字典: {'R1': ..., 'R5': ..., 'mAP': ..., 'mINP': ...}
    results = evaluator.eval(model, i2t_metric=False)
    
    # 最终输出
    print(f"\nFinal Results on {args.dataset_name}:")
    print(f"R1  : {results['R1']:.2f}%")
    print(f"R5  : {results['R5']:.2f}%")
    print(f"R10 : {results['R10']:.2f}%")
    print(f"mAP : {results['mAP']:.2f}%")
    print(f"mINP: {results['mINP']:.2f}%")


if __name__ == '__main__':
    main()