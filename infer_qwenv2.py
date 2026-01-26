import os
import argparse
import torch
from utils.logger import setup_logger
from datasets.build import build_dataloader_qwen
from model.build_finetune_qwen import build_finetune_model_qwen
from utils.metrics_qwen import Evaluator
import os.path as op


def main():
    parser = argparse.ArgumentParser(description="Infer using finetuned Qwen2.5-VL-7B as backbone for ReID")
    parser.add_argument("--qwen_model_path", default='/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_with_score')
    parser.add_argument("--checkpoint", default=None, help='optional checkpoint for additional weights (not required if qwen path contains finetuned adapter)')
    parser.add_argument("--dataset_name", default='RSTPReid', help='dataset factory name, e.g. CUHK-PEDES, ICFG-PEDES, RSTPReid')
    parser.add_argument("--root_dir", default='/home/wangrui/code/MLLM4Text-ReID-main/data', help='root directory for datasets')
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default='./logs/qwen_infer')
    args = parser.parse_args()

    # Minimal args object expected by dataset builder
    args.training = False
    args.dataset_name = args.dataset_name
    args.root_dir = args.root_dir
    args.test_batch_size = args.test_batch_size
    args.num_workers = args.num_workers
    args.output_dir = args.output_dir

    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build qwen-compatible dataloaders (raw PIL images and raw caption strings)
    test_img_loader, test_txt_loader, num_classes = build_dataloader_qwen(args)

    # build model wrapper that embeds qwen model and exposes encode_image/encode_text
    model = build_finetune_model_qwen(args, num_classes=num_classes)

    # if user provided additional checkpoint (torch .pth), try load
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        try:
            ck = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(ck, strict=False)
            logger.info(f"Loaded checkpoint {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {args.checkpoint}: {e}")

    # move classifier/other parts to device; Qwen internal model likely used device_map, rely on that
    try:
        model.to(device)
    except Exception:
        # model may have its internal qwen model already on devices via device_map, ensure classifier on device
        for n, p in model.named_parameters():
            if p.device.type == 'cpu' and p.requires_grad:
                p.data = p.data.to(device)

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    # evaluate (only t2i by default)
    top1 = evaluator.eval(model, i2t_metric=False)
    print(f"T2I Rank-1: {top1}")


if __name__ == '__main__':
    main()
