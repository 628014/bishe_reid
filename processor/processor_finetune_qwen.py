import collections
import logging
import time
import torch
from utils.metrics_qwen import EvaluatorQwen as Evaluator
from utils.comm import get_rank, synchronize
from prettytable import PrettyTable
import torch.nn.functional as F


def do_inference(model, test_img_loader, test_txt_loader):
    """Qwen模型推理评估，复用原始指标计算"""
    logger = logging.getLogger("IRRA-Qwen.test")
    logger.info("Qwen Model Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval(), i2t_metric=True)
    return top1


def evaluate_model(model, test_img_loaders, test_txt_loaders, dataset_names):
    """评估模型在多个数据集上的性能"""
    logger = logging.getLogger("IRRA-Qwen.evaluate")
    logger.info("Starting Qwen Model Evaluation on multiple datasets...")
    
    results = {}
    
    for i, (test_img_loader, test_txt_loader, dataset_name) in enumerate(zip(test_img_loaders, test_txt_loaders, dataset_names)):
        logger.info(f"Evaluating on {dataset_name}...")
        evaluator = Evaluator(test_img_loader, test_txt_loader)
        
        with torch.no_grad():
            top1 = evaluator.eval(model.eval(), i2t_metric=True)
            results[dataset_name] = top1
    
    # 打印汇总结果
    logger.info("\n=== Qwen Model Evaluation Results ===")
    table = PrettyTable(["Dataset", "Rank-1"])
    for dataset_name, top1 in results.items():
        table.add_row([dataset_name, top1])
    logger.info("\n" + str(table))
    
    return results
