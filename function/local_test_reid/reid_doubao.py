import os
import re
import torch
import numpy as np
import json
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer
import sys

# 添加上级目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# 从bases.py导入tokenize函数（注意其参数定义）
from datasets.bases import tokenize

def get_image_transforms(img_size):
    if isinstance(img_size, tuple):
        resize_size = (int(img_size[0]), int(img_size[1]))
    else:
        resize_size = (int(img_size), int(img_size))
    
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def load_rstpreid_test_data(img_dir, anno_path):
    """仅加载split为test的图像数据"""
    with open(anno_path, 'r') as f:
        annos = json.load(f)
    
    test_images = []
    test_captions = []
    test_pids = []
    test_conut = 0
    for anno in annos:
        if anno.get('split', '').lower() == 'test':
            img_path = os.path.join(img_dir, anno['img_path'])
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0 and test_conut < 100:
                test_images.append(img_path)
                test_pids.append(int(anno['id']))
                test_captions.append(anno['captions'])
                test_conut += 1
    
    print(f"已加载测试集图像: {len(test_images)} 张")
    return {
        'img_paths': test_images,
        'captions': test_captions,
        'pids': test_pids
    }

def show_retrieval_results(query, results, top_k=5, is_query_image=True):
    plt.figure(figsize=(15, 5))
    
    if is_query_image:
        try:
            plt.subplot(1, top_k+1, 1)
            img = Image.open(query).resize((128, 256))
            plt.imshow(img)
            plt.title("Query Image")
        except (FileNotFoundError, IOError):
            # 如果无法打开图像，显示为文本
            plt.subplot(1, top_k+1, 1)
            plt.text(0.5, 0.5, str(query)[:50] + "..." if len(str(query)) > 50 else str(query), 
                    ha='center', va='center', wrap=True)
            plt.axis('off')
            plt.title("Query")
    else:
        plt.subplot(1, top_k+1, 1)
        plt.text(0.5, 0.5, str(query), ha='center', va='center', wrap=True)
        plt.axis('off')
        plt.title("Query Text")
    
    for i, result_path in enumerate(results[:top_k]):
        plt.subplot(1, top_k+1, i+2)
        try:
            img = Image.open(result_path).resize((128, 256))
            plt.imshow(img)
            plt.title(f"Top {i+1}")
        except (FileNotFoundError, IOError):
            plt.text(0.5, 0.5, f"无法加载\n图像 {i+1}", ha='center', va='center')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.show()

class IRRADemo:
    def __init__(self, config_file, img_dir, anno_path):
        self.config_file = config_file
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载配置
        self.args = load_train_configs(config_file)
        self.args.training = False
        
        # 构建模型
        self.model = self._build_model()
        
        # 文本处理：获取tokenizer实例（原项目中应已初始化）
        
        self.tokenizer = SimpleTokenizer()  # 创建SimpleTokenizer实例
        
        # 仅加载测试集数据
        self.test_dataset = load_rstpreid_test_data(img_dir, anno_path)
        self.image_transform = get_image_transforms(self.args.img_size)
        
        # 预计算测试集图像特征
        self.test_image_features = self._precompute_test_image_features()
        print(f"完成测试集图像特征预计算，共 {len(self.test_image_features)} 张图像")

    def _build_model(self):
        from datasets import build_dataloader
        _, _, num_classes = build_dataloader(self.args)
        model = build_model(self.args, num_classes=num_classes)
        
        # 加载权重
        # checkpoint_path = '/home/wangrui/code/MLLM4Text-ReID-main/logs/RSTPReid_1023/20251023_234421_finetune/best2.pth'
        checkpoint_path = '/home/wangrui/code/MLLM4Text-ReID-main/checkpoint/best2.pth'
        assert os.path.exists(checkpoint_path), f"模型权重文件不存在: {checkpoint_path}"
        checkpointer = Checkpointer(model)
        checkpointer.load(f=checkpoint_path)
        
        model.to(self.device)
        model.eval()
        return model

    def _precompute_test_image_features(self):
        print("正在预计算测试集图像特征...")
        image_features = []
        
        with torch.no_grad():
            for idx, img_path in enumerate(self.test_dataset['img_paths']):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
                    feat = self.model.encode_image(img_tensor)
                    image_features.append(feat.cpu())
                    
                    if (idx + 1) % 50 == 0:
                        print(f"已处理测试集图像 {idx + 1}/{len(self.test_dataset['img_paths'])}")
                except Exception as e:
                    print(f"处理测试集图像 {img_path} 失败: {e}，已跳过")
        
        if not image_features:
            raise ValueError("未加载到有效测试集图像特征，请检查标注文件和图像路径")
        
        return F.normalize(torch.cat(image_features, dim=0), p=2, dim=1)

    def image_to_image_retrieval(self, query_img_path, top_k=5):
        try:
            with torch.no_grad():
                img = Image.open(query_img_path).convert('RGB')
                img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
                query_feat = self.model.encode_image(img_tensor)
                query_feat = F.normalize(query_feat, p=2, dim=1).cpu()
            
            similarity = torch.mm(query_feat, self.test_image_features.t()).squeeze()
            _, indices = torch.topk(similarity, k=top_k)
            
            return [self.test_dataset['img_paths'][i] for i in indices]
        except Exception as e:
            print(f"纯图像检索失败: {e}")
            return []

    def text_to_image_retrieval(self, query_text, top_k=5):
        
        try:
            with torch.no_grad():
                # 调用bases.py中的tokenize函数，参数严格匹配：(caption, tokenizer, text_length)
                text_tensor = tokenize(
                    caption=query_text,  # 第一个参数：文本字符串
                    tokenizer=self.tokenizer  # 第二个参数：tokenizer实例
                ).unsqueeze(0).to(self.device)  # 增加batch维度
                
                text_feat = self.model.encode_text(text_tensor)
                text_feat = F.normalize(text_feat, p=2, dim=1).cpu()
            
            similarity = torch.mm(text_feat, self.test_image_features.t()).squeeze()
            _, indices = torch.topk(similarity, k=top_k)
            
            return [self.test_dataset['img_paths'][i] for i in indices]
        except Exception as e:
            print(f"纯文本检索失败: {e}")
            return []

    def image_text_fusion_retrieval(self, query_img_path, query_text, 
                                   img_weight=0.5, text_weight=0.5, top_k=5):
        """融合检索的文本处理同样修正参数"""
        try:
            with torch.no_grad():
                # 图像特征
                img = Image.open(query_img_path).convert('RGB')
                img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
                img_feat = self.model.encode_image(img_tensor)
                img_feat = F.normalize(img_feat, p=2, dim=1).cpu()
                
                # 文本特征（修正参数传递）
                text_tensor = tokenize(
                    caption=query_text,
                    tokenizer=self.tokenizer,
                ).unsqueeze(0).to(self.device)
                text_feat = self.model.encode_text(text_tensor)
                text_feat = F.normalize(text_feat, p=2, dim=1).cpu()
            
            img_similarity = torch.mm(img_feat, self.test_image_features.t()).squeeze()
            text_similarity = torch.mm(text_feat, self.test_image_features.t()).squeeze()
            fused_similarity = img_weight * img_similarity + text_weight * text_similarity
            
            _, indices = torch.topk(fused_similarity, k=top_k)
            return [self.test_dataset['img_paths'][i] for i in indices]
        except Exception as e:
            print(f"融合检索失败: {e}")
            return []

if __name__ == "__main__":
    # 配置路径
    config_file = "/home/wangrui/code/MLLM4Text-ReID-main/logs/RSTPReid_1023/20251023_234421_finetune/configs.yaml"
    img_dir = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/"
    anno_path = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions.json"
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Text ReID Demo')
    parser.add_argument('--img_name', type=str, default='', help='要查询的图像文件名（不含路径）')
    parser.add_argument('--text_idx', type=int, default=0, help='对应图像的文本描述索引（0-4）')
    args = parser.parse_args()
    
    # 初始化演示器
    try:
        demo = IRRADemo(config_file, img_dir, anno_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        exit(1)
    
    # 示例查询
    if demo.test_dataset['img_paths']:
        # 通过文件名确定sample_img和sample_text
        sample_img = None
        sample_text = "a person in the image"
        
        if args.img_name:
            # 查找匹配的文件名
            for idx, img_path in enumerate(demo.test_dataset['img_paths']):
                if os.path.basename(img_path) == args.img_name:
                    sample_img = img_path
                    if (demo.test_dataset['captions'] and len(demo.test_dataset['captions']) > idx and 
                        demo.test_dataset['captions'][idx] and len(demo.test_dataset['captions'][idx]) > args.text_idx):
                        sample_text = demo.test_dataset['captions'][idx][args.text_idx]
                    break
        print(f"查询图像: {sample_img}, 文本描述: {sample_text}")
        # 如果没找到匹配的文件或未提供文件名，则使用默认索引
        if sample_img is None:
            print("未找到指定的图像文件，使用默认样本")
            sample_img = demo.test_dataset['img_paths'][0]
            sample_text = demo.test_dataset['captions'][0][2] if (demo.test_dataset['captions'] and demo.test_dataset['captions'][0]) else "a person in the image"
        
        # 纯图像检索
        print("=== 纯图像检索结果 ===")
        img_results = demo.image_to_image_retrieval(sample_img, top_k=5)
        if img_results:
            show_retrieval_results(sample_img, img_results, is_query_image=True)
        
        # 纯文本检索
        print(f"=== 纯文本检索结果 (查询: {sample_text}) ===")
        text_results = demo.text_to_image_retrieval(sample_text, top_k=5)
        if text_results:
            show_retrieval_results(sample_text, text_results, is_query_image=False)
        
        # 融合检索
            print("=== 图像+文本融合检索结果 ===")
            fusion_results = demo.image_text_fusion_retrieval(
                sample_img, sample_text, 
                img_weight=0.6, 
                text_weight=0.4, 
                top_k=5
            )
            if fusion_results:
                # 融合检索显示文本查询，设置is_query_image=False
                show_retrieval_results(
                    f"Image + Text\n({sample_text}...)", 
                    fusion_results, 
                    is_query_image=False
                )
    else:
        print("未加载到测试集图像，请检查标注文件中是否有'split: test'的样本")