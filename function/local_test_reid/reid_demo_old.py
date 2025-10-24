import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# 添加项目根目录到Python路径
import sys
sys.path.append('/home/wangrui/code/MLLM4Text-ReID-main')

from model.build import IRRA
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer

class ReIDDemo:
    def __init__(self, config_path, model_path):
        # 加载配置文件
        self.args = load_train_configs(config_path)
        self.args.training = False
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 构建模型
        self.model = IRRA(self.args)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 去除可能的模块前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                               [0.26862954, 0.26130258, 0.27577711])
        ])
        
        # 初始化tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # 加载数据集信息
        self.data_info = self._load_data_info()
        
        # 温度参数，与processor.py保持一致
        self.logit_scale = torch.ones([]) * (1 / self.args.temperature)
        
    def _load_data_info(self):
        """加载数据信息"""
        data_path = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _encode_image(self, image_path):
        """编码图像获取特征"""
        # 构建完整的图像路径
        root_dir = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid'
        full_path = os.path.join(root_dir, 'imgs', image_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"图像文件不存在: {full_path}")
        
        try:
            # 图像预处理
            image = Image.open(full_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            # 为了与model(image, text, ori_text)格式匹配，需要生成text和ori_text
            # 这里使用空文本作为占位符
            empty_text = [49406, 49407]  # <start> + <end>
            text_tensor = torch.tensor(empty_text, dtype=torch.long).unsqueeze(0).to(self.device)
            ori_text_tensor = text_tensor.clone()
            
            with torch.no_grad():
                i_feats, text_feats, fu_i_feats, fu_t_feats = self.model(image, text_tensor, ori_text_tensor)
            
            # 提取第一个token的特征并归一化
            image_feat = i_feats[:, 0, :]
            image_feat = F.normalize(image_feat, dim=-1)
            
            return image_feat
        except Exception as e:
            print(f"图像编码错误: {e}")
            batch_size = 1
            embed_dim = self.model.embed_dim if hasattr(self.model, 'embed_dim') else 512
            random_feat = torch.randn(batch_size, embed_dim).to(self.device)
            return F.normalize(random_feat, dim=-1)
    
    def _encode_text(self, text):
        """编码文本获取特征，按照IRRA类中的方式获取特征"""
        try:
            # 使用SimpleTokenizer进行文本编码
            tokens = self.tokenizer.encode(text)
            
            # 准备输入张量，格式化为模型所需的长度
            context_length = self.args.text_length
            text_ids = torch.zeros(context_length, dtype=torch.long).to(self.device)
            
            # 添加开始token (49406是<|startoftext|>的ID)
            text_ids[0] = 49406
            
            # 填充文本token，确保不超过上下文长度
            token_len = min(len(tokens), context_length - 2)  # 留出开始和结束token的位置
            text_ids[1:1+token_len] = torch.tensor(tokens[:token_len], dtype=torch.long).to(self.device)
            
            # 如果有剩余空间，添加结束token (49407是<|endoftext|>的ID)
            if 1 + token_len < context_length:
                text_ids[1 + token_len] = 49407
            
            # 添加batch维度
            text_tensor = text_ids.unsqueeze(0)
            
            # ori_text与text相同
            ori_text_tensor = text_tensor.clone()
            
            # 为了与model(image, text, ori_text)格式匹配，需要生成空图像作为占位符
            # 创建一个空的图像tensor [1, 3, 224, 224]
            dummy_image = torch.zeros(1, 3, self.args.img_size, self.args.img_size).to(self.device)
            
            with torch.no_grad():
                # 使用model(image, text, ori_text)获取特征
                i_feats, text_feats, fu_i_feats, fu_t_feats = self.model(dummy_image, text_tensor, ori_text_tensor)
            
            # 按照processor.py中的方式提取文本特征
            # 找到结束token的位置并提取对应的特征
            caption_ids = text_tensor
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            
            # 归一化特征
            t_feats = F.normalize(t_feats, dim=-1)
            
            return t_feats
        except Exception as e:
            print(f"文本编码错误: {e}")
            # 错误处理
            batch_size = 1
            embed_dim = self.model.embed_dim if hasattr(self.model, 'embed_dim') else 512
            random_feat = torch.randn(batch_size, embed_dim).to(self.device)
            return F.normalize(random_feat, dim=-1)
    
    def _get_all_test_images(self):
        """获取所有测试图像的路径和ID"""
        test_images = []
        for item in self.data_info:
            if item['split'] == 'test':
                test_images.append({
                    'id': item['id'],
                    'img_path': item['img_path']
                })
        return test_images
    
    def search_similar_images(self, query_image_path, text_query=None, top_k=5, img_weight=0.5, text_weight=0.5):
        """搜索相似图像，按照IRRA类中的方式计算相似度
        
        Args:
            query_image_path: 查询图像路径
            text_query: 文本查询（可选）
            top_k: 返回前k个结果
            img_weight: 图像特征权重
            text_weight: 文本特征权重
            
        Returns:
            top_results: 排序后的结果列表，每个元素为 (image_path, person_id, similarity)
        """
        try:
            # 获取查询图像的绝对路径
            if not os.path.isabs(query_image_path):
                root_dir = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid'
                query_image_path = os.path.join(root_dir, query_image_path)
            
            # 确保权重正常化
            total_weight = img_weight + text_weight
            img_weight = img_weight / total_weight if total_weight > 0 else 0.5
            text_weight = text_weight / total_weight if total_weight > 0 else 0.5
            
            print(f"正在搜索相似图像...")
            print(f"- 查询图像: {query_image_path}")
            print(f"- 文本查询: {text_query}")
            print(f"- 权重设置: 图像={img_weight:.2f}, 文本={text_weight:.2f}")
            
            # 获取所有测试图像
            all_images = self._get_all_test_images()
            
            # 预处理所有图像特征和文本特征（如果需要）
            if not hasattr(self, '_cached_feats') or len(self._cached_feats) != len(all_images):
                print("正在预处理所有图像特征...")
                self._cached_feats = []
                for img_info in all_images:
                    try:
                        img_path = img_info['img_path']
                        # 构建完整的图像路径
                        full_path = os.path.join('/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs', img_path)
                        
                        # 加载图像
                        image = Image.open(full_path).convert('RGB')
                        image = self.transform(image).unsqueeze(0).to(self.device)
                        
                        # 使用空文本作为占位符
                        empty_text = [49406, 49407]  # <start> + <end>
                        text_tensor = torch.tensor(empty_text, dtype=torch.long).unsqueeze(0).to(self.device)
                        ori_text_tensor = text_tensor.clone()
                        
                        # 获取特征
                        with torch.no_grad():
                            i_feats, text_feats, fu_i_feats, fu_t_feats = self.model(image, text_tensor, ori_text_tensor)
                        
                        # 存储图像特征（第一个token）
                        self._cached_feats.append({
                            'i_feats': i_feats,
                            'fu_i_feats': fu_i_feats,
                            'img_path': img_path,
                            'id': img_info['id']
                        })
                    except Exception as e:
                        print(f"预处理图像 {img_info['img_path']} 时出错: {e}")
                        self._cached_feats.append(None)
                print(f"预处理完成，共 {len(self._cached_feats)} 个图像特征")
            
            # 处理查询
            query_feats = None
            query_fu_feats = None
            
            # 如果有文本查询，处理文本特征
            if text_query:
                try:
                    # 编码文本
                    tokens = self.tokenizer.encode(text_query)
                    
                    # 准备输入张量
                    context_length = self.args.text_length
                    text_ids = torch.zeros(context_length, dtype=torch.long).to(self.device)
                    text_ids[0] = 49406  # <start>
                    
                    # 填充文本token
                    token_len = min(len(tokens), context_length - 2)
                    text_ids[1:1+token_len] = torch.tensor(tokens[:token_len], dtype=torch.long).to(self.device)
                    
                    # 添加结束token
                    if 1 + token_len < context_length:
                        text_ids[1 + token_len] = 49407  # <end>
                    
                    # 添加batch维度
                    text_tensor = text_ids.unsqueeze(0)
                    ori_text_tensor = text_tensor.clone()
                    
                    # 创建空图像占位符
                    dummy_image = torch.zeros(1, 3, self.args.img_size, self.args.img_size).to(self.device)
                    
                    # 获取文本特征
                    with torch.no_grad():
                        _, text_feats, _, fu_t_feats = self.model(dummy_image, text_tensor, ori_text_tensor)
                    
                    # 提取文本特征（结束token位置）
                    t_feats = text_feats[torch.arange(text_feats.shape[0]), text_tensor.argmax(dim=-1)].float()
                    query_feats = F.normalize(t_feats, dim=-1)
                    query_fu_feats = F.normalize(fu_t_feats, dim=-1)
                    
                except Exception as e:
                    print(f"处理文本查询时出错: {e}")
            
            # 处理查询图像
            if query_image_path:
                try:
                    # 加载查询图像
                    query_image = Image.open(query_image_path).convert('RGB')
                    query_image = self.transform(query_image).unsqueeze(0).to(self.device)
                    
                    # 使用空文本作为占位符
                    empty_text = [49406, 49407]  # <start> + <end>
                    text_tensor = torch.tensor(empty_text, dtype=torch.long).unsqueeze(0).to(self.device)
                    ori_text_tensor = text_tensor.clone()
                    
                    # 获取查询图像特征
                    with torch.no_grad():
                        i_feats, _, fu_i_feats, _ = self.model(query_image, text_tensor, ori_text_tensor)
                    
                    # 提取图像特征（第一个token）
                    i_feats = i_feats[:, 0, :].float()
                    
                    # 如果没有文本查询，使用图像特征作为查询特征
                    if query_feats is None:
                        query_feats = F.normalize(i_feats, dim=-1)
                        query_fu_feats = F.normalize(fu_i_feats, dim=-1)
                    else:
                        # 加权组合图像和文本特征
                        img_feats_norm = F.normalize(i_feats, dim=-1)
                        query_feats = img_weight * img_feats_norm + text_weight * query_feats
                        query_feats = F.normalize(query_feats, dim=-1)
                        
                        # 组合融合特征
                        img_fu_feats_norm = F.normalize(fu_i_feats, dim=-1)
                        query_fu_feats = img_weight * img_fu_feats_norm + text_weight * query_fu_feats
                        query_fu_feats = F.normalize(query_fu_feats, dim=-1)
                except Exception as e:
                    print(f"处理查询图像时出错: {e}")
            
            # 如果没有查询特征，返回空结果
            if query_feats is None:
                print("无法提取查询特征，搜索失败")
                return []
            
            # 计算相似度
            similarities = []
            for i, cached_feat in enumerate(self._cached_feats):
                if cached_feat is not None:
                    # 获取图像特征
                    i_feats = cached_feat['i_feats'][:, 0, :].float()
                    i_feats_norm = F.normalize(i_feats, dim=-1)
                    
                    # 计算基础相似度
                    base_sim = torch.matmul(query_feats, i_feats_norm.T).item()
                    
                    # 计算融合特征相似度（类似processor.py中的sdm_loss计算）
                    fu_i_feats_norm = F.normalize(cached_feat['fu_i_feats'], dim=-1)
                    fusion_sim = torch.einsum('nld,nkd->nlk', [query_fu_feats, fu_i_feats_norm[:, 1:, :]])
                    fusion_sim = fusion_sim.max(-1)[0].max(-1)[0].item()
                    
                    # 组合相似度
                    similarity = 0.7 * base_sim + 0.3 * fusion_sim
                    similarities.append((cached_feat['img_path'], cached_feat['id'], similarity, base_sim, fusion_sim))
                else:
                    similarities.append((None, None, -1.0, 0.0, 0.0))
            
            # 过滤无效结果并按相似度降序排序
            similarities = [s for s in similarities if s[0] is not None]
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # 打印详细结果
            print(f"\nTop {min(top_k, len(similarities))} 结果详情:")
            for i, (img_path, person_id, combined_sim, img_sim, fusion_sim) in enumerate(similarities[:top_k]):
                print(f"{i+1}. 图像: {img_path}, ID: {person_id}, 综合相似度: {combined_sim:.4f}, 基础相似度: {img_sim:.4f}, 融合相似度: {fusion_sim:.4f}")
            
            # 返回top_k结果（只返回前三个元素，保持向后兼容）
            return [(path, pid, sim) for path, pid, sim, _, _ in similarities[:top_k]]
        except Exception as e:
            print(f"搜索过程中出错: {e}")
            return []

def main():
    """测试脚本入口"""
    # 配置和模型路径
    config_path = '/home/wangrui/code/MLLM4Text-ReID-main/logs/RSTPReid_1023/20251023_234421_finetune/configs.yaml'
    model_path = '/home/wangrui/code/MLLM4Text-ReID-main/checkpoint/best2.pth'
    
    # 创建演示实例
    demo = ReIDDemo(config_path, model_path)
    
    # 测试用例1：使用修改后的查询图像路径和文本描述
    print("\n=== 测试用例1：查询图像 + 文本描述1 ===")
    results1 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query='a man wearing a black coat and black trousers',
        img_weight=0.5,
        text_weight=0.5,
        top_k=5
    )
    
    # 测试用例2：使用不同的文本描述
    print("\n=== 测试用例2：查询图像 + 文本描述2 ===")
    results2 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query='a person in black clothing walking',
        img_weight=0.5,
        text_weight=0.5,
        top_k=5
    )
    
    # 测试用例3：使用更具体的文本描述
    print("\n=== 测试用例3：查询图像 + 文本描述3 ===")
    results3 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query='a male in black coat with hands in pockets',
        img_weight=0.5,
        text_weight=0.5,
        top_k=5
    )
    
    # 测试用例4：仅使用图像搜索（比较效果）
    print("\n=== 测试用例4：仅图像搜索（对比） ===")
    results4 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query=None,
        img_weight=1.0,
        text_weight=0.0,
        top_k=5
    )
    
    # 测试用例5：高文本权重搜索
    print("\n=== 测试用例5：高文本权重搜索 ===")
    results5 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query='a male in black coat with hands in pockets',
        img_weight=0.3,
        text_weight=0.7,
        top_k=5
    )
    
    print("\n=== 测试完成 ===")
    print("已按照IRRA类方式重新实现特征提取和相似度计算")
    print("使用model(image, text, ori_text)方式获取特征")
    print("结合基础特征和融合特征计算相似度")
    print("\n注意事项：")
    print("1. 确保配置文件和模型权重路径正确")
    print("2. 确保查询图像路径存在")
    print("3. 文本描述应尽量准确反映目标人物特征")
    print("4. 首次运行会预处理所有图像，可能需要一些时间")

if __name__ == '__main__':
    main()