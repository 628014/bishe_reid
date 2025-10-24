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
        
        # 加载数据集信息
        self.data_info = self._load_data_info()
        
        # 温度参数，与processor.py保持一致
        self.logit_scale = torch.ones([]) * (1 / self.args.temperature)
        
        # 初始化SimpleTokenizer，与IRRA保持一致
        self.tokenizer = SimpleTokenizer()
        
        # 初始化缓存
        self._cached_feats = None
        
    def _load_data_info(self):
        """加载数据信息"""
        data_path = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_text_input(self, text=None):
        """准备文本输入，生成text和ori_text张量
        使用SimpleTokenizer进行文本处理，参照IRRA的tokenizer方式
        
        Args:
            text: 输入文本，如果为None则生成空文本占位符
            
        Returns:
            text_tensor: 文本张量
            ori_text_tensor: 原始文本张量
        """
        context_length = self.args.text_length
        
        # 如果没有文本，使用空文本占位符，但确保长度符合模型要求
        if text is None:
            # 创建一个长度为context_length的空文本张量
            text_ids = torch.zeros(context_length, dtype=torch.long).to(self.device)
            # 设置开始和结束token
            text_ids[0] = 49406  # <start>
            text_ids[1] = 49407  # <end>
            # 其余位置保持为0（填充）
            text_tensor = text_ids.unsqueeze(0)
            ori_text_tensor = text_tensor.clone()
            return text_tensor, ori_text_tensor
        
        # 使用SimpleTokenizer进行文本处理，与IRRA保持一致
        context_length = self.args.text_length
        tokens = []
        
        # 添加开始token
        tokens.append(49406)  # <start>
        
        # 使用SimpleTokenizer的encoder进行分词
        for token in self.tokenizer.pat.findall(text):
            token = ''.join(self.tokenizer.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(self.tokenizer.bpe(token).split(' '))
        
        # 将BPE token转换为ID
        token_ids = []
        for token in tokens:
            if token in self.tokenizer.encoder:
                token_ids.append(self.tokenizer.encoder[token])
        
        # 截取到context_length-1的长度，留出结束token的位置
        token_ids = token_ids[:context_length-1]
        
        # 添加结束token
        token_ids.append(49407)  # <end>
        
        # 创建文本张量，填充到context_length长度
        text_ids = torch.zeros(context_length, dtype=torch.long).to(self.device)
        text_ids[:len(token_ids)] = torch.tensor(token_ids, dtype=torch.long).to(self.device)
        
        text_tensor = text_ids.unsqueeze(0)
        ori_text_tensor = text_tensor.clone()
        
        return text_tensor, ori_text_tensor
    
    def _extract_features(self, image=None, text=None):
        """同时提取图像和文本特征
        
        Args:
            image: 图像张量或None
            text: 文本字符串或None
            
        Returns:
            i_feats: 图像特征
            text_feats: 文本特征
            fu_i_feats: 融合后的图像特征
            fu_t_feats: 融合后的文本特征
        """
        # 准备文本输入
        text_tensor, ori_text_tensor = self._prepare_text_input(text)
        
        # 如果没有图像，创建一个空图像占位符
        if image is None:
            dummy_image = torch.zeros(1, 3, self.args.img_size, self.args.img_size).to(self.device)
            input_tensor = dummy_image
        else:
            input_tensor = image
        
        with torch.no_grad():
            try:
                # 尝试获取四个返回值
                output = self.model(input_tensor, text_tensor, ori_text_tensor)
                
                # 检查返回值的类型和结构
                if isinstance(output, tuple) and len(output) == 4:
                    i_feats, text_feats, fu_i_feats, fu_t_feats = output
                else:
                    # 如果返回值不是预期的四个值，尝试从基础模型直接获取特征
                    # 这种情况可能发生在模型处于不同的运行模式时
                    image_feats, text_feats = self.model.base_model(input_tensor, text_tensor)
                    # 创建融合特征的副本作为替代
                    fu_i_feats = image_feats.clone()
                    fu_t_feats = text_feats.clone()
                    i_feats = image_feats
            except ValueError as e:
                # 捕获解包错误，提供更详细的调试信息
                print(f"模型返回值解包错误: {e}")
                # 使用基础模型直接获取特征作为后备方案
                image_feats, text_feats = self.model.base_model(input_tensor, text_tensor)
                fu_i_feats = image_feats.clone()
                fu_t_feats = text_feats.clone()
                i_feats = image_feats
        
        return i_feats, text_feats, fu_i_feats, fu_t_feats
    
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
    
    def search_similar_images(self, query_image_path=None, text_query=None, top_k=5, img_weight=0.5, text_weight=0.5):
        """搜索相似图像，支持三种输入模式：纯图像、纯文本、图像和文本
        
        Args:
            query_image_path: 查询图像路径（可选）
            text_query: 文本查询（可选）
            top_k: 返回前k个结果
            img_weight: 图像特征权重
            text_weight: 文本特征权重
            
        Returns:
            top_results: 排序后的结果列表，每个元素为 (image_path, person_id, similarity)
        """
        try:
            # 检查输入模式
            if query_image_path is None and text_query is None:
                print("错误：必须提供图像路径或文本查询")
                return []
            
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
            
            # 预处理所有图像特征（如果需要）
            if self._cached_feats is None or len(self._cached_feats) != len(all_images):
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
                    
                        # 按照processor.py中的方式，同时提取图像特征
                        # 使用空文本作为占位符
                        _, _, fu_i_feats, _ = self._extract_features(image=image, text=None)
                    
                        # 按照processor.py中的方式，获取图像的第一个token特征
                        i_feats_0 = fu_i_feats[:, 0, :]  # 第一个token特征
                    
                        # 存储图像特征
                        self._cached_feats.append({
                            'i_feats_0': i_feats_0,
                            'fu_i_feats': fu_i_feats,
                            'img_path': img_path,
                            'id': img_info['id']
                        })
                    except Exception as e:
                        print(f"预处理图像 {img_info['img_path']} 时出错: {e}")
                        self._cached_feats.append(None)
                print(f"预处理完成，共 {len(self._cached_feats)} 个图像特征")
            
            # 处理查询
            query_i_feats_0 = None  # 图像特征
            query_fu_feats = None  # 图像融合特征
            query_t_feats = None  # 文本特征
            query_fu_t_feats = None  # 文本融合特征
            
            # 模式1：纯图像输入
            if query_image_path and text_query is None:
                # 获取查询图像的绝对路径
                if not os.path.isabs(query_image_path):
                    root_dir = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/'
                    query_image_path = os.path.join(root_dir, query_image_path)
                
                # 检查文件是否存在
                if not os.path.exists(query_image_path):
                    raise FileNotFoundError(f"图像文件不存在: {query_image_path}")
                
                try:
                    # 加载图像
                    query_image = Image.open(query_image_path).convert('RGB')
                    query_image = self.transform(query_image).unsqueeze(0).to(self.device)
                    
                    # 提取图像特征（使用空文本）
                    i_feats, text_feats, fu_i_feats, fu_t_feats = self._extract_features(image=query_image, text=None)
                    
                    # 提取图像特征（第一个token）
                    query_i_feats_0 = i_feats[:, 0, :].float()
                    query_i_feats_0 = F.normalize(query_i_feats_0, dim=-1)
                    
                    # 融合特征
                    query_fu_feats = F.normalize(fu_i_feats, dim=-1)
                    
                except Exception as e:
                    print(f"处理查询图像时出错: {e}")
                    return []
            
            # 模式2：纯文本输入
            elif query_image_path is None and text_query:
                try:
                    # 提取文本特征（使用空图像）
                    i_feats, text_feats, fu_i_feats, fu_t_feats = self._extract_features(image=None, text=text_query)
                    
                    # 提取文本特征（找到caption_ids的argmax位置）
                    # 这里简化处理，使用第一个非零位置
                    caption_ids = torch.argmax(text_feats, dim=-1)
                    t_feats = text_feats[:, torch.argmax(caption_ids != 0, dim=-1), :].squeeze(1).float()
                    query_t_feats = F.normalize(t_feats, dim=-1)
                    
                    # 文本融合特征
                    query_fu_t_feats = F.normalize(fu_t_feats, dim=-1)
                    
                except Exception as e:
                    print(f"处理查询文本时出错: {e}")
                    return []
            
            # 模式3：图像和文本输入
            elif query_image_path and text_query:
                # 获取查询图像的绝对路径
                if not os.path.isabs(query_image_path):
                    root_dir = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid'
                    query_image_path = os.path.join(root_dir, query_image_path)
                
                # 检查文件是否存在
                if not os.path.exists(query_image_path):
                    raise FileNotFoundError(f"图像文件不存在: {query_image_path}")
                
                try:
                    # 加载图像
                    query_image = Image.open(query_image_path).convert('RGB')
                    query_image = self.transform(query_image).unsqueeze(0).to(self.device)
                    
                    # 同时提取图像和文本特征
                    i_feats, text_feats, fu_i_feats, fu_t_feats = self._extract_features(image=query_image, text=text_query)
                    
                    # 提取图像特征（第一个token）
                    query_i_feats_0 = i_feats[:, 0, :].float()
                    query_i_feats_0 = F.normalize(query_i_feats_0, dim=-1)
                    
                    # 图像融合特征
                    query_fu_feats = F.normalize(fu_i_feats, dim=-1)
                    
                    # 提取文本特征
                    caption_ids = torch.argmax(text_feats, dim=-1)
                    t_feats = text_feats[:, torch.argmax(caption_ids != 0, dim=-1), :].squeeze(1).float()
                    query_t_feats = F.normalize(t_feats, dim=-1)
                    
                    # 文本融合特征
                    query_fu_t_feats = F.normalize(fu_t_feats, dim=-1)
                    
                except Exception as e:
                    print(f"处理查询时出错: {e}")
                    return []
            
            # 计算相似度
            similarities = []
            for i, cached_feat in enumerate(self._cached_feats):
                if cached_feat is not None:
                    try:
                        combined_sim = 0.0
                        img_sim = 0.0
                        text_sim = 0.0
                        fusion_sim = 0.0
                        
                        # 获取图像特征并归一化
                        i_feats_0 = F.normalize(cached_feat['i_feats_0'], dim=-1)
                        fu_i_feats_norm = F.normalize(cached_feat['fu_i_feats'], dim=-1)
                        
                        # 根据输入模式计算不同的相似度
                        if query_image_path and text_query is None:
                            # 纯图像：计算图与图之间的相似度
                            # 基础相似度
                            img_sim = torch.matmul(query_i_feats_0, i_feats_0.T).item()
                            # 融合特征相似度
                            fusion_sim = torch.einsum('nld,nkd->nlk', [query_fu_feats, fu_i_feats_norm[:, 1:, :]])
                            fusion_sim = fusion_sim.max(-1)[0].max(-1)[0].item()
                            # 组合相似度
                            combined_sim = 0.7 * img_sim + 0.3 * fusion_sim
                        
                        elif query_image_path is None and text_query:
                            # 纯文本：计算文本与图像的相似度
                            # 文本与图像基础相似度
                            text_sim = torch.matmul(query_t_feats, i_feats_0.T).item()
                            # 文本与图像融合特征相似度
                            fusion_sim = torch.einsum('nld,nkd->nlk', [query_fu_t_feats, fu_i_feats_norm[:, 1:, :]])
                            fusion_sim = fusion_sim.max(-1)[0].max(-1)[0].item()
                            # 组合相似度
                            combined_sim = 0.7 * text_sim + 0.3 * fusion_sim
                        
                        elif query_image_path and text_query:
                            # 图像和文本：计算加权组合相似度
                            # 图与图基础相似度
                            img_sim = torch.matmul(query_i_feats_0, i_feats_0.T).item()
                            # 文本与图基础相似度
                            text_sim = torch.matmul(query_t_feats, i_feats_0.T).item()
                            # 图与图融合特征相似度
                            img_fusion_sim = torch.einsum('nld,nkd->nlk', [query_fu_feats, fu_i_feats_norm[:, 1:, :]])
                            img_fusion_sim = img_fusion_sim.max(-1)[0].max(-1)[0].item()
                            # 文本与图融合特征相似度
                            text_fusion_sim = torch.einsum('nld,nkd->nlk', [query_fu_t_feats, fu_i_feats_norm[:, 1:, :]])
                            text_fusion_sim = text_fusion_sim.max(-1)[0].max(-1)[0].item()
                            # 加权组合
                            combined_img_sim = 0.7 * img_sim + 0.3 * img_fusion_sim
                            combined_text_sim = 0.7 * text_sim + 0.3 * text_fusion_sim
                            combined_sim = img_weight * combined_img_sim + text_weight * combined_text_sim
                            
                            # 更新显示的fusion_sim值
                            fusion_sim = (img_fusion_sim + text_fusion_sim) / 2
                            img_sim = (img_sim + text_sim) / 2  # 用于显示
                        
                        similarities.append((cached_feat['img_path'], cached_feat['id'], combined_sim, img_sim, fusion_sim))
                    except Exception as e:
                        print(f"计算与图像 {cached_feat['img_path']} 的相似度时出错: {e}")
                        similarities.append((cached_feat['img_path'], cached_feat['id'], -1.0, 0.0, 0.0))
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
    
    # 测试用例1：使用修改后的查询图像路径和文本描述（图像+文本模式）
    print("\n=== 测试用例1：查询图像 + 文本描述1（图像文本模式） ===")
    results1 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query='a man wearing a black coat and black trousers',
        img_weight=0.5,
        text_weight=0.5,
        top_k=5
    )
    
    # 测试用例2：纯图像输入模式
    print("\n=== 测试用例2：纯图像输入模式 ===")
    results2 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query=None,
        img_weight=1.0,
        text_weight=0.0,
        top_k=5
    )
    
    # 测试用例3：纯文本输入模式
    print("\n=== 测试用例3：纯文本输入模式 ===")
    results3 = demo.search_similar_images(
        query_image_path=None,
        text_query='a man wearing a black coat and black trousers',
        img_weight=0.0,
        text_weight=1.0,
        top_k=5
    )
    
    # 测试用例4：图像和文本输入（自定义权重）
    print("\n=== 测试用例4：高文本权重搜索 ===")
    results4 = demo.search_similar_images(
        query_image_path='/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/3902_c5_0005.jpg',
        text_query='a male in black coat with hands in pockets',
        img_weight=0.3,
        text_weight=0.7,
        top_k=5
    )
    
    print("\n=== 测试完成 ===")
    print("已按照IRRA方式重新实现文本tokenization和特征提取")
    print("支持三种输入模式：纯图像、纯文本、图像和文本（可自定义权重）")
    print("使用SimpleTokenizer进行文本处理")
    print("结合基础特征和融合特征计算相似度")
    print("\n注意事项：")
    print("1. 确保配置文件和模型权重路径正确")
    print("2. 确保查询图像路径存在")
    print("3. 文本描述应尽量准确反映目标人物特征")
    print("4. 首次运行会预处理所有图像，可能需要一些时间")

if __name__ == '__main__':
    main()