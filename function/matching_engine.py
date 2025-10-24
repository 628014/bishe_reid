import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

class MatchingEngine:
    """
    匹配引擎，用于计算相似度和匹配目标
    """
    def __init__(self, feature_extractor, text_processor, sim_threshold=0.5):
        """
        初始化匹配引擎
        
        Args:
            feature_extractor: 特征提取器实例
            text_processor: 文本处理器实例
            sim_threshold: 相似度阈值
        """
        self.feature_extractor = feature_extractor
        self.text_processor = text_processor
        self.sim_threshold = sim_threshold
        print(f"匹配引擎初始化完成，相似度阈值: {sim_threshold}")
    
    def compute_similarity(self, features1, features2):
        """
        计算两组特征之间的余弦相似度
        
        Args:
            features1: 第一组特征
            features2: 第二组特征
            
        Returns:
            相似度矩阵
        """
        # 归一化特征
        features1_norm = torch.nn.functional.normalize(features1, dim=1)
        features2_norm = torch.nn.functional.normalize(features2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(features1_norm, features2_norm.t())
        
        return similarity
    
    def text_image_matching(self, text, image_features, top_k=10):
        """
        文本到图像的匹配
        
        Args:
            text: 查询文本
            image_features: 图像特征列表或张量
            top_k: 返回前k个匹配结果
            
        Returns:
            匹配索引和相似度列表
        """
        # 处理文本并提取特征
        text_tokens = self.text_processor.process_text(text)
        text_features = self.feature_extractor.extract_text_features(text_tokens)
        
        # 计算相似度
        if isinstance(image_features, list):
            image_features_tensor = torch.cat(image_features, dim=0)
        else:
            image_features_tensor = image_features
        
        similarity = self.compute_similarity(text_features, image_features_tensor)
        similarity_np = similarity.cpu().numpy().flatten()
        
        # 排序并返回前k个结果
        sorted_indices = np.argsort(similarity_np)[::-1]
        top_indices = sorted_indices[:top_k]
        top_similarities = similarity_np[top_indices]
        
        return top_indices, top_similarities
    
    def batch_text_image_matching(self, texts, image_features, top_k=10):
        """
        批量文本到图像的匹配
        
        Args:
            texts: 文本列表
            image_features: 图像特征张量
            top_k: 返回前k个匹配结果
            
        Returns:
            每个文本的匹配结果列表
        """
        results = []
        
        for text in tqdm(texts, desc="文本匹配"):
            indices, similarities = self.text_image_matching(text, image_features, top_k)
            results.append({
                'text': text,
                'indices': indices,
                'similarities': similarities
            })
        
        return results
    
    def image_image_matching(self, query_image_features, gallery_image_features, top_k=10):
        """
        图像到图像的匹配
        
        Args:
            query_image_features: 查询图像特征
            gallery_image_features: 图库图像特征
            top_k: 返回前k个匹配结果
            
        Returns:
            匹配索引和相似度列表
        """
        # 计算相似度
        similarity = self.compute_similarity(query_image_features, gallery_image_features)
        similarity_np = similarity.cpu().numpy()
        
        results = []
        
        # 对每个查询进行排序
        for i in range(similarity_np.shape[0]):
            sorted_indices = np.argsort(similarity_np[i])[::-1]
            top_indices = sorted_indices[:top_k]
            top_similarities = similarity_np[i][top_indices]
            
            results.append({
                'query_idx': i,
                'indices': top_indices,
                'similarities': top_similarities
            })
        
        return results
    
    def associate_with_hungarian(self, similarity_matrix, threshold=None):
        """
        使用匈牙利算法进行关联
        
        Args:
            similarity_matrix: 相似度矩阵
            threshold: 相似度阈值，低于此值的匹配将被拒绝
            
        Returns:
            关联结果列表 [(row_idx, col_idx, similarity)]
        """
        if threshold is None:
            threshold = self.sim_threshold
        
        # 转换为成本矩阵
        cost_matrix = 1 - similarity_matrix.cpu().numpy()
        
        # 应用匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 筛选高于阈值的匹配
        associations = []
        for i, j in zip(row_ind, col_ind):
            similarity = similarity_matrix[i, j].item()
            if similarity > threshold:
                associations.append((i, j, similarity))
        
        return associations
    
    def find_mutual_nearest_neighbors(self, features1, features2, k=5):
        """
        查找相互最近邻
        
        Args:
            features1: 第一组特征
            features2: 第二组特征
            k: 近邻数量
            
        Returns:
            相互近邻掩码矩阵
        """
        # 计算相似度
        sim_matrix = self.compute_similarity(features1, features2)
        
        # 找到k近邻
        _, indices1 = torch.topk(sim_matrix, k, dim=1)
        _, indices2 = torch.topk(sim_matrix.t(), k, dim=1)
        
        # 创建掩码矩阵
        mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        
        for i in range(features1.size(0)):
            for j in indices1[i]:
                # 检查i是否在j的k近邻中
                if i in indices2[j]:
                    mask[i, j] = True
        
        return mask
    
    def build_gallery(self, image_list):
        """
        构建图像特征图库
        
        Args:
            image_list: 图像列表
            
        Returns:
            图库特征张量
        """
        gallery_features = []
        
        for image in tqdm(image_list, desc="构建特征图库"):
            features = self.feature_extractor.extract_image_features(image)
            gallery_features.append(features)
        
        return torch.cat(gallery_features, dim=0)
    
    def save_gallery(self, gallery_features, file_path):
        """
        保存图库特征
        
        Args:
            gallery_features: 图库特征张量
            file_path: 保存路径
        """
        torch.save(gallery_features.cpu(), file_path)
        print(f"图库特征已保存到: {file_path}")
    
    def load_gallery(self, file_path):
        """
        加载图库特征
        
        Args:
            file_path: 文件路径
            
        Returns:
            图库特征张量
        """
        features = torch.load(file_path)
        if torch.cuda.is_available():
            features = features.cuda()
        print(f"已加载图库特征，形状: {features.shape}")
        return features
    
    def evaluate_matching(self, query_features, gallery_features, query_labels, gallery_labels, top_k=10):
        """
        评估匹配性能
        
        Args:
            query_features: 查询特征
            gallery_features: 图库特征
            query_labels: 查询标签
            gallery_labels: 图库标签
            top_k: 评估的top-k值
            
        Returns:
            评估结果字典
        """
        # 计算相似度
        similarity = self.compute_similarity(query_features, gallery_features)
        
        # 排序
        _, indices = torch.topk(similarity, k=min(top_k, gallery_features.size(0)), dim=1)
        
        # 计算准确率
        accuracy = {}
        for k in range(1, top_k + 1):
            correct = 0
            for i in range(len(query_labels)):
                top_k_indices = indices[i, :k].cpu().numpy()
                top_k_labels = [gallery_labels[j] for j in top_k_indices]
                if query_labels[i] in top_k_labels:
                    correct += 1
            accuracy[f'top{k}'] = correct / len(query_labels)
        
        return accuracy

# 示例用法
if __name__ == "__main__":
    # 这里需要先初始化特征提取器和文本处理器
    # from feature_extractor import FeatureExtractor
    # from text_processor import TextProcessor
    # 
    # feature_extractor = FeatureExtractor(model)
    # text_processor = TextProcessor()
    # matching_engine = MatchingEngine(feature_extractor, text_processor)
    # 
    # # 文本图像匹配示例
    # text = "穿红色短袖的女孩"
    # image_features = ...  # 这里应该是预提取的图像特征
    # indices, similarities = matching_engine.text_image_matching(text, image_features, top_k=10)
    # print(f"匹配结果: {list(zip(indices, similarities))}")