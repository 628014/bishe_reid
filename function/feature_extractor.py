import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

class FeatureExtractor:
    """
    特征提取器，用于提取行人图像和文本描述的特征
    """
    def __init__(self, model, device=None):
        """
        初始化特征提取器
        
        Args:
            model: 预加载的IRRA模型
            device: 运行设备，如'cuda'或'cpu'，默认为None（自动选择）
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        
        # 定义图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"特征提取器初始化完成，使用设备: {self.device}")
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: OpenCV图像(BGR)或PIL图像
            
        Returns:
            预处理后的张量
        """
        if isinstance(image, np.ndarray):
            # 转换BGR到RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        if not isinstance(image, Image.Image):
            raise TypeError("图像必须是OpenCV格式(BGR)或PIL图像")
        
        return self.transform(image).unsqueeze(0)
    
    def extract_image_features(self, image):
        """
        提取图像特征
        
        Args:
            image: OpenCV图像(BGR)或PIL图像或图像路径
            
        Returns:
            特征向量
        """
        # 如果输入是路径，读取图像
        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"图像文件不存在: {image}")
            image = Image.open(image).convert('RGB')
        
        # 预处理
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
        
        return features
    
    def extract_text_features(self, text_tokens):
        """
        提取文本特征
        
        Args:
            text_tokens: 预处理后的文本token张量
            
        Returns:
            特征向量
        """
        text_tokens = text_tokens.to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
        
        return features
    
    def extract_features_from_detections(self, image, detections):
        """
        从检测结果中提取所有行人的特征
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            特征向量列表，与检测结果一一对应
        """
        features_list = []
        
        for det in detections:
            person_img = det['person_img']
            features = self.extract_image_features(person_img)
            features_list.append(features)
        
        return features_list
    
    def batch_extract_features(self, images, batch_size=32):
        """
        批量提取图像特征
        
        Args:
            images: 图像列表
            batch_size: 批次大小
            
        Returns:
            所有图像的特征向量
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensors = []
            
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
            
            all_features.append(batch_features)
        
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.tensor([])
    
    def normalize_features(self, features):
        """
        归一化特征向量
        
        Args:
            features: 特征向量
            
        Returns:
            归一化后的特征向量
        """
        return torch.nn.functional.normalize(features, dim=1)
    
    def compute_similarity(self, features1, features2):
        """
        计算特征之间的余弦相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            相似度矩阵
        """
        # 归一化特征
        features1_norm = self.normalize_features(features1)
        features2_norm = self.normalize_features(features2)
        
        # 计算余弦相似度
        similarity = torch.matmul(features1_norm, features2_norm.t())
        
        return similarity

# 示例用法
if __name__ == "__main__":
    # 这里需要先加载模型
    # from model.build_finetune import IRRA
    # import argparse
    # 
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # # 设置模型参数...
    # 
    # model = IRRA(args, num_classes)
    # # 加载预训练权重...
    # 
    # extractor = FeatureExtractor(model)
    # features = extractor.extract_image_features("person.jpg")
    # print("特征维度:", features.shape)