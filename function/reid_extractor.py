import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到Python路径
sys.path.append('/home/wangrui/code/MLLM4Text-ReID-main')

# 尝试导入模型相关模块
try:
    from model.build import build_model
    from utils.options import get_args
    import warnings
    warnings.filterwarnings('ignore')
except Exception as e:
    print(f"导入模型模块时出错: {e}")
    print("将使用简化的特征提取器")

class ReIDExtractor:
    """
    行人重识别特征提取器
    使用IRRA模型提取行人特征用于跨摄像头追踪
    """
    def __init__(self, model_path='/home/wangrui/code/MLLM4Text-ReID-main/checkpoint/best2.pth', 
                 config_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化ReID特征提取器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.transform = self._get_transform()
        self.loaded = False
        
        # 尝试加载模型
        self._load_model()
    
    def _get_transform(self):
        """
        获取图像预处理变换
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """
        加载IRRA模型
        """
        try:
            # 获取配置参数
            if '--config_file' not in sys.argv:
                sys.argv.append('--config_file')
                sys.argv.append('/home/wangrui/code/MLLM4Text-ReID-main/logs/RSTPReid_1023/20251023_234421_finetune/configs.yaml')
            
            args = get_args()
            
            # 构建模型
            self.model = build_model(args)
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同格式的检查点
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # 将模型移至指定设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            print(f"成功加载ReID模型: {self.model_path}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用简化的特征提取器作为后备方案")
            self.loaded = False
    
    def extract_feature(self, image):
        """
        从单张图像中提取特征
        
        Args:
            image: 输入图像 (numpy array 或 PIL Image)
            
        Returns:
            特征向量 (numpy array)
        """
        try:
            # 图像预处理
            if isinstance(image, np.ndarray):
                # BGR转RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # 应用变换
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 使用模型提取特征
            with torch.no_grad():
                if self.loaded and self.model is not None:
                    # 使用IRRA模型提取图像特征
                    image_features = self.model.encode_image(input_tensor)
                    features = image_features.squeeze().cpu().numpy()
                else:
                    # 简化的特征提取作为后备
                    features = self._simple_feature_extraction(input_tensor)
            
            # 归一化特征向量
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"提取特征时出错: {e}")
            # 返回默认特征向量
            return np.zeros(2048)
    
    def extract_features_batch(self, images):
        """
        批量提取特征
        
        Args:
            images: 图像列表
            
        Returns:
            特征向量列表
        """
        features_list = []
        for image in images:
            feat = self.extract_feature(image)
            features_list.append(feat)
        return features_list
    
    def _simple_feature_extraction(self, input_tensor):
        """
        简化的特征提取作为后备方案
        主要用于模型加载失败的情况
        """
        # 简单地使用平均池化和扁平化作为后备特征
        # 调整为固定长度的特征向量
        try:
            # 简单的特征提取
            x = input_tensor.squeeze().cpu().numpy()
            # 计算颜色直方图
            hist_r = np.histogram(x[0], bins=16)[0]
            hist_g = np.histogram(x[1], bins=16)[0]
            hist_b = np.histogram(x[2], bins=16)[0]
            # 组合为特征向量
            features = np.concatenate([hist_r, hist_g, hist_b])
            # 调整为固定长度
            if len(features) < 2048:
                features = np.pad(features, (0, 2048 - len(features)), 'constant')
            else:
                features = features[:2048]
            return features
        except:
            return np.zeros(2048)
    
    def compute_similarity(self, feat1, feat2):
        """
        计算两个特征向量的余弦相似度
        
        Args:
            feat1: 第一个特征向量
            feat2: 第二个特征向量
            
        Returns:
            余弦相似度值
        """
        try:
            # 确保输入是numpy数组
            if isinstance(feat1, list):
                feat1 = np.array(feat1)
            if isinstance(feat2, list):
                feat2 = np.array(feat2)
            
            # 归一化
            feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
            feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)
            
            # 计算余弦相似度
            return np.dot(feat1, feat2)
        except Exception as e:
            print(f"计算相似度时出错: {e}")
            return 0.0
    
    def compare_images(self, image1, image2):
        """
        比较两张图像的相似度
        
        Args:
            image1: 第一张图像
            image2: 第二张图像
            
        Returns:
            相似度值
        """
        feat1 = self.extract_feature(image1)
        feat2 = self.extract_feature(image2)
        return self.compute_similarity(feat1, feat2)
    
    def is_loaded(self):
        """
        检查模型是否成功加载
        
        Returns:
            布尔值，表示模型是否加载成功
        """
        return self.loaded

class DetectionFeatureExtractor:
    """
    检测结果特征提取器
    将检测到的行人边界框区域提取特征
    """
    def __init__(self, reid_extractor=None):
        """
        初始化特征提取器
        
        Args:
            reid_extractor: ReID特征提取器实例
        """
        if reid_extractor is None:
            self.reid_extractor = ReIDExtractor()
        else:
            self.reid_extractor = reid_extractor
    
    def extract_features_from_detections(self, frame, detections):
        """
        从检测结果中提取特征
        
        Args:
            frame: 原始图像帧
            detections: 检测结果列表，每个元素为 [x1, y1, x2, y2, score]
            
        Returns:
            特征向量列表
        """
        features = []
        
        for det in detections:
            # 确保检测框有效
            x1, y1, x2, y2 = map(int, det[:4])
            
            # 检查边界
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 确保边界框有效
            if x2 > x1 and y2 > y1:
                # 裁剪行人区域
                person_roi = frame[y1:y2, x1:x2]
                
                # 提取特征
                feat = self.reid_extractor.extract_feature(person_roi)
                features.append(feat)
            else:
                # 无效边界框，添加零向量
                features.append(np.zeros(2048))
        
        return features
    
    def extract_features_with_tracks(self, frame, tracks):
        """
        从追踪结果中提取特征
        
        Args:
            frame: 原始图像帧
            tracks: 追踪结果列表
            
        Returns:
            特征字典 {track_id: feature}
        """
        features_dict = {}
        
        for track in tracks:
            if hasattr(track, 'bbox'):
                bbox = track.bbox
                
                # 确保边界框有效
                x1, y1, x2, y2 = map(int, bbox)
                
                # 检查边界
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    # 裁剪行人区域
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # 提取特征
                    feat = self.reid_extractor.extract_feature(person_roi)
                    features_dict[track.track_id] = feat
        
        return features_dict

# 示例用法
if __name__ == "__main__":
    # 创建特征提取器实例
    extractor = ReIDExtractor()
    
    # 测试提取器是否加载成功
    print(f"模型加载状态: {extractor.is_loaded()}")
    
    # 可以在这里添加更多测试代码
    print("ReID特征提取器初始化完成")