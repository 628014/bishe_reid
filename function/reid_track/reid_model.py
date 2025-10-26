import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入必要的模块
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize

class ReIDModel:
    def __init__(self, config_file, checkpoint_path, img_size=224, device="cuda"):
        """初始化ReID模型"""
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.img_size = img_size
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 加载配置
        self.args = load_train_configs(config_file)
        self.args.training = False
        
        # 构建模型
        self.model = self._build_model()
        
        # 初始化tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # 初始化图像转换器
        self.image_transform = self._get_image_transforms()
    
    def _get_image_transforms(self):
        """获取图像预处理转换"""
        if isinstance(self.img_size, tuple):
            resize_size = (int(self.img_size[0]), int(self.img_size[1]))
        else:
            resize_size = (int(self.img_size), int(self.img_size))
        
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def _build_model(self):
        """构建并加载ReID模型"""
        from datasets import build_dataloader
        _, _, num_classes = build_dataloader(self.args)
        model = build_model(self.args, num_classes=num_classes)
        
        # 加载权重
        assert os.path.exists(self.checkpoint_path), f"模型权重文件不存在: {self.checkpoint_path}"
        checkpointer = Checkpointer(model)
        checkpointer.load(f=self.checkpoint_path)
        
        model.to(self.device)
        model.eval()
        return model
    
    def extract_image_feature(self, image):
        """从图像中提取特征向量
        Args:
            image: PIL图像或图像路径
        Returns:
            归一化的特征向量
        """
        try:
            # 如果输入是路径，打开图像
            if isinstance(image, str) or isinstance(image, os.PathLike):
                img = Image.open(image).convert('RGB')
            else:
                img = image
            
            # 预处理
            img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                feat = self.model.encode_image(img_tensor)
                feat = F.normalize(feat, p=2, dim=1).cpu()
            
            return feat
        except Exception as e:
            print(f"提取图像特征失败: {e}")
            return None
    
    def extract_text_feature(self, text):
        """从文本中提取特征向量
        Args:
            text: 描述性文本
        Returns:
            归一化的特征向量
        """
        try:
            # 文本编码
            text_tensor = tokenize(
                caption=text,
                tokenizer=self.tokenizer
            ).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                feat = self.model.encode_text(text_tensor)
                feat = F.normalize(feat, p=2, dim=1).cpu()
            
            return feat
        except Exception as e:
            print(f"提取文本特征失败: {e}")
            return None
    
    def calculate_similarity(self, feat1, feat2):
        """计算两个特征向量之间的相似度
        Args:
            feat1: 第一个特征向量
            feat2: 第二个特征向量
        Returns:
            相似度分数
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        with torch.no_grad():
            similarity = torch.mm(feat1, feat2.t()).squeeze().item()
        
        return similarity
    
    def image_text_similarity(self, image, text):
        """计算图像和文本之间的相似度
        Args:
            image: PIL图像或图像路径
            text: 描述性文本
        Returns:
            相似度分数
        """
        img_feat = self.extract_image_feature(image)
        text_feat = self.extract_text_feature(text)
        
        return self.calculate_similarity(img_feat, text_feat)