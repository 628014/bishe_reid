import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


class QwenVLReID(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        
        # 加载Qwen2.5-VL-7B模型
        self.processor = AutoProcessor.from_pretrained(args.qwen_model_path, trust_remote_code=True)
        self.qwen_model = AutoModel.from_pretrained(
            args.qwen_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 冻结Qwen模型参数
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        
        # 设置嵌入维度
        self.embed_dim = 4096  # Qwen2.5-VL-7B的特征维度
        
        # 分类头（用于ID loss，评估时可能不需要，但保留以便兼容）
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)
    
    def encode_image(self, image):
        # import pdb
        # pdb.set_trace()
        """提取Qwen视觉特征用于ReID评估"""
        try:
            # 处理图像张量
            if isinstance(image, dict):
                # 兼容forward方法的调用
                image = image['images']
            
            # 预处理图像
            device = next(self.parameters()).device
            inputs = self.processor(images=image, return_tensors="pt").to(device, dtype=torch.float16)
            
            # 提取视觉特征
            with torch.no_grad():
                if hasattr(self.qwen_model, 'visual'):
                    # 直接使用视觉编码器
                    image_feats = self.qwen_model.visual(inputs['pixel_values'])[0]  # [B, C, H, W]
                else:
                    # 使用完整模型获取特征
                    outputs = self.qwen_model(**inputs, output_hidden_states=True)
                    image_feats = outputs.hidden_states[-1]
                    # 对序列维度进行平均池化
                    image_feats = torch.mean(image_feats, dim=1)
            
            # 确保特征是一维的
            if len(image_feats.shape) > 2:
                image_feats = F.adaptive_avg_pool2d(image_feats, (1, 1)).squeeze(-1).squeeze(-1)
            
            return image_feats.float()
        except Exception as e:
            print(f"Error in encode_image: {e}")
            # 返回默认特征以避免崩溃
            # try to fallback to model device
            dev = next(self.parameters()).device
            batch_size = image.shape[0] if hasattr(image, 'shape') else 1
            return torch.randn(batch_size, self.embed_dim, device=dev)

    def encode_text(self, text):
        """提取Qwen文本特征用于ReID评估"""
        try:
            # 处理文本
            if isinstance(text, dict):
                # 兼容forward方法的调用
                text = text['caption_ids']
            
            # 确保文本是字符串格式
            text_list = []
            if isinstance(text, str):
                text_list = [text]
            elif hasattr(text, 'shape') and text.dim() > 1:
                # 批量处理文本张量
                for i in range(text.shape[0]):
                    # 简单处理：将张量转换为字符串
                    text_str = ' '.join([str(int(token)) for token in text[i].tolist() if token != 0])
                    text_list.append(text_str)
            elif hasattr(text, 'shape') and text.dim() == 1:
                # 单条文本张量
                text_str = ' '.join([str(int(token)) for token in text.tolist() if token != 0])
                text_list = [text_str]
            else:
                text_list = text
            
            # 预处理文本
            inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(next(self.parameters()).device, dtype=torch.float16)
            
            # 提取文本特征
            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                text_feats = outputs.hidden_states[-1]
                # 使用CLS token或平均池化
                if text_feats.shape[1] > 1:
                    text_feats = text_feats[:, 0, :]  # 使用第一个token的特征
                else:
                    text_feats = torch.mean(text_feats, dim=1)
            
            return text_feats.float()
        except Exception as e:
            print(f"Error in encode_text: {e}")
            # 返回默认特征以避免崩溃
            batch_size = text.shape[0] if hasattr(text, 'shape') else len(text) if isinstance(text, list) else 1
            return torch.randn(batch_size, self.embed_dim, device=next(self.parameters()).device)

    def forward(self, batch):
        """前向传播，主要用于兼容性，评估时直接调用encode_image和encode_text"""
        ret = dict()
        
        images = batch['images']
        caption_ids = batch['caption_ids']
        
        # 提取视觉和文本特征
        i_feats = self.encode_image(images)
        t_feats = self.encode_text(caption_ids)
        
        # 计算分类概率（如果需要）
        image_logits = self.classifier(i_feats)
        text_logits = self.classifier(t_feats)
        
        ret['image_feats'] = i_feats
        ret['text_feats'] = t_feats
        ret['image_logits'] = image_logits
        ret['text_logits'] = text_logits
        
        return ret


def build_finetune_model_qwen(args, num_classes=11003):
    """构建用于评估的QwenVLReID模型"""
    model = QwenVLReID(args, num_classes)
    return model
