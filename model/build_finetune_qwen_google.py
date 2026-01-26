import torch
import torch.nn as nn
from transformers import Qwen2_5_VLModel, AutoProcessor

class QwenVLReID(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        
        print(f"Loading Qwen2.5-VL from: {args.qwen_model_path}")
        self.processor = AutoProcessor.from_pretrained(args.qwen_model_path, trust_remote_code=True)
        # 使用 Qwen2_5_VLModel 作为 Base Model
        self.qwen_model = Qwen2_5_VLModel.from_pretrained(
            args.qwen_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        for param in self.qwen_model.parameters():
            param.requires_grad = False
            
        self.embed_dim = 3584 
        
    def get_last_token_features(self, last_hidden_state, attention_mask):
        """
        获取 Batch 中每个序列最后一个有效 token 的特征 (EOS Pooling)
        """
        # attention_mask shape: [batch, seq_len], 1 for valid, 0 for pad
        # sum(1) 得到每个样本的真实长度
        # 索引是从0开始的，所以最后一个 token 的索引是 length - 1
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]

        # 构造索引 [0, 1, ... B-1] 和 [len0, len1, ... lenB]
        # 选取对应的 hidden state
        last_token_features = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        
        return last_token_features

    def encode_image(self, image):
        """提取图像特征：使用 <|image_pad|> 占位，取最后一个 token"""
        try:
            device = self.qwen_model.device
            # 构造 Prompt: 纯图片输入
            text_prompts = ["<|image_pad|>"] * len(image)
            
            inputs = self.processor(
                text=text_prompts,
                images=image,
                videos=None,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                # 使用 Last Token Pooling
                feature = self.get_last_token_features(outputs.last_hidden_state, inputs.attention_mask)
            
            return feature.float()

        except Exception as e:
            print(f"Error extracting image feature: {e}")
            import traceback
            traceback.print_exc()
            batch_size = len(image) if isinstance(image, list) else 1
            return torch.randn(batch_size, self.embed_dim, device=self.qwen_model.device)

    def encode_text(self, text):
        """提取文本特征：取最后一个 token"""
        try:
            device = self.qwen_model.device
            
            # 确保 text 是 list
            if isinstance(text, str): text = [text]
            
            # 这里的 text 是原始 caption，需要 processor 处理添加 special tokens
            inputs = self.processor(
                text=text,
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                # 使用 Last Token Pooling
                feature = self.get_last_token_features(outputs.last_hidden_state, inputs.attention_mask)
                
            return feature.float()

        except Exception as e:
            print(f"Error extracting text feature: {e}")
            batch_size = len(text) if isinstance(text, list) else 1
            return torch.randn(batch_size, self.embed_dim, device=self.qwen_model.device)

    def forward(self, batch):
        return self.encode_image(batch['images'])

def build_finetune_model_qwen(args, num_classes=11003):
    model = QwenVLReID(args, num_classes)
    return model