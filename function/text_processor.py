import torch
import os
from datasets.bases import tokenize

class TextProcessor:
    """
    文本处理器，用于处理行人描述文本
    """
    def __init__(self, text_length=77, tokenizer=None):
        """
        初始化文本处理器
        
        Args:
            text_length: 文本长度，默认为77（CLIP模型的默认长度）
            tokenizer: 分词器，默认为None（使用项目默认的分词器）
        """
        self.text_length = text_length
        self.tokenizer = tokenizer
        print(f"文本处理器初始化完成，文本长度设置为: {text_length}")
    
    def process_text(self, text):
        """
        处理单个文本描述
        
        Args:
            text: 文本描述字符串
            
        Returns:
            处理后的文本token张量
        """
        # 使用项目中的tokenize函数进行处理
        tokens = tokenize(
            text, 
            tokenizer=self.tokenizer, 
            text_length=self.text_length, 
            truncate=True
        )
        
        # 增加批次维度
        tokens = tokens.unsqueeze(0)
        
        return tokens
    
    def process_text_list(self, texts):
        """
        批量处理文本描述列表
        
        Args:
            texts: 文本描述列表
            
        Returns:
            处理后的文本token张量（批次维度在前）
        """
        tokens_list = []
        
        for text in texts:
            tokens = tokenize(
                text, 
                tokenizer=self.tokenizer, 
                text_length=self.text_length, 
                truncate=True
            )
            tokens_list.append(tokens)
        
        # 堆叠成批次
        batch_tokens = torch.stack(tokens_list, dim=0)
        
        return batch_tokens
    
    def load_descriptions_from_file(self, file_path):
        """
        从文件加载描述列表
        
        Args:
            file_path: 文本文件路径，每行一个描述
            
        Returns:
            描述列表
        """
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
        
        descriptions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    descriptions.append(line)
        
        return descriptions
    
    def save_descriptions_to_file(self, descriptions, file_path):
        """
        将描述列表保存到文件
        
        Args:
            descriptions: 描述列表
            file_path: 输出文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for desc in descriptions:
                f.write(desc + '\n')
        
        print(f"已保存 {len(descriptions)} 个描述到 {file_path}")
    
    def generate_query_features(self, text, feature_extractor):
        """
        生成查询文本的特征
        
        Args:
            text: 查询文本
            feature_extractor: 特征提取器实例
            
        Returns:
            文本特征向量
        """
        # 处理文本
        tokens = self.process_text(text)
        
        # 提取特征
        features = feature_extractor.extract_text_features(tokens)
        
        return features
    
    def find_matching_persons(self, text, image_features_list, top_k=10):
        """
        查找与文本描述最匹配的行人
        
        Args:
            text: 查询文本
            image_features_list: 图像特征列表
            top_k: 返回前k个匹配结果
            
        Returns:
            匹配索引和相似度列表
        """
        # 这个方法需要在实际使用时结合特征提取器
        # 这里仅作为接口定义
        pass
    
    def generate_person_descriptions(self, person_images, caption_model=None):
        """
        为行人图像生成文本描述（需要额外的图像描述模型）
        
        Args:
            person_images: 行人图像列表
            caption_model: 图像描述模型，默认为None
            
        Returns:
            生成的描述列表
        """
        # 这个功能需要额外的图像描述模型
        # 例如可以使用CLIP或其他多模态模型
        descriptions = []
        
        if caption_model is None:
            # 如果没有提供模型，返回占位描述
            for i in range(len(person_images)):
                descriptions.append(f"Person {i+1}")
        else:
            # 实际应用中，这里应该调用图像描述模型
            pass
        
        return descriptions

# 示例用法
if __name__ == "__main__":
    processor = TextProcessor(text_length=77)
    text = "穿红色短袖的女孩，黑色长裤，白色运动鞋"
    tokens = processor.process_text(text)
    print("文本token形状:", tokens.shape)
    
    # 批量处理
    texts = [
        "穿红色短袖的女孩",
        "穿蓝色夹克的男孩",
        "戴帽子的老人"
    ]
    batch_tokens = processor.process_text_list(texts)
    print("批量token形状:", batch_tokens.shape)