import torch
import numpy as np
import random
from torch.utils.data import Dataset
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from .bases import tokenize

class DynamicMaskingDataset(Dataset):
    """
    基于 Match Score 的动态遮蔽数据集类
    """
    def __init__(self, dataset, transform=None, text_length=77, truncate=True, max_score=10.0):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        
        # 参数
        self.base_prob = 0.15 
        self.min_prob = 0.05
        self.max_prob = 0.25
        self.max_score_ref = float(max_score)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]
        if len(data_item) == 5:
            pid, image_id, img_path, caption, raw_score = data_item
        else:
            pid, image_id, img_path, caption = data_item
            raw_score = self.max_score_ref 

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        # --- 核心修复：智能归一化 ---
        try:
            raw_s = float(raw_score)
        except:
            raw_s = self.max_score_ref

        # 逻辑：如果原始分数大于1.0，说明是0-10分制，必须除以10
        # 如果原始分数在0-1之间，说明已经是归一化过的（或者本来就很低），保持原样
        if raw_s > 1.0:
            sim = raw_s / 10.0
        else:
            sim = raw_s
        
        # 强制截断到 [0, 1] 之间，这对防止梯度爆炸至关重要
        sim = max(0.0, min(1.0, sim))

        # 计算Mask概率
        range_span = self.max_prob - self.min_prob
        current_mask_prob = self.max_prob - (range_span * sim)
        current_mask_prob = max(self.min_prob, min(current_mask_prob, self.max_prob))

        mlm_tokens, mlm_labels = self._build_masked_tokens(caption_tokens.cpu().numpy(), current_mask_prob)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            # 返回归一化后的 sim (0-1)，用于计算 MSE Loss
            'match_score': torch.tensor(sim, dtype=torch.float32) 
        }

        return ret

    def _build_masked_tokens(self, tokens, mask_prob):
        tokens = tokens.copy()
        labels = np.zeros_like(tokens)
        
        mask_token_id = self.tokenizer.encoder["<|mask|>"]
        vocab_range = list(range(1, 49405)) 

        valid_len = np.where(tokens == 0)[0]
        if len(valid_len) > 0:
            valid_len = valid_len[0]
        else:
            valid_len = len(tokens)
        
        for i in range(1, valid_len - 1):
            token = tokens[i]
            if random.random() < mask_prob:
                labels[i] = token 
                r = random.random()
                if r < 0.8:
                    tokens[i] = mask_token_id
                elif r < 0.9:
                    tokens[i] = random.choice(vocab_range)
        
        if np.sum(labels) == 0 and valid_len > 2:
            idx = random.randint(1, valid_len - 2)
            labels[idx] = tokens[idx]
            tokens[idx] = mask_token_id

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)