from typing import List
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import os
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'img_path':img_path,
            'caption':caption,
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption

def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

# 原始的15%掩码的版本
class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption, sim = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

# 新增的sim版本 只在预训练的时候使用来更新动态遮盖，微调的时候sim一直都是0.0
class FilterDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption, sim = self.dataset[index]
        # import pdb
        # pdb.set_trace()
        print("_______________________sim_____________________", sim)
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
            
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy(), sim)
        ori_tokens =  tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'caption_ids_ori':ori_tokens
        }
        
        return ret

    def _build_random_masked_tokens_and_labels(self, tokens, sim):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        
        if tokens[-1] == 0:
            valid_token_num = np.where(tokens == 0)[0][0]
        else:
            valid_token_num = len(tokens)
        # 根据LuPerson数据的初始化，sim是一个长度为77的列表，每个元素值为0.85
        # 根据Rstpreid的数据初始化，这里的sim原始是一个单独的0.0的值，后面改成了对应的图文匹配分match_score
        ori_sim = np.array(sim)
        # 归一化趋近于0.15
        ori_pro = 1 - ori_sim
        if ori_pro[-1] != 0.15:
            valid_prob = ori_pro[1:valid_token_num-1]
            # normalize the probisibility to match E = 0.15
            mean_prob = np.mean(valid_prob)
            normed_prob = valid_prob - mean_prob
            normalized_prob = normed_prob + 0.15
            normalized_prob = np.clip(normalized_prob, 0, 1)
            ori_pro[1:valid_token_num-1] = normalized_prob
        
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < ori_pro[i]:
                    prob /= ori_pro[i]

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

class DynamicMaskingDataset(ImageTextMLMDataset):
    """
    基于match-score的动态遮蔽数据集类
    核心特点：sim值越高，遮蔽概率越低，适合高质量文本
    """
    def __init__(self, dataset, transform=None, text_length=77, truncate=True, p=0.15, alpha=0.5):
        """
        初始化动态遮蔽数据集
        Args:
            dataset: 原始数据集，每个样本格式为(pid, image_id, img_path, caption, sim)
            transform: 图像变换
            text_length: 文本最大长度
            truncate: 是否截断
            p: 基础遮蔽概率，默认0.15
            alpha: 抑制因子系数，默认0.5
        """
        super().__init__(dataset, transform, text_length, truncate)
        self.p = p  # 基础遮蔽概率
        self.alpha = alpha  # 抑制因子系数
        self.tokenizer = SimpleTokenizer()
    
    def __getitem__(self, index):
        """重写__getitem__方法，处理match-score"""
        pid, image_id, img_path, caption, sim = self.dataset[index]

         # 1. 确保所有必要字段有效
        if not (pid and image_id and img_path and caption):
            return self.__getitem__((index + 1) % len(self))
        
        # 2. 确保sim值有效
        if sim is None or not isinstance(sim, (int, float)):
            sim = 0.0
        sim = float(sim)
        
        # 3. 确保图像路径有效
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
            return self.__getitem__((index + 1) % len(self))
        
        
        # 读取并处理图像
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        # 处理文本
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_dynamic_masked_tokens_and_labels(caption_tokens.cpu().numpy(), sim)
        ori_tokens =  tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'caption_ids_ori':ori_tokens
        }
    
    def _build_dynamic_masked_tokens_and_labels(self, tokens, sim):
        """
        基于match-score的动态遮蔽实现
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3))  # 1 ~ 49405
        
        # 确保tokens是numpy数组
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens, dtype=np.int64)
        
        # 计算有效token数量
        valid_token_num = len(tokens)
        if 0 in tokens:
            valid_token_num = np.where(tokens == 0)[0][0] if len(np.where(tokens == 0)[0]) > 0 else len(tokens)
        
        # 初始化原始遮蔽概率r
        r = np.ones_like(tokens, dtype=np.float32) * 0.15
        
        # 计算期望E[r]
        E_r = np.mean(r[1:valid_token_num-1]) if valid_token_num > 2 else 0.15
        
        # 计算抑制因子 e^(-alpha * sim)
        suppression_factor = np.exp(-self.alpha * sim)
        
        # 应用公式：\(\tilde{r} = r - E[r] + p * e^{-\alpha * sim}\)
        r = r - E_r + self.p * suppression_factor
        
        # 确保概率在[0, 1]范围内
        r = np.clip(r, 0, 1)
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = np.random.random()
                if prob < r[i]:
                    prob /= r[i]
                    if prob < 0.8:
                        tokens[i] = mask
                    elif prob < 0.9:
                        tokens[i] = np.random.choice(token_range)
                    labels.append(token)
                else:
                    labels.append(0)
            else:
                labels.append(0)
        
        # 确保至少有一个token被遮蔽
        if all(l == 0 for l in labels):
            labels[1] = tokens[1]
            tokens[1] = mask
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)