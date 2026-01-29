import os.path as op
from typing import List
from utils.iotools import read_json
from .bases import BaseDataset

class RSTPReid_match(BaseDataset):
    """
    RSTPReid with match_score support for finetuning.
    """
    dataset_dir = 'RSTPReid'

    def __init__(self, root='', verbose=True):
        super(RSTPReid_match, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        # 指定包含 match_score 的特定 JSON 文件
        self.anno_path = op.join(self.dataset_dir, 'data_caption_all_qwen.json')
        
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        # 训练集：需要读取 match_score
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        # 测试/验证集：保持标准格式，不需要 match_score
        self.test, self.test_id_container = self._process_anno(self.test_annos, training=False)
        self.val, self.val_id_container = self._process_anno(self.val_annos, training=False)

        if verbose:
            self.logger.info("=> RSTPReid (Match Version) Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        
        # 1. 建立 PID 映射 (pid -> label)
        for anno in annos:
            pid = int(anno['id'])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_label = pid2label[pid]
                img_path = op.join(self.img_dir, anno['img_path'])
                captions = anno['captions']  # caption list
                
                # 获取 match_score，做防错处理
                match_scores = anno.get('match_score', [0.85] * len(captions))
                
                # 如果 match_score 只是一个单独的 float (有的json格式可能不同)，转为列表
                if not isinstance(match_scores, list):
                    match_scores = [match_scores] * len(captions)
                
                # 如果长度不一致，进行填充或截断
                if len(match_scores) < len(captions):
                    match_scores = match_scores + [0.85] * (len(captions) - len(match_scores))
                # 遍历每个 caption 和对应的 score
                for caption, score in zip(captions, match_scores):
                    try:
                        match_score_float = float(score)
                    except (ValueError, TypeError):
                        match_score_float = 0.85 # 默认高质量

                    # 核心修改：返回 5 元组
                    dataset.append((pid_label, image_id, img_path, caption, match_score_float))
                
                image_id += 1
            
            return dataset, pid_container
        
        else:
            # 验证/测试集处理：返回字典格式，供 Evaluator 使用
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            
            for anno in annos:
                pid = int(anno['id'])
                pid_label = pid2label[pid]
                img_path = op.join(self.img_dir, anno['img_path'])
                
                img_paths.append(img_path)
                image_pids.append(pid_label)
                
                caption_list = anno['captions']
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid_label)
            
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))