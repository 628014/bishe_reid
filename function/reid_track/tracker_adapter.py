import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional

class CrossCameraTracker:
    def __init__(self, reid_model, similarity_thresh=0.5, use_text_filter=True):
        """初始化跨摄像头跟踪器
        Args:
            reid_model: ReID模型实例
            similarity_thresh: 跨摄像头ID匹配的相似度阈值
            use_text_filter: 是否使用文本描述进行过滤
        """
        self.reid_model = reid_model
        self.similarity_thresh = similarity_thresh
        self.use_text_filter = use_text_filter
        
        # 存储每个摄像头的轨迹信息
        self.camera_tracks = {}
        
        # 全局ID映射表 (camera_id + local_track_id) -> global_track_id
        self.global_id_map = {}
        
        # 存储全局ID的特征信息
        self.global_features = {}
        
        # 当前最大全局ID
        self.current_global_id = 0
        
        # 文本描述缓存
        self.text_descriptions = {}
    
    def register_text_description(self, global_id, text):
        """为全局ID注册文本描述"""
        self.text_descriptions[global_id] = text
    
    def get_image_from_bbox(self, frame, bbox):
        """从帧中裁剪出边界框区域
        Args:
            frame: 原始图像
            bbox: 边界框 [x1, y1, x2, y2]
        Returns:
            PIL图像
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保边界框在图像范围内
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪并转换为PIL图像
        crop = frame[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        return crop_pil
    
    def get_global_id(self, camera_id, local_track_id, frame, bbox):
        """获取或分配全局ID
        Args:
            camera_id: 摄像头ID
            local_track_id: 本地跟踪ID
            frame: 原始帧
            bbox: 边界框 [x1, y1, x2, y2]
        Returns:
            全局ID
        """
        # 生成唯一键
        key = f"{camera_id}_{local_track_id}"
        
        # 如果已经有映射，直接返回
        if key in self.global_id_map:
            return self.global_id_map[key]
        
        # 提取目标特征
        crop_pil = self.get_image_from_bbox(frame, bbox)
        feature = self.reid_model.extract_image_feature(crop_pil)
        
        if feature is None:
            # 如果无法提取特征，分配新ID
            return self._assign_new_global_id(key)
        
        # 尝试匹配现有全局ID
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, global_feat in self.global_features.items():
            similarity = self.reid_model.calculate_similarity(feature, global_feat)
            
            # 如果有文本描述，使用文本过滤
            if self.use_text_filter and global_id in self.text_descriptions:
                text_sim = self.reid_model.image_text_similarity(crop_pil, self.text_descriptions[global_id])
                # 综合考虑图像相似度和文本相似度
                similarity = (similarity + text_sim) / 2
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = global_id
        
        # 如果相似度超过阈值，使用现有ID
        if best_similarity >= self.similarity_thresh:
            self.global_id_map[key] = best_match_id
            # 更新特征（使用指数移动平均）
            alpha = 0.3
            self.global_features[best_match_id] = alpha * feature + (1 - alpha) * self.global_features[best_match_id]
            return best_match_id
        
        # 否则分配新ID
        return self._assign_new_global_id(key, feature)
    
    def _assign_new_global_id(self, key, feature=None):
        """分配新的全局ID
        Args:
            key: 唯一键
            feature: 特征向量
        Returns:
            新的全局ID
        """
        self.current_global_id += 1
        global_id = self.current_global_id
        self.global_id_map[key] = global_id
        
        if feature is not None:
            self.global_features[global_id] = feature
        
        return global_id
    
    def update_tracks(self, camera_id: str, tracks: List[Dict], frame: np.ndarray):
        """更新轨迹信息并分配全局ID
        Args:
            camera_id: 摄像头ID
            tracks: 轨迹列表，每个元素包含 'id', 'bbox', 'confidence', 'class_id'
            frame: 当前帧
        Returns:
            更新后的轨迹列表，包含全局ID
        """
        updated_tracks = []
        
        for track in tracks:
            # 获取全局ID
            global_id = self.get_global_id(
                camera_id=camera_id,
                local_track_id=track['id'],
                frame=frame,
                bbox=track['bbox']
            )
            
            # 添加全局ID到轨迹信息
            updated_track = track.copy()
            updated_track['global_id'] = global_id
            updated_tracks.append(updated_track)
        
        # 更新摄像头轨迹信息
        self.camera_tracks[camera_id] = updated_tracks
        
        return updated_tracks
    
    def get_track_statistics(self):
        """获取轨迹统计信息
        Returns:
            统计信息字典
        """
        stats = {
            'total_global_ids': self.current_global_id,
            'cameras': {},
            'global_id_cameras': {}
        }
        
        # 统计每个摄像头的轨迹数
        for camera_id, tracks in self.camera_tracks.items():
            stats['cameras'][camera_id] = len(tracks)
            
            # 统计每个全局ID出现在哪些摄像头
            for track in tracks:
                global_id = track['global_id']
                if global_id not in stats['global_id_cameras']:
                    stats['global_id_cameras'][global_id] = []
                if camera_id not in stats['global_id_cameras'][global_id]:
                    stats['global_id_cameras'][global_id].append(camera_id)
        
        return stats