import numpy as np
import cv2
import os
import time
from collections import defaultdict
import sys

class BYTETracker:
    """
    基于ByteTrack的行人追踪器
    参考ByteTrack算法实现高效的多目标追踪
    """
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, min_box_area=10):
        """
        初始化ByteTrack追踪器
        
        Args:
            track_thresh: 检测阈值，用于区分高分检测和低分检测
            track_buffer: 追踪缓冲区大小，决定目标消失后保留多长时间
            match_thresh: 匹配阈值，用于关联检测和追踪
            min_box_area: 最小边界框面积，过滤小目标
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.active_tracks = []  # 活跃追踪目标
        self.inactive_tracks = []  # 不活跃追踪目标
        self.next_id = 1
        self.frame_id = 0
        
        # 全局ID映射，用于跨摄像头追踪
        self.global_id_map = {}
        self.global_id_counter = 1
        
        # 历史轨迹记录
        self.track_history = defaultdict(list)
    
    class TrackObject:
        """
        追踪目标对象
        """
        def __init__(self, track_id, bbox, score, feature=None):
            self.track_id = track_id
            self.global_track_id = None  # 跨摄像头的全局ID
            self.bbox = bbox  # [x1, y1, x2, y2]
            self.score = score
            self.feature = feature  # 用于ReID的特征向量
            self.frame_id = 0
            self.time_since_update = 0
            self.hits = 1
            self.last_seen = time.time()
            
            # 计算中心点和边界框大小
            self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        def update(self, bbox, score, feature=None):
            """
            更新追踪目标
            """
            self.bbox = bbox
            self.score = score
            if feature is not None:
                self.feature = feature
            self.time_since_update = 0
            self.hits += 1
            self.last_seen = time.time()
            self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        def predict(self):
            """
            预测下一帧位置（简单版本）
            """
            self.time_since_update += 1
    
    def iou(self, bbox1, bbox2):
        """
        计算两个边界框的IoU
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def cosine_similarity(self, feat1, feat2):
        """
        计算两个特征向量的余弦相似度
        """
        if feat1 is None or feat2 is None:
            return 0
        
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
    
    def associate_detections_to_tracks(self, detections, tracks, use_feature=False):
        """
        将检测结果关联到追踪目标
        使用IoU或特征相似度进行匹配
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 计算代价矩阵
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                if use_feature and det.get('feature') is not None and track.feature is not None:
                    # 使用特征相似度
                    sim = self.cosine_similarity(det['feature'], track.feature)
                    cost_matrix[i, j] = 1 - sim  # 转换为代价
                else:
                    # 使用IoU
                    iou_val = self.iou(det['bbox'], track.bbox)
                    cost_matrix[i, j] = 1 - iou_val  # 转换为代价
        
        # 使用匈牙利算法进行匹配
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # 筛选匹配结果
        for i, j in zip(row_ind, col_ind):
            if use_feature:
                # 使用特征相似度阈值
                if 1 - cost_matrix[i, j] > self.match_thresh:
                    matches.append((i, j))
                    unmatched_detections.remove(i)
                    unmatched_tracks.remove(j)
            else:
                # 使用IoU阈值
                if 1 - cost_matrix[i, j] > self.match_thresh:
                    matches.append((i, j))
                    unmatched_detections.remove(i)
                    unmatched_tracks.remove(j)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections, features=None):
        """
        更新追踪器状态
        
        Args:
            detections: 检测结果，ByteTrack格式 (x1, y1, x2, y2, score)
            features: 对应的特征向量列表
        
        Returns:
            当前帧中活跃的追踪目标列表
        """
        self.frame_id += 1
        
        # 将检测结果转换为字典格式
        dets = []
        for i, det in enumerate(detections):
            det_dict = {
                'bbox': det[:4].tolist(),
                'score': det[4]
            }
            if features is not None and i < len(features):
                det_dict['feature'] = features[i]
            dets.append(det_dict)
        
        # 分离高分和低分检测
        high_score_dets = [d for d in dets if d['score'] >= self.track_thresh]
        low_score_dets = [d for d in dets if d['score'] < self.track_thresh and d['score'] >= 0.1]
        
        # 预测所有活跃追踪目标
        for track in self.active_tracks:
            track.predict()
        
        # 使用高分检测匹配活跃追踪目标
        matches, unmatched_dets, unmatched_tracks = self.associate_detections_to_tracks(
            high_score_dets, self.active_tracks, use_feature=features is not None)
        
        # 更新匹配的追踪目标
        for det_idx, track_idx in matches:
            det = high_score_dets[det_idx]
            self.active_tracks[track_idx].update(
                det['bbox'], det['score'], det.get('feature'))
        
        # 将未匹配的活跃追踪目标移到不活跃列表
        for track_idx in unmatched_tracks:
            track = self.active_tracks[track_idx]
            self.inactive_tracks.append(track)
        
        # 移除未匹配的活跃追踪目标
        self.active_tracks = [self.active_tracks[i] for i in range(len(self.active_tracks)) 
                             if i not in unmatched_tracks]
        
        # 使用低分检测匹配不活跃追踪目标
        if low_score_dets:
            matches, unmatched_dets, unmatched_inactive = self.associate_detections_to_tracks(
                low_score_dets, self.inactive_tracks, use_feature=False)
            
            # 更新匹配的不活跃追踪目标
            for det_idx, track_idx in matches:
                det = low_score_dets[det_idx]
                track = self.inactive_tracks[track_idx]
                track.update(det['bbox'], det['score'], det.get('feature'))
                self.active_tracks.append(track)
            
            # 更新不活跃列表
            self.inactive_tracks = [self.inactive_tracks[i] for i in range(len(self.inactive_tracks)) 
                                  if i not in unmatched_inactive]
        
        # 创建新的追踪目标
        for det in [high_score_dets[i] for i in unmatched_dets]:
            # 过滤小目标
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_box_area:
                continue
            
            # 创建新的追踪目标
            new_track = self.TrackObject(self.next_id, det['bbox'], det['score'], det.get('feature'))
            self.next_id += 1
            self.active_tracks.append(new_track)
        
        # 移除长时间未更新的不活跃追踪目标
        self.inactive_tracks = [t for t in self.inactive_tracks if t.time_since_update < self.track_buffer]
        
        # 记录历史轨迹
        for track in self.active_tracks:
            self.track_history[track.track_id].append({
                'frame': self.frame_id,
                'bbox': track.bbox,
                'score': track.score,
                'time': time.time()
            })
        
        # 返回活跃的追踪目标
        return self.active_tracks
    
    def assign_global_ids(self, reid_features=None, sim_threshold=0.5):
        """
        为追踪目标分配全局ID，用于跨摄像头追踪
        
        Args:
            reid_features: 用于ReID的特征向量字典 {track_id: feature}
            sim_threshold: 特征相似度阈值
        """
        for track in self.active_tracks:
            # 如果已经有全局ID，跳过
            if track.global_track_id is not None:
                continue
            
            # 如果没有特征，直接分配新的全局ID
            if reid_features is None or track.track_id not in reid_features:
                track.global_track_id = self.global_id_counter
                self.global_id_map[track.track_id] = self.global_id_counter
                self.global_id_counter += 1
                continue
            
            # 查找相似的已有全局ID
            current_feature = reid_features[track.track_id]
            best_match_id = None
            best_sim = 0
            
            for g_id in set(self.global_id_map.values()):
                # 查找该全局ID对应的所有track_id
                track_ids = [tid for tid, gid in self.global_id_map.items() if gid == g_id]
                for tid in track_ids:
                    if tid in reid_features:
                        sim = self.cosine_similarity(current_feature, reid_features[tid])
                        if sim > best_sim and sim > sim_threshold:
                            best_sim = sim
                            best_match_id = g_id
            
            # 分配全局ID
            if best_match_id is not None:
                track.global_track_id = best_match_id
                self.global_id_map[track.track_id] = best_match_id
            else:
                track.global_track_id = self.global_id_counter
                self.global_id_map[track.track_id] = self.global_id_counter
                self.global_id_counter += 1
    
    def get_active_tracks(self):
        """
        获取当前活跃的追踪目标
        """
        return self.active_tracks
    
    def get_track_history(self, track_id):
        """
        获取指定追踪目标的历史轨迹
        """
        return self.track_history.get(track_id, [])
    
    def get_all_history(self):
        """
        获取所有追踪目标的历史轨迹
        """
        return dict(self.track_history)
    
    def reset(self):
        """
        重置追踪器
        """
        self.active_tracks = []
        self.inactive_tracks = []
        self.next_id = 1
        self.frame_id = 0

class MultiCameraTracker:
    """
    多摄像头追踪器，用于管理多个摄像头的追踪
    """
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        """
        初始化多摄像头追踪器
        
        Args:
            track_thresh: 检测阈值
            track_buffer: 追踪缓冲区
            match_thresh: 匹配阈值
        """
        self.trackers = {}
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        # 全局ID映射
        self.global_id_map = {}
        self.global_counter = 1
        
        # 摄像头间的相似度缓存
        self.camera_similarities = defaultdict(dict)
    
    def get_or_create_tracker(self, camera_id):
        """
        获取或创建指定摄像头的追踪器
        """
        if camera_id not in self.trackers:
            self.trackers[camera_id] = BYTETracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh
            )
        return self.trackers[camera_id]
    
    def update_camera(self, camera_id, detections, features=None):
        """
        更新指定摄像头的追踪结果
        """
        tracker = self.get_or_create_tracker(camera_id)
        tracks = tracker.update(detections, features)
        return tracks
    
    def associate_cross_camera(self, camera1_id, camera2_id, reid_features1, reid_features2, sim_threshold=0.6):
        """
        关联两个摄像头之间的追踪目标
        """
        from scipy.optimize import linear_sum_assignment
        
        tracker1 = self.get_or_create_tracker(camera1_id)
        tracker2 = self.get_or_create_tracker(camera2_id)
        
        # 获取两个摄像头的活跃追踪目标
        tracks1 = tracker1.get_active_tracks()
        tracks2 = tracker2.get_active_tracks()
        
        # 过滤没有特征的追踪目标
        tracks1_with_feat = [t for t in tracks1 if t.track_id in reid_features1]
        tracks2_with_feat = [t for t in tracks2 if t.track_id in reid_features2]
        
        if not tracks1_with_feat or not tracks2_with_feat:
            return []
        
        # 计算代价矩阵
        cost_matrix = np.zeros((len(tracks1_with_feat), len(tracks2_with_feat)))
        
        for i, track1 in enumerate(tracks1_with_feat):
            feat1 = reid_features1[track1.track_id]
            for j, track2 in enumerate(tracks2_with_feat):
                feat2 = reid_features2[track2.track_id]
                sim = self._cosine_similarity(feat1, feat2)
                cost_matrix[i, j] = 1 - sim
        
        # 使用匈牙利算法进行匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 筛选匹配结果
        matches = []
        for i, j in zip(row_ind, col_ind):
            if 1 - cost_matrix[i, j] > sim_threshold:
                matches.append((tracks1_with_feat[i], tracks2_with_feat[j], 1 - cost_matrix[i, j]))
        
        # 更新全局ID
        self._update_global_ids(matches)
        
        return matches
    
    def _cosine_similarity(self, feat1, feat2):
        """
        计算余弦相似度
        """
        if isinstance(feat1, list):
            feat1 = np.array(feat1)
        if isinstance(feat2, list):
            feat2 = np.array(feat2)
        
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        return np.dot(feat1, feat2)
    
    def _update_global_ids(self, matches):
        """
        根据匹配结果更新全局ID
        """
        for track1, track2, _ in matches:
            # 如果track1有全局ID，给track2分配相同的ID
            if track1.global_track_id is not None:
                track2.global_track_id = track1.global_track_id
                self.global_id_map[track2.track_id] = track1.global_track_id
            # 如果track2有全局ID，给track1分配相同的ID
            elif track2.global_track_id is not None:
                track1.global_track_id = track2.global_track_id
                self.global_id_map[track1.track_id] = track2.global_track_id
            # 否则分配新的全局ID
            else:
                new_id = self.global_counter
                track1.global_track_id = new_id
                track2.global_track_id = new_id
                self.global_id_map[track1.track_id] = new_id
                self.global_id_map[track2.track_id] = new_id
                self.global_counter += 1
    
    def get_all_tracks(self):
        """
        获取所有摄像头的追踪目标
        """
        all_tracks = {}
        for camera_id, tracker in self.trackers.items():
            all_tracks[camera_id] = tracker.get_active_tracks()
        return all_tracks
    
    def reset_camera(self, camera_id):
        """
        重置指定摄像头的追踪器
        """
        if camera_id in self.trackers:
            self.trackers[camera_id].reset()
    
    def reset_all(self):
        """
        重置所有追踪器
        """
        for tracker in self.trackers.values():
            tracker.reset()
        self.global_id_map = {}
        self.global_counter = 1