import os
# 启用OpenCV无头模式，避免Qt显示问题
os.environ['PYOPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['DISPLAY'] = ':0'
# 使用更基础的设置，避免依赖特定的Qt平台插件

import cv2
cv2.setNumThreads(0)
import yaml
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CrossCameraTracker')

# 导入自定义模块
from reid_model import ReIDModel
from tracker_adapter import CrossCameraTracker

# 导入YOLO模型
from ultralytics import YOLO

class EnhancedCrossCameraReIDTracker:
    def __init__(self, config_file):
        """初始化增强版跨摄像头ReID追踪系统
        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_file)
        
        # 初始化YOLO模型
        self.detector = self._init_detector()
        
        # 初始化ReID模型
        self.reid_model = self._init_reid_model()
        
        # 初始化跨摄像头跟踪器 - 使用更低的相似度阈值以提高匹配率
        self.cross_camera_tracker = self._init_cross_camera_tracker()
        
        # 准备输出目录
        self._prepare_output_dir()
        
        # 存储所有摄像头的历史跟踪数据
        self.all_camera_tracks = {}
        
        # 全局ID使用统计
        self.global_id_usage = {}
        
        # 调试模式
        self.debug = self.config.get('debug', False)
        
        # 配置优化间隔
        self.optimization_interval = self.config.get('optimization_interval', 30)
    
    def _load_config(self, config_file):
        """加载配置文件"""
        assert os.path.exists(config_file), f"配置文件不存在: {config_file}"
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _init_detector(self):
        """初始化YOLO检测器"""
        # 加载YOLO模型
        detector_config = self.config['detection']
        model_path = detector_config['model_path']
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"检测模型文件不存在: {model_path}，尝试下载默认模型")
            # 使用Ultralytics的默认路径
            model = YOLO(detector_config['model_type'])
        else:
            model = YOLO(model_path)
        
        return model
    
    def _init_reid_model(self):
        """初始化ReID模型"""
        reid_config = self.config['reid']
        return ReIDModel(
            config_file=reid_config['config_file'],
            checkpoint_path=reid_config['checkpoint_path'],
            img_size=reid_config['img_size'],
            device=reid_config['device']
        )
    
    def _init_cross_camera_tracker(self):
        """初始化跨摄像头跟踪器 - 使用更低的相似度阈值以提高匹配率"""
        cross_camera_config = self.config['cross_camera']
        # 降低相似度阈值以提高跨摄像头匹配成功率
        similarity_thresh = max(0.3, cross_camera_config.get('similarity_thresh', 0.5) * 0.8)
        logger.info(f"使用相似度阈值: {similarity_thresh}")
        return CrossCameraTracker(
            reid_model=self.reid_model,
            similarity_thresh=similarity_thresh,
            use_text_filter=cross_camera_config.get('use_text_filter', True)
        )
    
    def _prepare_output_dir(self):
        """准备输出目录"""
        output_dir = self.config['output']['save_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def track_video(self, video_path: str, camera_id: str, text_description: Optional[str] = None):
        """增强版视频跟踪 - 改进跨摄像头ID一致性
        Args:
            video_path: 视频文件路径
            camera_id: 摄像头ID
            text_description: 文本描述（可选）
        Returns:
            处理后的轨迹信息
        """
        assert os.path.exists(video_path), f"视频文件不存在: {video_path}"
        logger.info(f"开始处理摄像头 {camera_id} 的视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 准备输出视频
        output_config = self.config['output']
        if output_config['save_video']:
            output_video_path = os.path.join(output_config['save_dir'], f"{camera_id}_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # 准备检测器参数
        detector_config = self.config['detection']
        classes = detector_config.get('classes', [0])  # 默认只检测人
        conf_threshold = detector_config['conf']
        iou_threshold = detector_config['iou']
        device = detector_config['device']

        # 跟踪结果存储
        all_tracks = []
        
        # 帧计数器，用于定期执行跨摄像头匹配优化
        frame_count = 0
        
        # 处理视频帧
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 使用YOLO进行检测和跟踪
                results = self.detector.track(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=classes,
                    persist=True,
                    verbose=False
                )
                
                # 处理跟踪结果
                frame_tracks = []
                for result in results:
                    # 检查是否有跟踪ID
                    if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()  # 边界框
                        ids = result.boxes.id.cpu().numpy()  # 跟踪ID
                        confs = result.boxes.conf.cpu().numpy()  # 置信度
                        class_ids = result.boxes.cls.cpu().numpy()  # 类别ID
                        
                        # 转换为字典格式
                        for i in range(len(boxes)):
                            track = {
                                'id': int(ids[i]),
                                'bbox': boxes[i].tolist(),
                                'confidence': float(confs[i]),
                                'class_id': int(class_ids[i]),
                                'camera_id': camera_id,  # 添加摄像头ID
                                'frame_id': int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 添加帧ID
                            }
                            frame_tracks.append(track)
                
                # 使用跨摄像头跟踪器更新全局ID - 关键步骤
                updated_tracks = self.cross_camera_tracker.update_tracks(
                    camera_id=camera_id,
                    tracks=frame_tracks,
                    frame=frame
                )
                
                # 增强：记录每个全局ID的使用情况
                for track in updated_tracks:
                    global_id = track['global_id']
                    if global_id not in self.global_id_usage:
                        self.global_id_usage[global_id] = {'cameras': set(), 'count': 0}
                    self.global_id_usage[global_id]['cameras'].add(camera_id)
                    self.global_id_usage[global_id]['count'] += 1
                
                # 存储结果
                all_tracks.append({
                    'frame_id': int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                    'tracks': updated_tracks
                })
                
                # 可视化结果
                if output_config['save_video'] or output_config['show_video']:
                    self._visualize_tracks(frame, updated_tracks, camera_id)
                
                # 保存视频帧
                if output_config['save_video']:
                    out.write(frame)
                
                # 显示视频
                if output_config['show_video']:
                    cv2.imshow(f"Tracking - {camera_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
                
                # 定期执行跨摄像头ID一致性优化
                if frame_count % self.optimization_interval == 0:
                    self._optimize_global_ids(camera_id, updated_tracks, frame)
        
        # 保存该摄像头的所有跟踪结果
        self.all_camera_tracks[camera_id] = all_tracks
        
        # 最终的全局ID优化
        self._optimize_global_ids(camera_id, [t for ft in all_tracks for t in ft['tracks']], frame)
        
        # 释放资源
        cap.release()
        if output_config['save_video']:
            out.release()
        if output_config['show_video']:
            cv2.destroyAllWindows()
        
        # 如果有文本描述，注册到最近的跟踪ID
        if text_description:
            # 获取该摄像头最新的跟踪结果
            if all_tracks and all_tracks[-1]['tracks']:
                # 选择置信度最高的跟踪目标
                latest_track = max(all_tracks[-1]['tracks'], key=lambda x: x['confidence'])
                self.cross_camera_tracker.register_text_description(
                    latest_track['global_id'],
                    text_description
                )
                logger.info(f"为全局ID {latest_track['global_id']} 注册文本描述")
        
        # 保存跟踪结果
        if output_config['save_results']:
            results_path = os.path.join(output_config['save_dir'], f"{camera_id}_results.txt")
            self._save_tracking_results(results_path, all_tracks)
        
        logger.info(f"完成摄像头 {camera_id} 的视频处理")
        return all_tracks
    
    def _optimize_global_ids(self, camera_id, current_tracks, frame):
        """优化全局ID分配，确保跨摄像头一致性
        Args:
            camera_id: 当前摄像头ID
            current_tracks: 当前帧的跟踪结果
            frame: 当前帧图像
        """
        if not current_tracks or len(self.all_camera_tracks) < 2:
            return
        
        # 获取其他摄像头的历史跟踪数据
        other_cameras_data = []
        for cam_id, cam_tracks in self.all_camera_tracks.items():
            if cam_id != camera_id and cam_tracks:
                # 获取该摄像头最近的几个关键帧
                recent_frames = cam_tracks[-min(5, len(cam_tracks)):]
                for frame_data in recent_frames:
                    for track in frame_data['tracks']:
                        if track['confidence'] > 0.7:  # 只考虑高置信度的跟踪
                            other_cameras_data.append(track)
        
        # 对当前摄像头的每个跟踪目标，尝试与其他摄像头的进行匹配
        for current_track in current_tracks:
            if current_track['confidence'] < 0.7:  # 跳过低置信度的跟踪
                continue
            
            current_global_id = current_track['global_id']
            current_bbox = current_track['bbox']
            
            # 提取当前目标的特征
            current_crop = self.cross_camera_tracker.get_image_from_bbox(frame, current_bbox)
            current_feature = self.reid_model.extract_image_feature(current_crop)
            
            if current_feature is None:
                continue
            
            # 查找最佳匹配
            best_match = None
            best_similarity = 0.0
            
            for other_track in other_cameras_data:
                # 提取其他目标的特征
                other_bbox = other_track['bbox']
                # 使用跟踪器的全局特征库进行比较
                other_global_id = other_track['global_id']
                if hasattr(self.cross_camera_tracker, 'global_features') and other_global_id in self.cross_camera_tracker.global_features:
                    other_feature = self.cross_camera_tracker.global_features[other_global_id]
                    similarity = self.reid_model.calculate_similarity(current_feature, other_feature)
                    
                    # 如果相似度足够高且比当前最佳匹配好
                    if similarity > best_similarity and similarity > self.cross_camera_tracker.similarity_thresh * 1.2:
                        best_similarity = similarity
                        best_match = other_global_id
            
            # 如果找到好的匹配，更新ID映射
            if best_match and best_match != current_global_id:
                # 合并两个ID
                key = f"{camera_id}_{current_track['id']}"
                old_global_id = current_global_id
                if hasattr(self.cross_camera_tracker, 'global_id_map'):
                    old_global_id = self.cross_camera_tracker.global_id_map.get(key, current_global_id)
                
                if self.debug:
                    logger.info(f"合并ID: {old_global_id} -> {best_match} (相似度: {best_similarity:.3f})")
                
                # 更新ID映射
                if hasattr(self.cross_camera_tracker, 'global_id_map'):
                    self.cross_camera_tracker.global_id_map[key] = best_match
                
                # 更新特征（如果可以访问）
                if hasattr(self.cross_camera_tracker, 'global_features'):
                    if old_global_id in self.cross_camera_tracker.global_features and best_match in self.cross_camera_tracker.global_features:
                        # 保留使用更广泛的ID
                        usage_old = self.global_id_usage.get(old_global_id, {'count': 0})['count']
                        usage_best = self.global_id_usage.get(best_match, {'count': 0})['count']
                        
                        if usage_best > usage_old:
                            # 保留best_match，合并特征
                            alpha = 0.7  # 给现有特征更高权重
                            self.cross_camera_tracker.global_features[best_match] = (
                                alpha * self.cross_camera_tracker.global_features[best_match] +
                                (1 - alpha) * self.cross_camera_tracker.global_features[old_global_id]
                            )
                            # 从特征库中移除旧ID
                            del self.cross_camera_tracker.global_features[old_global_id]
        
        if self.debug:
            try:
                stats = self.cross_camera_tracker.get_track_statistics()
                logger.debug(f"跟踪统计: {stats}")
            except:
                logger.debug("无法获取跟踪统计信息")
    
    def _visualize_tracks(self, frame, tracks, camera_id):
        """增强版可视化跟踪结果"""
        for track in tracks:
            # 绘制边界框
            x1, y1, x2, y2 = map(int, track['bbox'])
            global_id = track['global_id']
            confidence = track['confidence']
            
            # 根据全局ID生成一致的颜色
            color = self._get_color_for_id(global_id)
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 增强的标签信息
            label = f"GlobalID: {global_id} (Cam: {camera_id})"
            confidence_label = f"Conf: {confidence:.2f}"
            
            # 绘制标签
            cv2.putText(frame, label, (x1, y1 - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, confidence_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加摄像头ID水印和全局统计
        cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 添加全局ID统计信息
        total_global_ids = len(self.global_id_usage)
        cv2.putText(frame, f"Global IDs: {total_global_ids}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _get_color_for_id(self, global_id):
        """根据全局ID生成一致的颜色"""
        # 使用ID生成种子，确保相同ID获得相同颜色
        np.random.seed(global_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def _save_tracking_results(self, results_path, all_tracks):
        """保存跟踪结果到文件"""
        with open(results_path, 'w') as f:
            for frame_data in all_tracks:
                frame_id = frame_data['frame_id']
                for track in frame_data['tracks']:
                    x1, y1, x2, y2 = track['bbox']
                    global_id = track['global_id']
                    confidence = track['confidence']
                    f.write(f"{frame_id},{global_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{confidence:.2f}\n")
    
    def process_multiple_videos(self, video_list, camera_ids, text_descriptions=None):
        """处理多个视频（多个摄像头）
        Args:
            video_list: 视频文件路径列表
            camera_ids: 摄像头ID列表
            text_descriptions: 文本描述列表（可选）
        Returns:
            所有视频的跟踪结果
        """
        assert len(video_list) == len(camera_ids), "视频列表和摄像头ID列表长度必须相同"
        
        if text_descriptions:
            assert len(video_list) == len(text_descriptions), "视频列表和文本描述列表长度必须相同"
        else:
            text_descriptions = [None] * len(video_list)
        
        all_results = {}
        
        # 处理每个视频
        for video_path, camera_id, text_desc in zip(video_list, camera_ids, text_descriptions):
            logger.info(f"处理视频 {video_path} (摄像头: {camera_id})...")
            results = self.track_video(video_path, camera_id, text_desc)
            all_results[camera_id] = results
        
        # 打印统计信息
        logger.info("\n跟踪统计信息:")
        logger.info(f"总全局ID数量: {len(self.global_id_usage)}")
        logger.info("跨摄像头出现的全局ID:")
        for global_id, usage in self.global_id_usage.items():
            if len(usage['cameras']) > 1:
                cameras_str = ", ".join(usage['cameras'])
                logger.info(f"  ID {global_id}: {cameras_str}")
        
        return all_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="增强版跨摄像头ReID跟踪系统")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--videos', type=str, nargs='+', help='视频文件路径列表')
    parser.add_argument('--cameras', type=str, nargs='+', help='摄像头ID列表')
    parser.add_argument('--texts', type=str, nargs='*', help='文本描述列表')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info(f"启动跟踪系统，配置文件: {args.config}")
    
    # 初始化增强版系统
    tracker = EnhancedCrossCameraReIDTracker(args.config)
    
    # 如果启用调试模式
    if args.debug:
        tracker.debug = True
        logger.setLevel(logging.DEBUG)
    
    # 处理视频
    if args.videos and args.cameras:
        tracker.process_multiple_videos(args.videos, args.cameras, args.texts)
        
        # 输出最终统计信息
        logger.info("跟踪完成，全局ID统计:")
        for global_id, usage in tracker.global_id_usage.items():
            cameras_str = ", ".join(usage['cameras'])
            logger.info(f"  全局ID {global_id}: 出现在摄像头 [{cameras_str}], 总计 {usage['count']} 次")
    else:
        print("请提供视频文件路径和摄像头ID")

if __name__ == "__main__":
    main()