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
from tqdm import tqdm
from typing import List, Dict, Optional

# 导入自定义模块
from reid_model import ReIDModel
from tracker_adapter import CrossCameraTracker

# 导入YOLO模型
from ultralytics import YOLO

class CrossCameraReIDTracker:
    def __init__(self, config_file):
        """初始化跨摄像头ReID追踪系统
        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_file)
        
        # 初始化YOLO模型
        self.detector = self._init_detector()
        
        # 初始化ReID模型
        self.reid_model = self._init_reid_model()
        
        # 初始化跨摄像头跟踪器
        self.cross_camera_tracker = self._init_cross_camera_tracker()
        
        # 准备输出目录
        self._prepare_output_dir()
    
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
        """初始化跨摄像头跟踪器"""
        cross_camera_config = self.config['cross_camera']
        return CrossCameraTracker(
            reid_model=self.reid_model,
            similarity_thresh=cross_camera_config['similarity_thresh'],
            use_text_filter=cross_camera_config['use_text_filter']
        )
    
    def _prepare_output_dir(self):
        """准备输出目录"""
        output_dir = self.config['output']['save_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def track_video(self, video_path: str, camera_id: str, text_description: Optional[str] = None):
        """跟踪单个视频
        Args:
            video_path: 视频文件路径
            camera_id: 摄像头ID
            text_description: 文本描述（可选）
        Returns:
            处理后的轨迹信息
        """
        assert os.path.exists(video_path), f"视频文件不存在: {video_path}"
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 准备输出视频
        output_config = self.config['output']
        if output_config['save_video']:
            output_video_path = os.path.join(output_config['output_dir'], f"{camera_id}_output.mp4")
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
        
        # 处理视频帧
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                                'class_id': int(class_ids[i])
                            }
                            frame_tracks.append(track)
                
                # 使用跨摄像头跟踪器更新全局ID
                updated_tracks = self.cross_camera_tracker.update_tracks(
                    camera_id=camera_id,
                    tracks=frame_tracks,
                    frame=frame
                )
                
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
        
        # 保存跟踪结果
        if output_config['save_results']:
            results_path = os.path.join(output_config['save_dir'], f"{camera_id}_results.txt")
            self._save_tracking_results(results_path, all_tracks)
        
        return all_tracks
    
    def _visualize_tracks(self, frame, tracks, camera_id):
        """可视化跟踪结果"""
        for track in tracks:
            # 绘制边界框
            x1, y1, x2, y2 = map(int, track['bbox'])
            global_id = track['global_id']
            
            # 根据全局ID生成一致的颜色
            color = self._get_color_for_id(global_id)
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {global_id} (Cam: {camera_id})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加摄像头ID水印
        cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
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
            print(f"Processing video {video_path} (Camera: {camera_id})...")
            results = self.track_video(video_path, camera_id, text_desc)
            all_results[camera_id] = results
        
        # 打印统计信息
        stats = self.cross_camera_tracker.get_track_statistics()
        print("\nTracking Statistics:")
        print(f"Total Global IDs: {stats['total_global_ids']}")
        print("Cameras:")
        for camera_id, count in stats['cameras'].items():
            print(f"  {camera_id}: {count} tracks")
        print("Global IDs appearing in multiple cameras:")
        for global_id, cameras in stats['global_id_cameras'].items():
            if len(cameras) > 1:
                print(f"  ID {global_id}: {', '.join(cameras)}")
        
        return all_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Cross-Camera Person Tracking with ReID")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--videos', type=str, nargs='+', help='视频文件路径列表')
    parser.add_argument('--cameras', type=str, nargs='+', help='摄像头ID列表')
    parser.add_argument('--texts', type=str, nargs='*', help='文本描述列表')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 初始化系统
    tracker = CrossCameraReIDTracker(args.config)
    
    # 处理视频
    if args.videos and args.cameras:
        tracker.process_multiple_videos(args.videos, args.cameras, args.texts)
    else:
        print("请提供视频文件路径和摄像头ID")

if __name__ == "__main__":
    main()