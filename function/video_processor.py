import cv2
import os
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    """
    视频处理器，用于提取视频帧和处理视频数据
    """
    def __init__(self):
        pass
    
    def extract_frames(self, video_path, output_folder, frame_interval=1):
        """
        从视频中提取帧并保存为图像
        
        Args:
            video_path: 视频文件路径
            output_folder: 输出文件夹路径
            frame_interval: 帧提取间隔，默认为1（提取所有帧）
            
        Returns:
            提取的帧的数量
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        saved_count = 0
        
        with tqdm(total=total_frames, desc="提取视频帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔保存帧
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        print(f"从 {video_path} 提取了 {saved_count} 帧，保存到 {output_folder}")
        return saved_count
    
    def process_multiple_videos(self, video_paths, output_root_folder, frame_interval=1):
        """
        处理多个视频文件
        
        Args:
            video_paths: 视频文件路径列表
            output_root_folder: 输出根文件夹
            frame_interval: 帧提取间隔
            
        Returns:
            每个视频提取的帧数字典
        """
        if not os.path.exists(output_root_folder):
            os.makedirs(output_root_folder)
        
        results = {}
        
        for i, video_path in enumerate(video_paths):
            output_folder = os.path.join(output_root_folder, f"camera_{i}")
            frames_count = self.extract_frames(video_path, output_folder, frame_interval)
            results[video_path] = frames_count
        
        return results
    
    def load_frame(self, frame_path):
        """
        加载单个帧
        
        Args:
            frame_path: 帧图像路径
            
        Returns:
            OpenCV格式的图像（BGR）
        """
        return cv2.imread(frame_path)
    
    def save_video_from_frames(self, frames_folder, output_path, fps=30):
        """
        从帧图像创建视频
        
        Args:
            frames_folder: 包含帧图像的文件夹
            output_path: 输出视频路径
            fps: 视频帧率
        """
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            raise ValueError(f"文件夹中没有找到图像文件: {frames_folder}")
        
        # 获取第一帧的尺寸
        first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_file in tqdm(frame_files, desc="创建视频"):
            frame = cv2.imread(os.path.join(frames_folder, frame_file))
            out.write(frame)
        
        out.release()
        print(f"视频已保存到: {output_path}")

# 示例用法
if __name__ == "__main__":
    processor = VideoProcessor()
    # processor.extract_frames("sample.mp4", "output_frames", frame_interval=5)