import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import time
from datetime import datetime
import threading
import queue

# 添加项目根目录到Python路径
import sys
sys.path.append('/home/wangrui/code/MLLM4Text-ReID-main')
sys.path.append('/home/wangrui/code/MLLM4Text-ReID-main/function')

# 导入自定义模块
try:
    from person_detector import PersonDetector
    from tracker import BYTETracker, MultiCameraTracker
    from reid_extractor import ReIDExtractor, DetectionFeatureExtractor
    print("成功导入自定义模块")
except Exception as e:
    print(f"导入自定义模块时出错: {e}")

class MultiCameraReIDApp:
    """
    跨摄像头行人重识别应用
    支持两个摄像头视频的上传和跨摄像头行人追踪
    """
    def __init__(self):
        """
        初始化应用
        """
        self.detector = None
        self.tracker = None
        self.reid_extractor = None
        self.feature_extractor = None
        
        self.video_paths = {}
        self.processing = False
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue()
        
        self.initialize_components()
    
    def initialize_components(self):
        """
        初始化各个组件
        """
        try:
            # 初始化检测器
            self.detector = PersonDetector()
            print("行人检测器初始化完成")
            
            # 初始化多摄像头追踪器
            self.tracker = MultiCameraTracker(
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8
            )
            print("多摄像头追踪器初始化完成")
            
            # 初始化ReID特征提取器
            self.reid_extractor = ReIDExtractor()
            self.feature_extractor = DetectionFeatureExtractor(self.reid_extractor)
            print("ReID特征提取器初始化完成")
            
        except Exception as e:
            print(f"初始化组件时出错: {e}")
    
    def process_video(self, video_path, camera_id):
        """
        处理单个视频
        
        Args:
            video_path: 视频路径
            camera_id: 摄像头ID
            
        Returns:
            处理后的视频路径
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 创建临时输出视频
        output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = output_file.name
        output_file.close()
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理当前帧
            processed_frame, _ = self.process_frame(frame, camera_id)
            
            # 写入输出视频
            out.write(processed_frame)
            
            frame_count += 1
            # 更新进度
            progress = frame_count / total_frames * 100
            self.result_queue.put((camera_id, 'progress', progress))
            
        # 释放资源
        cap.release()
        out.release()
        
        return output_path
    
    def process_frame(self, frame, camera_id):
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            camera_id: 摄像头ID
            
        Returns:
            处理后的帧和追踪结果
        """
        # 复制原始帧
        processed_frame = frame.copy()
        
        try:
            # 检测行人
            detection_results = self.detector.detect_persons(frame)
            
            if detection_results and 'byte_track_dets' in detection_results:
                detections = detection_results['byte_track_dets']
                
                # 提取特征
                features = self.feature_extractor.extract_features_from_detections(
                    frame, detections
                )
                
                # 更新追踪器
                tracks = self.tracker.update_camera(
                    camera_id, 
                    np.array(detections), 
                    features
                )
                
                # 可视化追踪结果
                processed_frame = self.visualize_tracks(processed_frame, tracks)
                
                return processed_frame, tracks
        except Exception as e:
            print(f"处理帧时出错: {e}")
        
        return processed_frame, []
    
    def visualize_tracks(self, frame, tracks):
        """
        可视化追踪结果
        
        Args:
            frame: 输入帧
            tracks: 追踪结果列表
            
        Returns:
            可视化后的帧
        """
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # 绘制边界框
            color = self.get_track_color(track.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 显示ID
            track_id_text = f"ID: {track.track_id}"
            if hasattr(track, 'global_track_id') and track.global_track_id is not None:
                track_id_text += f" (Global: {track.global_track_id})"
            
            cv2.putText(frame, track_id_text, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
            
            # 显示置信度
            if hasattr(track, 'score'):
                score_text = f"Conf: {track.score:.2f}"
                cv2.putText(frame, score_text, 
                            (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, color, 1)
        
        return frame
    
    def get_track_color(self, track_id):
        """
        根据追踪ID生成颜色
        
        Args:
            track_id: 追踪ID
            
        Returns:
            BGR颜色值
        """
        # 使用ID生成一致的颜色
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3).tolist()))
    
    def perform_cross_camera_association(self, video_path1, video_path2, progress=gr.Progress()):
        """
        执行跨摄像头行人关联
        
        Args:
            video_path1: 第一个视频路径
            video_path2: 第二个视频路径
            progress: Gradio进度对象
            
        Returns:
            处理后的两个视频路径
        """
        # 重置状态
        self.stop_event.clear()
        self.result_queue = queue.Queue()
        self.tracker.reset_all()
        
        # 存储视频路径
        self.video_paths = {
            'camera1': video_path1,
            'camera2': video_path2
        }
        
        try:
            # 分阶段处理
            progress(0, desc="初始化处理")
            
            # 第一阶段：处理两个视频
            progress(0.1, desc="处理第一个视频...")
            output_path1 = self.process_video(video_path1, 'camera1')
            
            progress(0.5, desc="处理第二个视频...")
            output_path2 = self.process_video(video_path2, 'camera2')
            
            # 第二阶段：进行跨摄像头关联（使用简化版，实际应用中需要更复杂的逻辑）
            # 这里我们简单地基于特征进行匹配
            progress(0.8, desc="执行跨摄像头关联...")
            
            # 为了演示，我们重新处理视频以应用全局ID
            progress(0.9, desc="更新全局ID并重新处理...")
            
            # 重新处理两个视频，应用全局ID
            self.tracker.reset_all()
            final_output1 = self.process_video(video_path1, 'camera1')
            final_output2 = self.process_video(video_path2, 'camera2')
            
            # 清理临时文件
            if os.path.exists(output_path1) and output_path1 != final_output1:
                os.unlink(output_path1)
            if os.path.exists(output_path2) and output_path2 != final_output2:
                os.unlink(output_path2)
            
            progress(1.0, desc="处理完成")
            
            return final_output1, final_output2, "跨摄像头行人追踪处理完成！"
            
        except Exception as e:
            error_msg = f"处理时出错: {str(e)}"
            print(error_msg)
            return None, None, error_msg
    
    def stop_processing(self):
        """
        停止处理
        """
        self.stop_event.set()
        return "处理已停止"
    
    def clear_files(self, video1, video2):
        """
        清理文件
        """
        # 停止处理
        self.stop_processing()
        
        # 重置追踪器
        if self.tracker:
            self.tracker.reset_all()
        
        # 清理临时文件
        for path in self.video_paths.values():
            if path and os.path.exists(path) and path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(path)
                except:
                    pass
        
        self.video_paths = {}
        
        return None, None, "所有文件已清理"
    
    def create_interface(self):
        """
        创建Gradio界面
        """
        with gr.Blocks(title="跨摄像头行人重识别追踪系统") as interface:
            gr.Markdown("# 跨摄像头行人重识别追踪系统")
            gr.Markdown("上传两个不同摄像头的视频，系统将自动检测、追踪并关联跨摄像头的相同行人。")
            
            with gr.Row():
                with gr.Column():
                    video1_input = gr.Video(label="摄像头1视频")
                    video2_input = gr.Video(label="摄像头2视频")
                    
                    with gr.Row():
                        process_btn = gr.Button("开始处理", variant="primary")
                        stop_btn = gr.Button("停止处理")
                        clear_btn = gr.Button("清理")
                    
                    status_text = gr.Textbox(label="状态", interactive=False)
                
                with gr.Column():
                    video1_output = gr.Video(label="处理后视频1")
                    video2_output = gr.Video(label="处理后视频2")
            
            # 设置事件处理
            process_btn.click(
                fn=self.perform_cross_camera_association,
                inputs=[video1_input, video2_input],
                outputs=[video1_output, video2_output, status_text]
            )
            
            stop_btn.click(
                fn=self.stop_processing,
                outputs=status_text
            )
            
            clear_btn.click(
                fn=self.clear_files,
                inputs=[video1_input, video2_input],
                outputs=[video1_output, video2_output, status_text]
            )
            
            # 添加使用说明
            gr.Markdown("## 使用说明")
            gr.Markdown("1. 上传两个包含相同行人的不同摄像头视频")
            gr.Markdown("2. 点击'开始处理'按钮")
            gr.Markdown("3. 系统将自动检测行人、提取特征、进行追踪和跨摄像头关联")
            gr.Markdown("4. 相同的行人在两个视频中会被标记为相同的全局ID")
        
        return interface
    
    def launch(self, server_port=7860):
        """
        启动Gradio应用
        
        Args:
            server_port: 服务器端口
        """
        interface = self.create_interface()
        interface.launch(
            server_port=server_port,
            server_name="0.0.0.0",
            share=False
        )

def main():
    """
    主函数
    """
    print("启动跨摄像头行人重识别追踪系统...")
    
    # 创建应用实例
    app = MultiCameraReIDApp()
    
    # 启动应用
    try:
        app.launch(server_port=7860)
    except Exception as e:
        print(f"启动应用时出错: {e}")
    finally:
        # 清理资源
        app.stop_processing()
        print("应用已关闭")

if __name__ == "__main__":
    main()