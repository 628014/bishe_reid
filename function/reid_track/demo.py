import os
import sys
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CrossCameraReIDTracker

def demo_single_camera():
    """单摄像头演示"""
    # 初始化系统
    tracker = CrossCameraReIDTracker('config.yaml')
    
    # 定义视频路径和摄像头ID
    video_path = input("请输入视频文件路径: ")
    camera_id = "camera_1"
    
    # 可以添加文本描述
    text_desc = input("请输入目标描述（可选，直接回车跳过）: ")
    
    if not text_desc:
        text_desc = None
    
    # 跟踪视频
    print(f"开始处理视频: {video_path}")
    results = tracker.track_video(video_path, camera_id, text_desc)
    
    print(f"处理完成，共处理了 {len(results)} 帧")

def demo_multi_camera():
    """多摄像头演示"""
    # 初始化系统
    tracker = CrossCameraReIDTracker('config.yaml')
    
    # 获取视频数量
    num_videos = int(input("请输入摄像头数量: "))
    
    video_list = []
    camera_ids = []
    text_descriptions = []
    
    # 收集视频信息
    for i in range(num_videos):
        video_path = input(f"请输入摄像头 {i+1} 的视频文件路径: ")
        camera_id = f"camera_{i+1}"
        
        # 可以添加文本描述
        text_desc = input(f"请输入摄像头 {i+1} 目标描述（可选，直接回车跳过）: ")
        
        video_list.append(video_path)
        camera_ids.append(camera_id)
        text_descriptions.append(text_desc if text_desc else None)
    
    # 处理多个视频
    print("开始处理多摄像头视频...")
    results = tracker.process_multiple_videos(video_list, camera_ids, text_descriptions)
    
    print("多摄像头处理完成")

def demo_predefined():
    """预定义配置演示"""
    # 示例配置
    configs = [
        {
            'name': '示例1: 单摄像头跟踪',
            'videos': ['/path/to/video1.mp4'],
            'cameras': ['camera_1'],
            'texts': None
        },
        {
            'name': '示例2: 双摄像头跟踪',
            'videos': ['/path/to/video1.mp4', '/path/to/video2.mp4'],
            'cameras': ['camera_1', 'camera_2'],
            'texts': None
        }
    ]
    
    # 显示示例
    print("预定义示例:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
    
    # 选择示例
    choice = int(input("请选择示例 (1-2): ")) - 1
    
    if 0 <= choice < len(configs):
        config = configs[choice]
        
        # 允许用户修改路径
        print(f"\n当前配置: {config['name']}")
        for i in range(len(config['videos'])):
            current_path = config['videos'][i]
            new_path = input(f"请输入视频 {i+1} 的路径 [默认: {current_path}]: ")
            if new_path:
                config['videos'][i] = new_path
        
        # 初始化系统并运行
        tracker = CrossCameraReIDTracker('config.yaml')
        tracker.process_multiple_videos(
            config['videos'],
            config['cameras'],
            config['texts']
        )
    else:
        print("无效的选择")

def main():
    """演示主函数"""
    print("=== 跨摄像头行人追踪系统演示 ===")
    print("请选择演示模式:")
    print("1. 单摄像头跟踪")
    print("2. 多摄像头跟踪")
    print("3. 预定义配置演示")
    
    choice = input("请选择 (1-3): ")
    
    if choice == '1':
        demo_single_camera()
    elif choice == '2':
        demo_multi_camera()
    elif choice == '3':
        demo_predefined()
    else:
        print("无效的选择")

if __name__ == "__main__":
    main()