import os
import sys
import tempfile
import shutil
import time
import pandas as pd
from typing import List, Dict, Optional

# 导入Gradio并检查版本
# import gradio
# print(f"Gradio版本: {gradio.__version__}")
import gradio as gr

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主程序
from main import CrossCameraReIDTracker

def process_videos(video_files, text_descriptions, config_path="config.yaml"):
    """处理上传的视频文件和文本描述
    Args:
        video_files: 上传的视频文件列表
        text_descriptions: 文本描述
        config_path: 配置文件路径
    Returns:
        处理结果和日志
    """
    if not video_files:
        return "请上传视频文件", None, None
    
    # 创建临时目录存储视频文件
    temp_dir = tempfile.mkdtemp()
    try:
        # 准备文本描述
        text = text_descriptions.strip() if text_descriptions else ""
        
        # 初始化跟踪器
        tracker = CrossCameraReIDTracker(config_path)
        
        # 记录开始时间
        start_time = time.time()
        
        # 处理每个视频文件
        for i, video_file in enumerate(video_files):
            # 保存上传的视频文件
            camera_id = f"camera_{i+1}"
            video_path = os.path.join(temp_dir, f"{camera_id}_{os.path.basename(video_file.name)}")
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            
            # 处理视频
            tracker.track_video(
                video_path=video_path,
                camera_id=camera_id,
                text_description=text
            )
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 获取结果文件（使用第一个摄像头的结果）
        output_dir = tracker.config['output']['save_dir']
        result_file = os.path.join(output_dir, "camera_1_results.txt")
        
        if not os.path.exists(result_file) and video_files:
            # 如果没有生成结果文件，创建一个简单的结果文件
            with open(result_file, "w") as f:
                f.write(f"# 视频处理结果: {len(video_files)}个文件\n")
                f.write(f"处理时间: {process_time:.2f}秒\n")
        
        # 生成统计报告
        stats = tracker.cross_camera_tracker.get_track_statistics()
        report = generate_report(stats, process_time, len(video_files))
        
        # 生成可视化结果
        visualization_html = generate_visualization(stats)
        
        return report, result_file, visualization_html
        
    except Exception as e:
        return f"处理失败: {str(e)}", None, None
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

def generate_report(stats: Dict, process_time: float, num_videos: int):
    """生成处理报告
    Args:
        stats: 统计信息
        process_time: 处理时间
        num_videos: 视频数量
    Returns:
        格式化的报告字符串
    """
    report = f"# 处理报告\n\n"
    report += f"**处理视频数:** {num_videos}\n"
    report += f"**总处理时间:** {process_time:.2f} 秒\n\n"
    
    report += "## 跟踪统计\n\n"
    report += f"- **总全局ID数量:** {stats['total_global_ids']}\n"
    
    report += "- **摄像头轨迹数:**\n"
    for camera_id, count in stats['cameras'].items():
        report += f"  - {camera_id}: {count} 个轨迹\n"
    
    report += "\n- **跨摄像头出现的目标:**\n"
    cross_camera_ids = [
        (global_id, cameras) 
        for global_id, cameras in stats['global_id_cameras'].items() 
        if len(cameras) > 1
    ]
    
    if cross_camera_ids:
        for global_id, cameras in cross_camera_ids:
            report += f"  - ID {global_id}: 出现在摄像头 {', '.join(cameras)}\n"
    else:
        report += "  无跨摄像头匹配目标\n"
    
    return report

def generate_visualization(stats: Dict):
    """生成可视化HTML
    Args:
        stats: 统计信息
    Returns:
        HTML字符串
    """
    # 简单的可视化HTML
    html = "<div style='padding: 20px;'>"
    
    # 统计图表
    html += "<h3>跟踪统计图表</h3>"
    
    # 摄像头轨迹数量条形图
    html += "<div style='margin-bottom: 30px;'>"
    html += "<h4>各摄像头轨迹数量</h4>"
    html += "<div style='display: flex; flex-direction: column;'>"
    
    max_count = max(stats['cameras'].values(), default=1)
    for camera_id, count in stats['cameras'].items():
        width = (count / max_count) * 100
        html += f"<div style='margin-bottom: 10px;'>"
        html += f"<span style='display: inline-block; width: 100px;'>{camera_id}:</span>"
        html += f"<div style='display: inline-block; width: 400px; height: 30px; background-color: #f0f0f0;'>"
        html += f"<div style='height: 100%; width: {width}%; background-color: #4CAF50;'></div>"
        html += f"</div>"
        html += f"<span style='margin-left: 10px;'>{count}</span>"
        html += f"</div>"
    html += "</div></div>"
    
    # 跨摄像头ID统计
    html += "<div style='margin-bottom: 30px;'>"
    html += "<h4>跨摄像头ID出现次数</h4>"
    html += "<ul>"
    
    for global_id, cameras in stats['global_id_cameras'].items():
        if len(cameras) > 1:
            html += f"<li>ID {global_id}: 出现在 {len(cameras)} 个摄像头 ({', '.join(cameras)})</li>"
    
    html += "</ul></div>"
    
    html += "</div>"
    return html

def create_interface():
    """创建Gradio界面"""
    # 创建输入组件 - 使用用户指定的gr.inputs.File方式
    video_input = gr.inputs.File(file_count="multiple", label="上传视频文件")
    text_input = gr.Textbox(
        label="文本描述（可选）",
        placeholder="输入视频中行人的描述。例如：穿红色外套的人",
        lines=5
    )
    config_input = gr.Textbox(
        label="配置文件路径",
        value="config.yaml",
        placeholder="默认使用当前目录下的config.yaml"
    )
    
    # 创建输出组件
    report_output = "markdown"
    result_files_output = "file"
    visualization_output = "html"
    
    # 创建界面
    interface = gr.Interface(
        fn=process_videos,
        inputs=[video_input, text_input, config_input],
        outputs=[report_output, result_files_output, visualization_output],
        title="跨摄像头行人追踪系统",
        description="上传视频文件和文本描述，系统将进行行人检测、跟踪和身份关联。",
        allow_flagging="never"
    )
    
    return interface

def main():
    """主函数"""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()