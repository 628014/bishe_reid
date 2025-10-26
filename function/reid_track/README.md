# 跨摄像头行人追踪系统

本系统实现了基于YOLO检测、ByteTracker追踪和ReID模型的跨摄像头行人追踪功能，支持输入视频文件列表和描述性语言，实现对同一行人在不同摄像头间的连续追踪。

## 系统架构

- **目标检测**: 使用YOLO11n模型进行行人检测
- **目标追踪**: 使用ByteTracker进行单摄像头内的目标追踪
- **ReID模型**: 使用基于文本-图像融合的ReID模型进行跨摄像头身份关联
- **跨摄像头管理**: 实现全局ID分配和特征管理

## 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-repo/MLLM4Text-ReID-main.git
cd MLLM4Text-ReID-main/function/reid_track

# 安装依赖
pip install -r requirements.txt
```

## 配置文件说明

系统使用`config.yaml`作为配置文件，主要包含以下部分：

### 1. 检测器配置 (detector)
- `model_path`: YOLO模型文件路径
- `model_type`: 模型类型（用于自动下载）
- `confidence_threshold`: 置信度阈值
- `iou_threshold`: IOU阈值
- `classes`: 检测类别（默认只检测人）

### 2. 跟踪器配置 (tracker)
- `type`: 跟踪器类型
- `track_buffer`: 跟踪缓存大小
- `match_thresh`: 匹配阈值

### 3. ReID模型配置 (reid)
- `config_file`: ReID模型配置文件路径
- `checkpoint_path`: ReID模型权重文件路径
- `img_size`: 输入图像尺寸
- `device`: 运行设备

### 4. 跨摄像头配置 (cross_camera)
- `similarity_threshold`: 相似度阈值
- `use_text_filter`: 是否使用文本过滤

### 5. 输出配置 (output)
- `save_video`: 是否保存视频
- `show_video`: 是否显示视频
- `save_results`: 是否保存结果
- `output_dir`: 输出目录

## 使用方法

### 1. 命令行使用

```bash
# 单摄像头跟踪
python main.py --config config.yaml --videos /path/to/video.mp4 --cameras camera_1

# 多摄像头跟踪
python main.py --config config.yaml --videos /path/to/video1.mp4 /path/to/video2.mp4 --cameras camera_1 camera_2

# 带文本描述的跟踪
python main.py --config config.yaml --videos /path/to/video.mp4 --cameras camera_1 --texts "穿红色外套的人"
```

### 2. 交互式演示

```bash
python demo.py
```

演示脚本提供三种模式：
- 单摄像头跟踪
- 多摄像头跟踪
- 预定义配置演示

### 3. 作为库使用

```python
from main import CrossCameraReIDTracker

# 初始化系统
tracker = CrossCameraReIDTracker('config.yaml')

# 处理单个视频
results = tracker.track_video(
    video_path='/path/to/video.mp4',
    camera_id='camera_1',
    text_description='穿蓝色牛仔裤的人'
)

# 处理多个视频
results = tracker.process_multiple_videos(
    video_list=['/path/to/video1.mp4', '/path/to/video2.mp4'],
    camera_ids=['camera_1', 'camera_2'],
    text_descriptions=['穿红色外套的人', '戴眼镜的人']
)
```

## 输出结果

### 1. 跟踪视频
如果`save_video=True`，系统会在输出目录中保存带有跟踪结果的视频。

### 2. 跟踪结果文件
如果`save_results=True`，系统会保存跟踪结果到文本文件，格式为：
```
frame_id,global_id,x1,y1,x2,y2,confidence
```

### 3. 统计信息
系统会输出跟踪统计信息，包括：
- 总全局ID数量
- 每个摄像头的轨迹数量
- 在多个摄像头中出现的全局ID

## 注意事项

1. 确保ReID模型配置文件和权重文件路径正确
2. 文本描述应尽可能详细，有助于提高跨摄像头匹配精度
3. 对于大规模场景，可能需要调整相似度阈值以获得最佳效果
4. GPU加速可以显著提高处理速度

## 后续开发

1. 支持实时摄像头输入
2. 开发Web界面进行结果可视化
3. 优化跨摄像头匹配算法
4. 支持更复杂的场景分析