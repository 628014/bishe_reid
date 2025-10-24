import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    """
    可视化工具，用于展示检测、追踪和匹配结果
    """
    def __init__(self):
        """
        初始化可视化工具
        """
        # 生成多种颜色用于不同的追踪ID
        self.colors = self._generate_colors(100)  # 预生成100种颜色
        print("可视化工具初始化完成")
    
    def _generate_colors(self, num_colors):
        """
        生成多种颜色
        
        Args:
            num_colors: 需要的颜色数量
            
        Returns:
            颜色列表，每个颜色是(B, G, R)格式
        """
        colors = []
        # 使用HSV色彩空间生成不同颜色
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            saturation = 255
            value = 255
            # OpenCV使用HSV格式为(H, S, V)，范围分别是[0, 180), [0, 255), [0, 255)
            color = cv2.cvtColor(np.array([[[hue, saturation, value]]], dtype=np.uint8), 
                               cv2.COLOR_HSV2BGR)[0, 0].tolist()
            colors.append(tuple(map(int, color)))
        return colors
    
    def get_color_by_id(self, track_id):
        """
        根据追踪ID获取颜色
        
        Args:
            track_id: 追踪目标ID
            
        Returns:
            BGR格式的颜色元组
        """
        return self.colors[track_id % len(self.colors)]
    
    def visualize_detections(self, image, detections, show_confidence=True):
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表，每个元素是包含'bbox'和'confidence'的字典
            show_confidence: 是否显示置信度
            
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det.get('confidence', 0)
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            if show_confidence:
                label = f"Person {i+1}: {confidence:.2f}"
            else:
                label = f"Person {i+1}"
            
            # 确保标签不超出图像范围
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1_label = max(0, y1 - label_size[1] - 5)
            
            # 绘制标签背景
            cv2.rectangle(vis_image, 
                         (x1, y1_label), 
                         (x1 + label_size[0], y1_label + label_size[1] + baseline), 
                         (0, 255, 0), -1)
            
            # 绘制标签文本
            cv2.putText(vis_image, label, (x1, y1_label + label_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_image
    
    def visualize_tracking(self, image, tracks, show_track_id=True):
        """
        可视化追踪结果
        
        Args:
            image: 原始图像
            tracks: 追踪结果列表，每个元素是包含'id'和'bbox'的字典
            show_track_id: 是否显示追踪ID
            
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            
            # 获取颜色
            color = self.get_color_by_id(track_id)
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if show_track_id:
                label = f"ID: {track_id}"
                
                # 确保标签不超出图像范围
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y1_label = max(0, y1 - label_size[1] - 5)
                
                # 绘制标签背景
                cv2.rectangle(vis_image, 
                             (x1, y1_label), 
                             (x1 + label_size[0], y1_label + label_size[1] + baseline), 
                             color, -1)
                
                # 绘制标签文本
                cv2.putText(vis_image, label, (x1, y1_label + label_size[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image
    
    def visualize_trajectories(self, image, track_history, current_frame, max_history=30):
        """
        可视化轨迹历史
        
        Args:
            image: 原始图像
            track_history: 轨迹历史字典 {track_id: [{'frame': frame_num, 'bbox': bbox}]}
            current_frame: 当前帧号
            max_history: 显示的最大历史帧数
            
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        for track_id, history in track_history.items():
            color = self.get_color_by_id(track_id)
            
            # 筛选最近的历史记录
            recent_history = [h for h in history 
                             if current_frame - h['frame'] <= max_history]
            
            # 绘制轨迹点
            for i in range(1, len(recent_history)):
                prev_bbox = recent_history[i-1]['bbox']
                curr_bbox = recent_history[i]['bbox']
                
                # 计算中心点
                prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, 
                              (prev_bbox[1] + prev_bbox[3]) // 2)
                curr_center = ((curr_bbox[0] + curr_bbox[2]) // 2, 
                              (curr_bbox[1] + curr_bbox[3]) // 2)
                
                # 绘制轨迹线
                cv2.line(vis_image, prev_center, curr_center, color, 2)
        
        return vis_image
    
    def visualize_matches(self, query_image, matched_images, similarities, top_k=5):
        """
        可视化匹配结果
        
        Args:
            query_image: 查询图像
            matched_images: 匹配的图像列表
            similarities: 相似度列表
            top_k: 显示前k个匹配结果
            
        Returns:
            组合后的可视化图像
        """
        # 确保匹配数量不超过可用图像数量
        top_k = min(top_k, len(matched_images))
        
        # 创建大图像以容纳查询和匹配结果
        query_h, query_w = query_image.shape[:2]
        max_h = query_h
        total_w = query_w
        
        for i in range(top_k):
            match_h, match_w = matched_images[i].shape[:2]
            max_h = max(max_h, match_h)
            total_w += match_w + 10  # 添加10像素间隔
        
        # 创建白色背景的大图像
        result = np.ones((max_h + 40, total_w, 3), dtype=np.uint8) * 255
        
        # 放置查询图像
        query_y = 20
        query_x = 10
        result[query_y:query_y+query_h, query_x:query_x+query_w] = query_image
        
        # 添加查询标签
        cv2.putText(result, "Query", (query_x, query_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 放置匹配结果
        current_x = query_x + query_w + 20
        
        for i in range(top_k):
            match_img = matched_images[i]
            match_h, match_w = match_img.shape[:2]
            match_y = 20
            
            # 居中放置
            if match_h < max_h:
                match_y += (max_h - match_h) // 2
            
            result[match_y:match_y+match_h, current_x:current_x+match_w] = match_img
            
            # 添加匹配标签和相似度
            label = f"Match {i+1}: {similarities[i]:.3f}"
            cv2.putText(result, label, (current_x, match_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_x += match_w + 20
        
        return result
    
    def save_visualization(self, image, output_path):
        """
        保存可视化结果
        
        Args:
            image: 要保存的图像
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, image)
        print(f"可视化结果已保存到: {output_path}")
    
    def create_video_from_frames(self, frames_folder, output_video_path, fps=30):
        """
        从帧图像创建视频
        
        Args:
            frames_folder: 包含帧图像的文件夹
            output_video_path: 输出视频路径
            fps: 视频帧率
        """
        # 获取所有图像文件并排序
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = sorted([f for f in os.listdir(frames_folder) 
                            if f.lower().endswith(image_extensions)])
        
        if not image_files:
            raise ValueError(f"文件夹中没有找到图像文件: {frames_folder}")
        
        # 获取第一帧的尺寸
        first_frame = cv2.imread(os.path.join(frames_folder, image_files[0]))
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 写入所有帧
        for image_file in tqdm(image_files, desc="创建视频"):
            frame_path = os.path.join(frames_folder, image_file)
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        print(f"视频已保存到: {output_video_path}")
    
    def visualize_similarity_matrix(self, similarity_matrix, query_labels=None, gallery_labels=None, title="相似度矩阵"):
        """
        可视化相似度矩阵
        
        Args:
            similarity_matrix: 相似度矩阵
            query_labels: 查询标签列表
            gallery_labels: 图库标签列表
            title: 图像标题
            
        Returns:
            Matplotlib图像对象
        """
        plt.figure(figsize=(10, 8))
        
        # 确保是numpy数组
        if isinstance(similarity_matrix, torch.Tensor):
            similarity_matrix = similarity_matrix.cpu().numpy()
        
        # 创建热力图
        plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='相似度')
        plt.title(title)
        
        # 设置坐标轴
        plt.xlabel('图库索引')
        plt.ylabel('查询索引')
        
        # 如果提供了标签，使用标签替换索引
        if query_labels is not None:
            plt.yticks(range(len(query_labels)), query_labels, rotation=45)
        if gallery_labels is not None:
            plt.xticks(range(len(gallery_labels)), gallery_labels, rotation=45)
        
        plt.tight_layout()
        return plt.gcf()

# 示例用法
if __name__ == "__main__":
    visualizer = Visualizer()
    # 在实际使用中，这里应该传入实际的图像和检测/追踪结果
    # image = cv2.imread("sample.jpg")
    # detections = [...]  # 检测结果
    # vis_image = visualizer.visualize_detections(image, detections)
    # visualizer.save_visualization(vis_image, "output/detections.jpg")