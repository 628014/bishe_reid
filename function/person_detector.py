import torch
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

class PersonDetector:
    """
    行人检测器，使用YOLO模型检测图像中的行人
    """
    def __init__(self, model_type='yolov5s', device=None):
        """
        初始化行人检测器
        
        Args:
            model_type: YOLO模型类型，如'yolov5s', 'yolov5m', 'yolov5l'
            device: 运行设备，如'cuda'或'cpu'，默认为None（自动选择）
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"正在加载{model_type}模型，使用设备: {self.device}")
        # 使用Ultralytics的YOLOv8模型
        self.model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成")
    
    def detect_persons(self, image, conf_threshold=0.5, iou_threshold=0.45):
        """
        在图像中检测行人，返回适合ByteTrack格式的检测结果
        
        Args:
            image: OpenCV图像(BGR)或图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            检测结果，包含两种格式：
            - detections: 原始格式的检测结果列表
            - byte_track_dets: ByteTrack格式的检测结果 (x1, y1, x2, y2, confidence)
        """
        # 如果输入是路径，读取图像
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
        
        # 设置置信度和IoU阈值
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        
        # 检测
        results = self.model(image)
        
        # 处理检测结果
        detections = []
        byte_track_dets = []
        
        for det in results.xyxy[0]:  # detections per image
            if int(det[5]) == 0:  # 类别ID为0表示人
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = float(det[4])
                
                # 确保边界框在图像范围内
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 裁剪行人图像
                person_img = image[y1:y2, x1:x2].copy()
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'person_img': person_img
                })
                
                # 添加ByteTrack格式的检测结果
                byte_track_dets.append([x1, y1, x2, y2, confidence])
        
        return {
            'detections': detections,
            'byte_track_dets': np.array(byte_track_dets) if byte_track_dets else np.empty((0, 5)),
            'image': image
        }
    
    def detect_persons_in_folder(self, images_folder, output_folder=None, conf_threshold=0.5):
        """
        检测文件夹中所有图像的行人
        
        Args:
            images_folder: 包含图像的文件夹
            output_folder: 可视化结果的输出文件夹，为None时不保存
            conf_threshold: 置信度阈值
            
        Returns:
            每个图像的检测结果字典
        """
        # 获取所有图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(image_extensions)]
        
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        results = {}
        
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            print(f"处理图像: {image_file}")
            detection_result = self.detect_persons(image_path, conf_threshold)
            results[image_file] = detection_result
            
            # 可视化并保存结果
            if output_folder:
                vis_image = self.visualize_detections(detection_result['image'].copy(), detection_result['detections'])
                output_path = os.path.join(output_folder, f"detected_{image_file}")
                cv2.imwrite(output_path, vis_image)
        
        return results
    
    def visualize_detections(self, image, detections):
        """
        可视化检测结果
        
        Args:
            image: OpenCV图像(BGR)
            detections: 检测结果列表
            
        Returns:
            可视化后的图像
        """
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Person {i+1}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def save_detections(self, image, detections, output_path):
        """
        保存检测结果可视化图像
        
        Args:
            image: OpenCV图像(BGR)
            detections: 检测结果列表
            output_path: 输出文件路径
        """
        vis_image = self.visualize_detections(image, detections)
        # 确保输出目录存在
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis_image)

# 示例用法
if __name__ == "__main__":
    detector = PersonDetector(model_type='yolov5s')
    # 可以在这里添加测试代码
    print("行人检测器已初始化")