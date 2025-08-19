# tile_inference.py
import torch
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import time

class TileInference:
    def __init__(self, model_path: str, tile_size: int = 1024, overlap_ratio: float = 0.25, 
                 conf_threshold: float = 0.2, iou_threshold: float = 0.5):
        """
        Tile推理器
        Args:
            model_path: YOLO模型路径
            tile_size: 切片大小
            overlap_ratio: 重叠比例
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 计算重叠像素数
        self.overlap_pixels = int(tile_size * overlap_ratio)
        self.step_size = tile_size - self.overlap_pixels
        
        print(f"TileInference初始化完成:")
        print(f"  - 切片大小: {tile_size}x{tile_size}")
        print(f"  - 重叠比例: {overlap_ratio} ({self.overlap_pixels}像素)")
        print(f"  - 步长: {self.step_size}")
        print(f"  - 置信度阈值: {conf_threshold}")
        print(f"  - IoU阈值: {iou_threshold}")

    def tile_image(self, img: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        将图像切片
        Returns:
            sub_images: 子图像列表
            offsets: 每个子图在原图中的偏移坐标 [(x_offset, y_offset), ...]
        """
        h, w = img.shape[:2]
        sub_images = []
        offsets = []
        
        # 计算切片数量
        n_rows = (h - self.overlap_pixels) // self.step_size + (1 if (h - self.overlap_pixels) % self.step_size > 0 else 0)
        n_cols = (w - self.overlap_pixels) // self.step_size + (1 if (w - self.overlap_pixels) % self.step_size > 0 else 0)
        
        for row in range(n_rows):
            for col in range(n_cols):
                # 计算切片坐标
                y_start = row * self.step_size
                x_start = col * self.step_size
                
                # 确保不超出图像边界
                y_end = min(y_start + self.tile_size, h)
                x_end = min(x_start + self.tile_size, w)
                
                # 如果切片太小，调整起始位置
                if y_end - y_start < self.tile_size:
                    y_start = max(0, y_end - self.tile_size)
                if x_end - x_start < self.tile_size:
                    x_start = max(0, x_end - self.tile_size)
                
                # 提取子图像
                sub_img = img[y_start:y_end, x_start:x_end]
                
                # 如果子图像小于期望大小，进行padding
                if sub_img.shape[0] < self.tile_size or sub_img.shape[1] < self.tile_size:
                    padded_img = np.zeros((self.tile_size, self.tile_size, img.shape[2]), dtype=img.dtype)
                    padded_img[:sub_img.shape[0], :sub_img.shape[1]] = sub_img
                    sub_img = padded_img
                
                sub_images.append(sub_img)
                offsets.append((x_start, y_start))
        
        print(f"图像 {w}x{h} 切分为 {len(sub_images)} 个子图 ({n_rows}x{n_cols})")
        return sub_images, offsets

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        非极大值抑制
        Args:
            boxes: 边界框数组 [[x1, y1, x2, y2], ...]
            scores: 置信度分数
            iou_threshold: IoU阈值
        Returns:
            保留的框索引列表
        """
        if len(boxes) == 0:
            return []
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按分数排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # 保留IoU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def merge_results(self, all_results: List[Tuple], img_shape: Tuple[int, int]) -> Dict:
        """
        合并所有子图的检测结果
        Args:
            all_results: [(results, offset), ...] 所有子图的检测结果
            img_shape: 原图尺寸 (height, width)
        Returns:
            合并后的检测结果
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        h, w = img_shape
        
        for results, (x_offset, y_offset) in all_results:
            if results[0].boxes is None:
                continue
                
            # 获取检测结果
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # 过滤低置信度检测
            valid_mask = scores >= self.conf_threshold
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]
            
            if len(boxes) == 0:
                continue
            
            # 将坐标映射回原图
            boxes[:, [0, 2]] += x_offset  # x坐标
            boxes[:, [1, 3]] += y_offset  # y坐标
            
            # 确保边界框在图像范围内
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            
            # 过滤无效框（面积太小的）
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid_area_mask = areas > 100  # 最小面积阈值
            
            boxes = boxes[valid_area_mask]
            scores = scores[valid_area_mask]
            classes = classes[valid_area_mask]
            
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend(classes)
        
        if not all_boxes:
            return {"boxes": np.array([]), "scores": np.array([]), "classes": np.array([])}
        
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)
        
        print(f"合并前检测框数量: {len(all_boxes)}")
        
        # 按类别分别进行NMS
        final_boxes = []
        final_scores = []
        final_classes = []
        
        unique_classes = np.unique(all_classes)
        for cls_id in unique_classes:
            cls_mask = all_classes == cls_id
            cls_boxes = all_boxes[cls_mask]
            cls_scores = all_scores[cls_mask]
            
            # 对该类别进行NMS
            keep_indices = self.nms(cls_boxes, cls_scores, self.iou_threshold)
            
            if keep_indices:
                final_boxes.extend(cls_boxes[keep_indices])
                final_scores.extend(cls_scores[keep_indices])
                final_classes.extend([cls_id] * len(keep_indices))
        
        print(f"NMS后检测框数量: {len(final_boxes)}")
        
        return {
            "boxes": np.array(final_boxes) if final_boxes else np.array([]),
            "scores": np.array(final_scores) if final_scores else np.array([]),
            "classes": np.array(final_classes) if final_classes else np.array([])
        }

    def predict_single_image(self, img_path: str, visualize: bool = True, 
                           save_path: str = None) -> Dict:
        """
        对单张图像进行预测
        Args:
            img_path: 图像路径
            visualize: 是否可视化结果
            save_path: 保存路径
        Returns:
            检测结果
        """
        print(f"\n开始处理图像: {img_path}")
        start_time = time.time()
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return {}
        
        original_shape = img.shape[:2]
        print(f"原图尺寸: {original_shape[1]}x{original_shape[0]}")
        
        # 切片
        sub_images, offsets = self.tile_image(img)
        
        # 对每个子图进行推理
        all_results = []
        print(f"开始推理 {len(sub_images)} 个子图...")
        
        for i, (sub_img, offset) in enumerate(zip(sub_images, offsets)):
            if i % 10 == 0:
                print(f"  处理进度: {i+1}/{len(sub_images)}")
            
            results = self.model.predict(
                sub_img, 
                conf=self.conf_threshold,
                verbose=False,
                save=False
            )
            all_results.append((results, offset))
        
        # 合并结果
        final_results = self.merge_results(all_results, original_shape)
        
        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}秒")
        print(f"最终检测到 {len(final_results['boxes'])} 个目标")
        
        # 可视化结果
        if visualize and len(final_results['boxes']) > 0:
            result_img = self.visualize_results(img, final_results)
            
            if save_path:
                cv2.imwrite(save_path, result_img)
                print(f"结果已保存到: {save_path}")
            else:
                # 显示结果（如果在支持GUI的环境中）
                try:
                    cv2.imshow('Detection Results', result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except:
                    print("无法显示图像，请确保在支持GUI的环境中运行")
        
        return final_results

    def visualize_results(self, img: np.ndarray, results: Dict, 
                         class_names: List[str] = None) -> np.ndarray:
        """
        可视化检测结果
        Args:
            img: 原图像
            results: 检测结果
            class_names: 类别名称列表
        Returns:
            标注后的图像
        """
        result_img = img.copy()
        
        if len(results['boxes']) == 0:
            return result_img
        
        # 默认类别名称
        if class_names is None:
            class_names = ['person', 'car', 'truck', 'bus', 'motorbike', 'bicycle']
        
        # 定义颜色
        colors = [
            (0, 255, 0),    # 绿色 - person
            (255, 0, 0),    # 蓝色 - car
            (0, 0, 255),    # 红色 - truck
            (255, 255, 0),  # 青色 - bus
            (255, 0, 255),  # 品红 - motorbike
            (0, 255, 255),  # 黄色 - bicycle
        ]
        
        boxes = results['boxes']
        scores = results['scores']
        classes = results['classes'].astype(int)
        
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            # 选择颜色
            color = colors[cls_id % len(colors)]
            
            # 画边界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            class_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
            label = f'{class_name}: {score:.2f}'
            
            # 计算文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 画标签背景
            cv2.rectangle(result_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # 画标签文本
            cv2.putText(result_img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return result_img

    def predict_batch(self, input_dir: str, output_dir: str = None, 
                     img_extensions: List[str] = ['.jpg', '.png', '.jpeg']):
        """
        批量处理图像
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            img_extensions: 支持的图像扩展名
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"输入目录不存在: {input_dir}")
            return
        
        # 创建输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        img_files = []
        for ext in img_extensions:
            img_files.extend(input_path.glob(f'*{ext}'))
            img_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"发现 {len(img_files)} 个图像文件")
        
        # 处理每个图像
        total_time = 0
        for i, img_file in enumerate(img_files):
            print(f"\n{'='*50}")
            print(f"处理进度: {i+1}/{len(img_files)}")
            
            start_time = time.time()
            
            # 设置输出路径
            save_path = None
            if output_dir:
                save_path = str(output_path / f"{img_file.stem}_result{img_file.suffix}")
            
            # 进行预测
            results = self.predict_single_image(
                str(img_file), 
                visualize=True, 
                save_path=save_path
            )
            
            process_time = time.time() - start_time
            total_time += process_time
            
            print(f"处理完成，耗时: {process_time:.2f}秒")
        
        avg_time = total_time / len(img_files) if img_files else 0
        print(f"\n{'='*50}")
        print(f"批量处理完成！")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均每张: {avg_time:.2f}秒")
        print(f"处理速度: {1/avg_time:.2f} FPS" if avg_time > 0 else "N/A")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Tile Inference')
    parser.add_argument('--model', required=True, help='YOLO模型路径')
    parser.add_argument('--input', required=True, help='输入图像路径或目录')
    parser.add_argument('--output', help='输出目录（可选）')
    parser.add_argument('--tile-size', type=int, default=1024, help='切片大小')
    parser.add_argument('--overlap', type=float, default=0.25, help='重叠比例')
    parser.add_argument('--conf', type=float, default=0.2, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU阈值')
    
    args = parser.parse_args()
    
    # 初始化推理器
    tile_inference = TileInference(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像处理
        save_path = None
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                save_path = str(output_path / f"{input_path.stem}_result{input_path.suffix}")
            else:
                save_path = str(output_path)
        
        results = tile_inference.predict_single_image(
            str(input_path), 
            visualize=True, 
            save_path=save_path
        )
        
    elif input_path.is_dir():
        # 批量处理
        tile_inference.predict_batch(
            input_dir=str(input_path),
            output_dir=args.output
        )
    else:
        print(f"输入路径不存在: {args.input}")

if __name__ == "__main__":
    # 示例用法
    if len(import sys) > 1 and sys.argv[1:]:
        main()
    else:
        # 示例代码
        print("示例用法:")
        print("1. 单张图像:")
        print("   python tile_inference.py --model best.pt --input test.jpg --output result.jpg")
        print("2. 批量处理:")
        print("   python tile_inference.py --model best.pt --input test_images/ --output results/")
        print("3. 自定义参数:")
        print("   python tile_inference.py --model best.pt --input test.jpg --tile-size 512 --overlap 0.3 --conf 0.3")
        
        # 如果没有命令行参数，运行测试代码
        model_path = "runs/detect/train/weights/best.pt"
        if Path(model_path).exists():
            tile_inference = TileInference(model_path)
            
            # 测试单张图像
            test_img = "test.jpg"
            if Path(test_img).exists():
                results = tile_inference.predict_single_image(test_img, save_path="result.jpg")
            else:
                print(f"测试图像不存在: {test_img}")
        else:
            print(f"模型文件不存在: {model_path}")
            print("请先训练模型或提供正确的模型路径")