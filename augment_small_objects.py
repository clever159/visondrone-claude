# augment_small_objects.py
import cv2
import numpy as np
import random
from pathlib import Path
import json
import shutil
from typing import List, Tuple
import albumentations as A

class SmallObjectAugmenter:
    def __init__(self, min_area_ratio=0.01, max_copy_objects=3):
        """
        小目标增强器
        Args:
            min_area_ratio: 定义小目标的最小面积比例
            max_copy_objects: Copy-Paste时最大复制目标数
        """
        self.min_area_ratio = min_area_ratio
        self.max_copy_objects = max_copy_objects
        
        # 定义数据增强管道
        self.transform = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=15, p=0.3),
                A.GaussianBlur(blur_limit=15, p=0.3),
                A.MedianBlur(blur_limit=15, p=0.2),
            ], p=0.4),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        ])

    def parse_yolo_label(self, label_path: str, img_shape: Tuple[int, int]) -> List[List[float]]:
        """解析YOLO格式标签文件"""
        if not Path(label_path).exists():
            return []
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # 转换为绝对坐标
                    img_h, img_w = img_shape
                    x_min = int((x_center - width/2) * img_w)
                    y_min = int((y_center - height/2) * img_h)
                    x_max = int((x_center + width/2) * img_w)
                    y_max = int((y_center + height/2) * img_h)
                    
                    boxes.append([x_min, y_min, x_max, y_max, cls_id])
        return boxes

    def save_yolo_label(self, boxes: List[List[float]], img_shape: Tuple[int, int], output_path: str):
        """保存YOLO格式标签"""
        img_h, img_w = img_shape
        with open(output_path, 'w') as f:
            for box in boxes:
                x_min, y_min, x_max, y_max, cls_id = box
                
                # 转换为YOLO格式
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                f.write(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def is_small_object(self, box: List[float], img_shape: Tuple[int, int]) -> bool:
        """判断是否为小目标"""
        x_min, y_min, x_max, y_max, _ = box
        area = (x_max - x_min) * (y_max - y_min)
        img_area = img_shape[0] * img_shape[1]
        return area / img_area < self.min_area_ratio

    def extract_object(self, img: np.ndarray, box: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """抠取目标对象"""
        x_min, y_min, x_max, y_max, _ = map(int, box[:4])
        
        # 确保坐标在图像范围内
        h, w = img.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return None, None
            
        # 抠取对象
        obj_img = img[y_min:y_max, x_min:x_max].copy()
        
        # 创建mask（这里简化，实际可用语义分割）
        mask = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8) * 255
        
        return obj_img, mask

    def paste_object(self, bg_img: np.ndarray, obj_img: np.ndarray, mask: np.ndarray, 
                    position: Tuple[int, int]) -> Tuple[np.ndarray, List[float]]:
        """粘贴对象到背景图像"""
        x, y = position
        obj_h, obj_w = obj_img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        
        # 确保粘贴位置合理
        if x + obj_w > bg_w or y + obj_h > bg_h or x < 0 or y < 0:
            return bg_img, None
            
        # 粘贴对象
        roi = bg_img[y:y+obj_h, x:x+obj_w]
        mask_inv = cv2.bitwise_not(mask)
        
        bg_masked = cv2.bitwise_and(roi, roi, mask=mask_inv)
        obj_masked = cv2.bitwise_and(obj_img, obj_img, mask=mask)
        
        result = cv2.add(bg_masked, obj_masked)
        bg_img[y:y+obj_h, x:x+obj_w] = result
        
        # 返回新的边界框
        new_box = [x, y, x + obj_w, y + obj_h]
        return bg_img, new_box

    def copy_paste_augment(self, src_img: np.ndarray, src_boxes: List[List[float]], 
                          bg_img: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        """Copy-Paste增强"""
        if not src_boxes:
            return bg_img, []
            
        result_img = bg_img.copy()
        new_boxes = []
        
        # 筛选小目标
        small_objects = [box for box in src_boxes if self.is_small_object(box, src_img.shape[:2])]
        
        if not small_objects:
            return result_img, new_boxes
            
        # 随机选择要复制的对象数量
        num_to_copy = min(len(small_objects), random.randint(1, self.max_copy_objects))
        selected_objects = random.sample(small_objects, num_to_copy)
        
        for box in selected_objects:
            # 抠取对象
            obj_img, mask = self.extract_object(src_img, box)
            if obj_img is None:
                continue
                
            # 随机选择粘贴位置（避免重叠）
            bg_h, bg_w = result_img.shape[:2]
            obj_h, obj_w = obj_img.shape[:2]
            
            max_attempts = 10
            for _ in range(max_attempts):
                x = random.randint(0, max(0, bg_w - obj_w))
                y = random.randint(0, max(0, bg_h - obj_h))
                
                # 检查是否与已有框重叠
                overlap = False
                for existing_box in new_boxes:
                    if self.boxes_overlap([x, y, x + obj_w, y + obj_h], existing_box[:4]):
                        overlap = True
                        break
                        
                if not overlap:
                    result_img, new_box = self.paste_object(result_img, obj_img, mask, (x, y))
                    if new_box:
                        new_boxes.append(new_box + [box[4]])  # 添加类别ID
                    break
                    
        return result_img, new_boxes

    def boxes_overlap(self, box1: List[float], box2: List[float], threshold: float = 0.1) -> bool:
        """检查两个框是否重叠"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算重叠面积
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = overlap_x * overlap_y
        
        # 计算较小框的面积
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        min_area = min(area1, area2)
        
        return overlap_area / max(min_area, 1e-6) > threshold

    def motion_blur(self, image: np.ndarray, degree: int = 10, angle: int = 45) -> np.ndarray:
        """运动模糊"""
        # 创建运动模糊核
        kernel = np.zeros((degree, degree))
        kernel[int((degree-1)/2), :] = np.ones(degree)
        kernel = kernel / degree
        
        # 旋转核
        M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (degree, degree))
        
        return cv2.filter2D(image, -1, kernel)

    def simulate_low_light(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """模拟低照度环境"""
        # Gamma校正模拟低光
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        
        return cv2.LUT(image, look_up_table)

    def augment_pipeline(self, img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        """增强管道"""
        # 应用albumentations增强
        if random.random() < 0.7:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        # 额外的运动模糊（针对低空视频特点）
        if random.random() < 0.3:
            degree = random.randint(5, 15)
            angle = random.randint(0, 360)
            img = self.motion_blur(img, degree, angle)
        
        # 模拟低照度
        if random.random() < 0.2:
            gamma = random.uniform(1.5, 3.0)
            img = self.simulate_low_light(img, gamma)
            
        return img, boxes

def main():
    """主函数：批量处理数据增强"""
    # 配置路径
    input_dir = Path("dataset/images/train")
    label_dir = Path("dataset/labels/train")
    output_img_dir = Path("dataset_aug/images/train")
    output_label_dir = Path("dataset_aug/labels/train")
    
    # 创建输出目录
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化增强器
    augmenter = SmallObjectAugmenter(min_area_ratio=0.01, max_copy_objects=2)
    
    # 获取所有图像文件
    img_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    print(f"发现 {len(img_files)} 个图像文件")
    
    # 复制原始数据
    for img_path in img_files:
        # 复制图像
        shutil.copy(img_path, output_img_dir / img_path.name)
        
        # 复制标签
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, output_label_dir / (img_path.stem + ".txt"))
    
    print("原始数据复制完成")
    
    # 生成增强数据
    augment_count = 0
    for img_path in img_files:
        try:
            # 读取图像和标签
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            label_path = label_dir / (img_path.stem + ".txt")
            boxes = augmenter.parse_yolo_label(str(label_path), img.shape[:2])
            
            # 生成2个增强版本
            for aug_idx in range(2):
                # 基础增强
                aug_img, aug_boxes = augmenter.augment_pipeline(img.copy(), boxes.copy())
                
                # Copy-Paste增强（50%概率）
                if random.random() < 0.5 and len(img_files) > 1:
                    # 随机选择背景图像
                    bg_img_path = random.choice([f for f in img_files if f != img_path])
                    bg_img = cv2.imread(str(bg_img_path))
                    
                    if bg_img is not None:
                        bg_label_path = label_dir / (bg_img_path.stem + ".txt")
                        bg_boxes = augmenter.parse_yolo_label(str(bg_label_path), bg_img.shape[:2])
                        
                        # 先对背景图像做基础增强
                        bg_img, bg_boxes = augmenter.augment_pipeline(bg_img, bg_boxes)
                        
                        # Copy-Paste
                        final_img, cp_boxes = augmenter.copy_paste_augment(img, boxes, bg_img)
                        final_boxes = bg_boxes + cp_boxes
                        
                        # 使用Copy-Paste结果
                        aug_img = final_img
                        aug_boxes = final_boxes
                
                # 保存增强后的图像和标签
                aug_name = f"{img_path.stem}_aug_{aug_idx}{img_path.suffix}"
                aug_img_path = output_img_dir / aug_name
                aug_label_path = output_label_dir / f"{img_path.stem}_aug_{aug_idx}.txt"
                
                cv2.imwrite(str(aug_img_path), aug_img)
                augmenter.save_yolo_label(aug_boxes, aug_img.shape[:2], str(aug_label_path))
                
                augment_count += 1
                
                if augment_count % 100 == 0:
                    print(f"已生成 {augment_count} 个增强样本")
                    
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
            continue
    
    print(f"数据增强完成！总共生成 {augment_count} 个增强样本")
    print(f"增强后数据保存在: {output_img_dir.parent}")

if __name__ == "__main__":
    main()
