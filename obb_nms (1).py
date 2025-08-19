# obb_nms.py
import torch
import numpy as np
import cv2
from typing import List, Tuple, Union
import math

def rbox_to_poly(rboxes):
    """
    将旋转框转换为多边形顶点
    Args:
        rboxes: 旋转框 [N, 5] (cx, cy, w, h, angle) 或 [cx, cy, w, h, angle]
    Returns:
        polys: 多边形顶点 [N, 4, 2] 或 [4, 2]
    """
    if isinstance(rboxes, torch.Tensor):
        device = rboxes.device
        rboxes_np = rboxes.detach().cpu().numpy()
    else:
        device = None
        rboxes_np = np.array(rboxes)
    
    single_box = False
    if rboxes_np.ndim == 1:
        rboxes_np = rboxes_np.reshape(1, -1)
        single_box = True
    
    cx, cy, w, h, angle = rboxes_np[..., 0], rboxes_np[..., 1], rboxes_np[..., 2], rboxes_np[..., 3], rboxes_np[..., 4]
    
    # 计算四个顶点 (相对于中心点)
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2], 
        [w/2, h/2],
        [-w/2, h/2]
    ])  # [4, 2]
    
    # 旋转变换
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # 扩展维度以支持批量处理
    cos_a = cos_a[..., np.newaxis, np.newaxis]  # [N, 1, 1]
    sin_a = sin_a[..., np.newaxis, np.newaxis]  # [N, 1, 1]
    
    # 旋转矩阵应用
    rotated_corners = np.zeros((len(rboxes_np), 4, 2))
    rotated_corners[..., 0] = corners[..., 0] * cos_a.squeeze(-1) - corners[..., 1] * sin_a.squeeze(-1)
    rotated_corners[..., 1] = corners[..., 0] * sin_a.squeeze(-1) + corners[..., 1] * cos_a.squeeze(-1)
    
    # 平移到实际位置
    rotated_corners[..., 0] += cx[..., np.newaxis]
    rotated_corners[..., 1] += cy[..., np.newaxis] 
    
    if single_box:
        rotated_corners = rotated_corners[0]
    
    if device is not None:
        return torch.from_numpy(rotated_corners).to(device).float()
    else:
        return rotated_corners

def poly_iou(poly1, poly2):
    """
    计算两个多边形的IoU
    Args:
        poly1: 多边形1顶点 [4, 2]
        poly2: 多边形2顶点 [4, 2]
    Returns:
        iou: IoU值
    """
    try:
        # 转换为整数坐标
        if isinstance(poly1, torch.Tensor):
            poly1 = poly1.detach().cpu().numpy()
        if isinstance(poly2, torch.Tensor):
            poly2 = poly2.detach().cpu().numpy()
            
        poly1 = poly1.astype(np.int32)
        poly2 = poly2.astype(np.int32)
        
        # 计算边界框
        x1_min, y1_min = poly1.min(axis=0)
        x1_max, y1_max = poly1.max(axis=0)
        x2_min, y2_min = poly2.min(axis=0)  
        x2_max, y2_max = poly2.max(axis=0)
        
        # 检查是否有重叠
        if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
            return 0.0
        
        # 创建mask计算重叠面积
        x_min = min(x1_min, x2_min) - 1
        y_min = min(y1_min, y2_min) - 1
        x_max = max(x1_max, x2_max) + 1
        y_max = max(y1_max, y2_max) + 1
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width <= 0 or height <= 0:
            return 0.0
        
        # 创建mask
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        
        # 偏移多边形坐标
        poly1_shifted = poly1.copy()
        poly2_shifted = poly2.copy()
        poly1_shifted[:, 0] -= x_min
        poly1_shifted[:, 1] -= y_min
        poly2_shifted[:, 0] -= x_min
        poly2_shifted[:, 1] -= y_min
        
        # 填充多边形
        cv2.fillPoly(mask1, [poly1_shifted], 1)
        cv2.fillPoly(mask2, [poly2_shifted], 1)
        
        # 计算交并比
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
        
    except Exception as e:
        # 如果出错，返回0
        return 0.0

def rbox_iou(rboxes1, rboxes2):
    """
    计算旋转框IoU矩阵
    Args:
        rboxes1: 旋转框组1 [N, 5]
        rboxes2: 旋转框组2 [M, 5] 
    Returns:
        iou_matrix: IoU矩阵 [N, M]
    """
    if len(rboxes1) == 0 or len(rboxes2) == 0:
        return np.zeros((len(rboxes1), len(rboxes2)))
    
    # 转换为多边形
    polys1 = rbox_to_poly(rboxes1)  # [N, 4, 2]
    polys2 = rbox_to_poly(rboxes2)  # [M, 4, 2]
    
    if isinstance(polys1, torch.Tensor):
        polys1 = polys1.detach().cpu().numpy()
    if isinstance(polys2, torch.Tensor):
        polys2 = polys2.detach().cpu().numpy()
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(polys1), len(polys2)))
    
    for i in range(len(polys1)):
        for j in range(len(polys2)):
            iou_matrix[i, j] = poly_iou(polys1[i], polys2[j])
    
    return iou_matrix

def rotated_nms(rboxes, scores, iou_threshold=0.5, score_threshold=0.0):
    """
    旋转框非极大值抑制
    Args:
        rboxes: 旋转框 [N, 5] (cx, cy, w, h, angle)
        scores: 置信度分数 [N]
        iou_threshold: IoU阈值
        score_threshold: 分数阈值
    Returns:
        keep_indices: 保留的框索引
    """
    if isinstance(rboxes, torch.Tensor):
        device = rboxes.device
        rboxes_np = rboxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
    else:
        device = None
        rboxes_np = np.array(rboxes)
        scores_np = np.array(scores)
    
    # 过滤低分框
    valid_mask = scores_np > score_threshold
    if not valid_mask.any():
        return []
    
    rboxes_np = rboxes_np[valid_mask]
    scores_np = scores_np[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 按分数排序
    order = np.argsort(-scores_np)
    
    keep = []
    while len(order) > 0:
        # 选择分数最高的框
        i = order[0]
        keep.append(valid_indices[i])
        
        if len(order) == 1:
            break
        
        # 计算当前框与剩余框的IoU
        current_rbox = rboxes_np[i:i+1]  # [1, 5]
        remaining_rboxes = rboxes_np[order[1:]]  # [remaining, 5]
        
        ious = rbox_iou(current_rbox, remaining_rboxes)[0]  # [remaining]
        
        # 保留IoU小于阈值的框
        suppress_mask = ious <= iou_threshold
        order = order[1:][suppress_mask]
    
    return keep

def soft_rotated_nms(rboxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    软旋转框非极大值抑制 (Soft-NMS)
    Args:
        rboxes: 旋转框 [N, 5] (cx, cy, w, h, angle)
        scores: 置信度分数 [N]
        iou_threshold: IoU阈值
        sigma: 高斯权重参数
        score_threshold: 最终分数阈值
    Returns:
        keep_indices: 保留的框索引
        updated_scores: 更新后的分数
    """
    if isinstance(rboxes, torch.Tensor):
        rboxes_np = rboxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
    else:
        rboxes_np = np.array(rboxes)
        scores_np = np.array(scores)
    
    N = len(rboxes_np)
    if N == 0:
        return [], []
    
    # 初始化
    updated_scores = scores_np.copy()
    keep_mask = np.ones(N, dtype=bool)
    
    for i in range(N):
        if not keep_mask[i]:
            continue
            
        # 计算当前框与所有其他框的IoU
        current_rbox = rboxes_np[i:i+1]
        ious = rbox_iou(current_rbox, rboxes_np)[0]
        
        # 对于IoU大于阈值的框，降低其分数
        for j in range(i + 1, N):
            if not keep_mask[j]:
                continue
                
            iou = ious[j]
            if iou > iou_threshold:
                # 使用高斯衰减
                weight = np.exp(-(iou ** 2) / sigma)
                updated_scores[j] *= weight
                
                # 如果分数太低，直接移除
                if updated_scores[j] < score_threshold:
                    keep_mask[j] = False
    
    # 返回保留的框和更新后的分数
    keep_indices = np.where(keep_mask & (updated_scores > score_threshold))[0].tolist()
    final_scores = updated_scores[keep_indices].tolist()
    
    return keep_indices, final_scores

def multiclass_rotated_nms(rboxes, scores, class_ids, iou_threshold=0.5, score_threshold=0.0):
    """
    多类别旋转框NMS
    Args:
        rboxes: 旋转框 [N, 5]
        scores: 置信度分数 [N]  
        class_ids: 类别ID [N]
        iou_threshold: IoU阈值
        score_threshold: 分数阈值
    Returns:
        keep_indices: 保留的框索引
    """
    if len(rboxes) == 0:
        return []
    
    unique_classes = np.unique(class_ids)
    all_keep_indices = []
    
    for cls_id in unique_classes:
        # 获取当前类别的框
        cls_mask = class_ids == cls_id
        cls_rboxes = rboxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        # 对当前类别进行NMS
        keep_indices = rotated_nms(
            cls_rboxes, 
            cls_scores, 
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )
        
        # 转换回原始索引
        original_indices = cls_indices[keep_indices]
        all_keep_indices.extend(original_indices.tolist())
    
    return sorted(all_keep_indices)

def rbox_distance_weighted_nms(rboxes, scores, iou_threshold=0.5, distance_threshold=50, 
                              score_threshold=0.0, distance_weight=0.1):
    """
    基于距离加权的旋转框NMS
    对于距离较远的框，即使IoU较高也可能保留
    """
    if isinstance(rboxes, torch.Tensor):
        rboxes_np = rboxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
    else:
        rboxes_np = np.array(rboxes)
        scores_np = np.array(scores)
    
    # 过滤低分框
    valid_mask = scores_np > score_threshold
    if not valid_mask.any():
        return []
    
    rboxes_np = rboxes_np[valid_mask]
    scores_np = scores_np[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 按分数排序
    order = np.argsort(-scores_np)
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(valid_indices[i])
        
        if len(order) == 1:
            break
        
        current_rbox = rboxes_np[i]
        remaining_rboxes = rboxes_np[order[1:]]
        
        # 计算IoU
        ious = rbox_iou(current_rbox[np.newaxis], remaining_rboxes)[0]
        
        # 计算中心点距离
        current_center = current_rbox[:2]
        remaining_centers = remaining_rboxes[:, :2]
        distances = np.linalg.norm(remaining_centers - current_center, axis=1)
        
        # 距离权重调整IoU阈值
        distance_weights = np.exp(-distances / distance_threshold)
        adjusted_thresholds = iou_threshold * (1 - distance_weight * (1 - distance_weights))
        
        # 保留调整后IoU小于阈值的框
        suppress_mask = ious <= adjusted_thresholds
        order = order[1:][suppress_mask]
    
    return keep

def cluster_rotated_nms(rboxes, scores, iou_threshold=0.5, cluster_threshold=0.3, score_threshold=0.0):
    """
    聚类旋转框NMS
    先将框聚类，然后在每个聚类内进行NMS
    """
    if len(rboxes) == 0:
        return []
        
    if isinstance(rboxes, torch.Tensor):
        rboxes_np = rboxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
    else:
        rboxes_np = np.array(rboxes)
        scores_np = np.array(scores)
    
    # 过滤低分框
    valid_mask = scores_np > score_threshold
    if not valid_mask.any():
        return []
    
    rboxes_np = rboxes_np[valid_mask]
    scores_np = scores_np[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    N = len(rboxes_np)
    
    # 计算IoU矩阵进行聚类
    iou_matrix = rbox_iou(rboxes_np, rboxes_np)
    
    # 简单的连通分量聚类
    visited = np.zeros(N, dtype=bool)
    clusters = []
    
    for i in range(N):
        if visited[i]:
            continue
            
        # BFS构建聚类
        cluster = []
        queue = [i]
        visited[i] = True
        
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            
            # 找相邻节点
            neighbors = np.where((iou_matrix[curr] > cluster_threshold) & (~visited))[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(neighbor)
        
        clusters.append(cluster)
    
    # 在每个聚类内进行NMS
    all_keep_indices = []
    for cluster in clusters:
        cluster_rboxes = rboxes_np[cluster]
        cluster_scores = scores_np[cluster]
        cluster_indices = [valid_indices[idx] for idx in cluster]
        
        keep_indices = rotated_nms(
            cluster_rboxes,
            cluster_scores, 
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )
        
        # 转换回原始索引
        original_indices = [cluster_indices[idx] for idx in keep_indices]
        all_keep_indices.extend(original_indices)
    
    return sorted(all_keep_indices)

def visualize_rboxes(img, rboxes, scores=None, class_ids=None, class_names=None, 
                    colors=None, thickness=2, font_scale=0.5):
    """
    可视化旋转框
    Args:
        img: 输入图像
        rboxes: 旋转框 [N, 5] (cx, cy, w, h, angle)
        scores: 置信度分数 [N] (可选)
        class_ids: 类别ID [N] (可选) 
        class_names: 类别名称列表 (可选)
        colors: 颜色列表 (可选)
        thickness: 线条粗细
        font_scale: 字体大小
    Returns:
        标注后的图像
    """
    if len(rboxes) == 0:
        return img
    
    result_img = img.copy()
    
    # 默认颜色
    if colors is None:
        colors = [
            (0, 255, 0),   # 绿色
            (255, 0, 0),   # 蓝色
            (0, 0, 255),   # 红色
            (255, 255, 0), # 青色
            (255, 0, 255), # 品红
            (0, 255, 255), # 黄色
        ]
    
    # 转换为多边形
    polys = rbox_to_poly(rboxes)
    if isinstance(polys, torch.Tensor):
        polys = polys.detach().cpu().numpy()
    
    for i, poly in enumerate(polys):
        # 选择颜色
        if class_ids is not None:
            color = colors[int(class_ids[i]) % len(colors)]
        else:
            color = colors[i % len(colors)]
        
        # 画旋转框
        poly_int = poly.astype(np.int32)
        cv2.polylines(result_img, [poly_int], isClosed=True, color=color, thickness=thickness)
        
        # 添加标签
        if scores is not None or class_names is not None:
            # 计算标签位置 (框的顶部中心)
            top_center = np.mean([poly[0], poly[1]], axis=0).astype(int)
            
            label_parts = []
            if class_names is not None and class_ids is not None:
                class_name = class_names[int(class_ids[i])] if int(class_ids[i]) < len(class_names) else f'cls_{int(class_ids[i])}'
                label_parts.append(class_name)
            
            if scores is not None:
                label_parts.append(f'{scores[i]:.2f}')
            
            label = ': '.join(label_parts)
            
            # 计算文本尺寸
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # 画标签背景
            cv2.rectangle(result_img, 
                         (top_center[0] - text_w//2, top_center[1] - text_h - baseline - 5),
                         (top_center[0] + text_w//2, top_center[1] - 5),
                         color, -1)
            
            # 画标签文本
            cv2.putText(result_img, label, 
                       (top_center[0] - text_w//2, top_center[1] - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return result_img

def test_rotated_nms():
    """测试旋转框NMS功能"""
    print("测试旋转框NMS功能...")
    
    # 创建测试数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成一些重叠的旋转框
    rboxes = []
    scores = []
    class_ids = []
    
    # 第一组：密集重叠的框
    center = [100, 100]
    for i in range(5):
        cx = center[0] + np.random.normal(0, 10)
        cy = center[1] + np.random.normal(0, 10)
        w = 50 + np.random.normal(0, 5)
        h = 30 + np.random.normal(0, 5)
        angle = np.random.uniform(-np.pi, np.pi)
        
        rboxes.append([cx, cy, w, h, angle])
        scores.append(np.random.uniform(0.6, 0.95))
        class_ids.append(0)
    
    # 第二组：另一个位置的框
    center = [200, 150]
    for i in range(4):
        cx = center[0] + np.random.normal(0, 8)
        cy = center[1] + np.random.normal(0, 8)
        w = 40 + np.random.normal(0, 3)
        h = 60 + np.random.normal(0, 3)
        angle = np.random.uniform(-np.pi, np.pi)
        
        rboxes.append([cx, cy, w, h, angle])
        scores.append(np.random.uniform(0.5, 0.9))
        class_ids.append(1)
    
    # 第三组：孤立的框
    for i in range(3):
        cx = np.random.uniform(50, 300)
        cy = np.random.uniform(250, 350)
        w = np.random.uniform(20, 50)
        h = np.random.uniform(20, 50)
        angle = np.random.uniform(-np.pi, np.pi)
        
        rboxes.append([cx, cy, w, h, angle])
        scores.append(np.random.uniform(0.4, 0.8))
        class_ids.append(np.random.randint(0, 3))
    
    rboxes = np.array(rboxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    print(f"生成测试数据: {len(rboxes)} 个旋转框")
    print(f"分数范围: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"类别: {np.unique(class_ids)}")
    
    # 测试标准NMS
    print("\n1. 测试标准旋转NMS:")
    keep_indices = rotated_nms(rboxes, scores, iou_threshold=0.5, score_threshold=0.3)
    print(f"   NMS前: {len(rboxes)} 个框")
    print(f"   NMS后: {len(keep_indices)} 个框")
    print(f"   保留索引: {keep_indices[:10]}...")  # 只显示前10个
    
    # 测试Soft-NMS
    print("\n2. 测试Soft-NMS:")
    soft_keep_indices, updated_scores = soft_rotated_nms(
        rboxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.3
    )
    print(f"   Soft-NMS后: {len(soft_keep_indices)} 个框")
    print(f"   分数更新示例: {scores[:5].round(3)} -> {np.array(updated_scores[:5]).round(3)}")
    
    # 测试多类别NMS
    print("\n3. 测试多类别NMS:")
    multiclass_keep = multiclass_rotated_nms(
        rboxes, scores, class_ids, iou_threshold=0.5, score_threshold=0.3
    )
    print(f"   多类别NMS后: {len(multiclass_keep)} 个框")
    
    # 测试距离加权NMS
    print("\n4. 测试距离加权NMS:")
    distance_keep = rbox_distance_weighted_nms(
        rboxes, scores, iou_threshold=0.5, distance_threshold=50, 
        score_threshold=0.3, distance_weight=0.2
    )
    print(f"   距离加权NMS后: {len(distance_keep)} 个框")
    
    # 测试聚类NMS
    print("\n5. 测试聚类NMS:")
    cluster_keep = cluster_rotated_nms(
        rboxes, scores, iou_threshold=0.5, cluster_threshold=0.3, score_threshold=0.3
    )
    print(f"   聚类NMS后: {len(cluster_keep)} 个框")
    
    # 测试IoU计算
    print("\n6. 测试IoU计算:")
    if len(rboxes) >= 2:
        test_iou = rbox_iou(rboxes[:2], rboxes[:2])
        print(f"   前两个框的IoU矩阵:\n{test_iou}")
        
        # 单独计算
        poly1 = rbox_to_poly(rboxes[0])
        poly2 = rbox_to_poly(rboxes[1])
        single_iou = poly_iou(poly1, poly2)
        print(f"   单独计算IoU: {single_iou:.4f}")
    
    # 可视化测试 (生成示例图像)
    print("\n7. 生成可视化示例:")
    img = np.ones((400, 350, 3), dtype=np.uint8) * 255
    
    # 原始框
    img_original = visualize_rboxes(
        img.copy(), rboxes, scores, class_ids, 
        class_names=['person', 'car', 'truck']
    )
    
    # NMS后的框
    nms_rboxes = rboxes[keep_indices]
    nms_scores = scores[keep_indices]
    nms_class_ids = class_ids[keep_indices]
    
    img_nms = visualize_rboxes(
        img.copy(), nms_rboxes, nms_scores, nms_class_ids,
        class_names=['person', 'car', 'truck']
    )
    
    print(f"   原始图像形状: {img_original.shape}")
    print(f"   NMS后图像形状: {img_nms.shape}")
    
    # 保存图像 (如果需要的话)
    try:
        cv2.imwrite('test_rboxes_original.jpg', img_original)
        cv2.imwrite('test_rboxes_nms.jpg', img_nms)
        print("   可视化图像已保存")
    except:
        print("   无法保存图像 (可能缺少OpenCV)")
    
    print("\n旋转框NMS测试完成！")

if __name__ == "__main__":
    test_rotated_nms()