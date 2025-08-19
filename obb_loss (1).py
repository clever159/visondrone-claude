# obb_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

class GaussianWassersteinDistance(nn.Module):
    """高斯Wasserstein距离损失 (GWD Loss)"""
    
    def __init__(self, tau=1.0, alpha=1.0, normalize=True):
        """
        Args:
            tau: 温度参数
            alpha: 损失权重
            normalize: 是否归一化
        """
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.normalize = normalize
    
    def forward(self, pred_rboxes, target_rboxes):
        """
        Args:
            pred_rboxes: 预测旋转框 [N, 5] (cx, cy, w, h, angle)
            target_rboxes: 目标旋转框 [N, 5] (cx, cy, w, h, angle)
        Returns:
            GWD损失
        """
        if pred_rboxes.size(0) == 0:
            return torch.tensor(0., device=pred_rboxes.device)
        
        # 转换为协方差矩阵表示
        pred_mu, pred_sigma = self._rbox_to_gaussian(pred_rboxes)
        target_mu, target_sigma = self._rbox_to_gaussian(target_rboxes)
        
        # 计算Wasserstein距离
        wd = self._wasserstein_distance(pred_mu, pred_sigma, target_mu, target_sigma)
        
        if self.normalize:
            wd = wd / (pred_rboxes.size(0) + 1e-6)
        
        return self.alpha * wd
    
    def _rbox_to_gaussian(self, rboxes):
        """将旋转框转换为高斯分布参数"""
        cx, cy, w, h, angle = rboxes.unbind(-1)
        
        # 均值 (中心点)
        mu = torch.stack([cx, cy], dim=-1)  # [N, 2]
        
        # 协方差矩阵
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # 旋转矩阵
        R11 = cos_angle
        R12 = -sin_angle
        R21 = sin_angle
        R22 = cos_angle
        
        # 对角协方差矩阵 (宽高的平方/12)
        D11 = w * w / 12
        D22 = h * h / 12
        D12 = torch.zeros_like(D11)
        
        # 旋转后的协方差矩阵 Sigma = R @ D @ R^T
        sigma11 = R11 * R11 * D11 + R12 * R12 * D22
        sigma12 = R11 * R21 * D11 + R12 * R22 * D22
        sigma21 = sigma12  # 对称矩阵
        sigma22 = R21 * R21 * D11 + R22 * R22 * D22
        
        # 组合协方差矩阵 [N, 2, 2]
        sigma = torch.stack([
            torch.stack([sigma11, sigma12], dim=-1),
            torch.stack([sigma21, sigma22], dim=-1)
        ], dim=-2)
        
        return mu, sigma
    
    def _wasserstein_distance(self, mu1, sigma1, mu2, sigma2):
        """计算两个高斯分布之间的Wasserstein距离"""
        # 均值差
        mu_diff = mu1 - mu2  # [N, 2]
        mu_dist = torch.sum(mu_diff * mu_diff, dim=-1)  # [N]
        
        # 协方差项
        sigma_sqrt1 = self._matrix_sqrt(sigma1)  # [N, 2, 2]
        sigma_sqrt2 = self._matrix_sqrt(sigma2)  # [N, 2, 2]
        
        # 计算 trace(sigma1 + sigma2 - 2*sqrt(sqrt(sigma1)*sigma2*sqrt(sigma1)))
        trace_sigma1 = torch.diagonal(sigma1, dim1=-2, dim2=-1).sum(-1)  # [N]
        trace_sigma2 = torch.diagonal(sigma2, dim1=-2, dim2=-1).sum(-1)  # [N]
        
        # 中间项计算
        middle_term = torch.matmul(sigma_sqrt1, torch.matmul(sigma2, sigma_sqrt1))
        middle_sqrt = self._matrix_sqrt(middle_term)
        trace_middle = 2 * torch.diagonal(middle_sqrt, dim1=-2, dim2=-1).sum(-1)
        
        sigma_dist = trace_sigma1 + trace_sigma2 - trace_middle
        sigma_dist = torch.clamp(sigma_dist, min=0)  # 确保非负
        
        # 总距离
        wd = mu_dist + sigma_dist
        return torch.mean(wd)
    
    def _matrix_sqrt(self, matrix):
        """计算2x2矩阵的平方根"""
        # 对于2x2矩阵，使用解析公式
        a = matrix[..., 0, 0]
        b = matrix[..., 0, 1] 
        c = matrix[..., 1, 0]
        d = matrix[..., 1, 1]
        
        # 计算特征值
        trace = a + d
        det = a * d - b * c
        
        lambda1 = (trace + torch.sqrt(trace * trace - 4 * det)) / 2
        lambda2 = (trace - torch.sqrt(trace * trace - 4 * det)) / 2
        
        sqrt_lambda1 = torch.sqrt(torch.clamp(lambda1, min=1e-8))
        sqrt_lambda2 = torch.sqrt(torch.clamp(lambda2, min=1e-8))
        
        # 重构平方根矩阵 (简化处理)
        sqrt_a = torch.sqrt(torch.clamp(a, min=1e-8))
        sqrt_d = torch.sqrt(torch.clamp(d, min=1e-8))
        sqrt_matrix = torch.stack([
            torch.stack([sqrt_a, b / (sqrt_a + sqrt_d + 1e-8)], dim=-1),
            torch.stack([c / (sqrt_a + sqrt_d + 1e-8), sqrt_d], dim=-1)
        ], dim=-2)
        
        return sqrt_matrix

class ProbIoULoss(nn.Module):
    """概率IoU损失 (ProbIoU Loss)"""
    
    def __init__(self, eps=1e-3, mode='l1'):
        """
        Args:
            eps: 小值防止除零
            mode: 损失模式 ('l1', 'l2')
        """
        super().__init__()
        self.eps = eps
        self.mode = mode
    
    def forward(self, pred_rboxes, target_rboxes):
        """
        Args:
            pred_rboxes: 预测旋转框 [N, 5] 
            target_rboxes: 目标旋转框 [N, 5]
        Returns:
            ProbIoU损失
        """
        if pred_rboxes.size(0) == 0:
            return torch.tensor(0., device=pred_rboxes.device)
        
        # 计算概率IoU
        prob_iou = self._calculate_prob_iou(pred_rboxes, target_rboxes)
        
        # 计算损失
        if self.mode == 'l1':
            loss = 1 - prob_iou
        elif self.mode == 'l2':
            loss = (1 - prob_iou) ** 2
        else:
            loss = -torch.log(prob_iou + self.eps)
        
        return torch.mean(loss)
    
    def _calculate_prob_iou(self, pred_rboxes, target_rboxes):
        """计算概率IoU"""
        # 将旋转框转换为协方差表示
        pred_mu, pred_sigma = self._rbox_to_gaussian(pred_rboxes)
        target_mu, target_sigma = self._rbox_to_gaussian(target_rboxes)
        
        # 计算重叠面积 (基于高斯分布)
        intersection = self._gaussian_intersection(pred_mu, pred_sigma, target_mu, target_sigma)
        
        # 计算各自面积
        pred_area = pred_rboxes[..., 2] * pred_rboxes[..., 3]
        target_area = target_rboxes[..., 2] * target_rboxes[..., 3]
        
        # IoU
        union = pred_area + target_area - intersection
        iou = intersection / (union + self.eps)
        
        return torch.clamp(iou, min=0, max=1)
    
    def _rbox_to_gaussian(self, rboxes):
        """旋转框转高斯分布 (简化版本)"""
        cx, cy, w, h, angle = rboxes.unbind(-1)
        mu = torch.stack([cx, cy], dim=-1)
        
        # 简化的协方差矩阵
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        w_var = (w / 4) ** 2
        h_var = (h / 4) ** 2
        
        sigma11 = cos_a * cos_a * w_var + sin_a * sin_a * h_var
        sigma12 = (cos_a * sin_a) * (w_var - h_var)
        sigma22 = sin_a * sin_a * w_var + cos_a * cos_a * h_var
        
        sigma = torch.stack([
            torch.stack([sigma11, sigma12], dim=-1),
            torch.stack([sigma12, sigma22], dim=-1)
        ], dim=-2)
        
        return mu, sigma
    
    def _gaussian_intersection(self, mu1, sigma1, mu2, sigma2):
        """计算两个高斯分布的重叠 (近似)"""
        # 简化计算，使用高斯分布的重叠近似
        mu_diff = mu1 - mu2
        sigma_sum = sigma1 + sigma2
        
        # 计算马氏距离
        sigma_inv = self._matrix_inverse_2x2(sigma_sum)
        mahalanobis = torch.sum(mu_diff.unsqueeze(-2) @ sigma_inv @ mu_diff.unsqueeze(-1), dim=(-2, -1))
        
        # 重叠近似
        overlap = torch.exp(-0.5 * mahalanobis)
        
        # 乘以面积因子
        det1 = torch.det(sigma1)
        det2 = torch.det(sigma2) 
        det_sum = torch.det(sigma_sum)
        
        area_factor = torch.sqrt(det1 * det2 / (det_sum + self.eps))
        
        return overlap * area_factor * (2 * math.pi)
    
    def _matrix_inverse_2x2(self, matrix):
        """2x2矩阵求逆"""
        a = matrix[..., 0, 0]
        b = matrix[..., 0, 1]
        c = matrix[..., 1, 0] 
        d = matrix[..., 1, 1]
        
        det = a * d - b * c
        det = torch.clamp(det, min=self.eps)
        
        inv_matrix = torch.stack([
            torch.stack([d, -b], dim=-1),
            torch.stack([-c, a], dim=-1)
        ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)
        
        return inv_matrix

class KLDivergenceLoss(nn.Module):
    """基于KL散度的旋转框损失"""
    
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
    
    def forward(self, pred_rboxes, target_rboxes):
        """
        计算KL散度损失
        """
        if pred_rboxes.size(0) == 0:
            return torch.tensor(0., device=pred_rboxes.device)
        
        pred_mu, pred_sigma = self._rbox_to_gaussian(pred_rboxes)
        target_mu, target_sigma = self._rbox_to_gaussian(target_rboxes)
        
        kl_div = self._kl_divergence(pred_mu, pred_sigma, target_mu, target_sigma)
        
        return torch.mean(kl_div) / self.tau
    
    def _rbox_to_gaussian(self, rboxes):
        """旋转框转高斯分布"""
        cx, cy, w, h, angle = rboxes.unbind(-1)
        mu = torch.stack([cx, cy], dim=-1)
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # 协方差矩阵
        w_var = (w / 6) ** 2
        h_var = (h / 6) ** 2
        
        sigma11 = cos_a * cos_a * w_var + sin_a * sin_a * h_var
        sigma12 = (cos_a * sin_a) * (w_var - h_var)
        sigma22 = sin_a * sin_a * w_var + cos_a * cos_a * h_var
        
        sigma = torch.stack([
            torch.stack([sigma11, sigma12], dim=-1),
            torch.stack([sigma12, sigma22], dim=-1)
        ], dim=-2)
        
        return mu, sigma
    
    def _kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """计算KL散度 KL(N1||N2)"""
        # KL(N1||N2) = 0.5 * [log(|Σ2|/|Σ1|) + tr(Σ2^(-1)Σ1) + (μ2-μ1)^T Σ2^(-1) (μ2-μ1) - k]
        
        # 计算行列式
        det1 = torch.det(sigma1) + 1e-8
        det2 = torch.det(sigma2) + 1e-8
        
        log_det_ratio = torch.log(det2 / det1)
        
        # 计算逆矩阵
        sigma2_inv = self._matrix_inverse_2x2(sigma2)
        
        # trace项
        trace_term = torch.diagonal(sigma2_inv @ sigma1, dim1=-2, dim2=-1).sum(-1)
        
        # 均值差项
        mu_diff = mu2 - mu1
        quad_term = torch.sum(mu_diff.unsqueeze(-2) @ sigma2_inv @ mu_diff.unsqueeze(-1), dim=(-2, -1)).squeeze()
        
        kl = 0.5 * (log_det_ratio + trace_term + quad_term - 2)
        
        return torch.clamp(kl, min=0)  # KL散度非负
    
    def _matrix_inverse_2x2(self, matrix):
        """2x2矩阵求逆"""
        a = matrix[..., 0, 0]
        b = matrix[..., 0, 1]
        c = matrix[..., 1, 0]
        d = matrix[..., 1, 1]
        
        det = a * d - b * c
        det = torch.clamp(det, min=1e-8)
        
        inv_matrix = torch.stack([
            torch.stack([d, -b], dim=-1),
            torch.stack([-c, a], dim=-1)
        ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)
        
        return inv_matrix

class CombinedOBBLoss(nn.Module):
    """组合的旋转框损失函数"""
    
    def __init__(self, gwd_weight=1.0, prob_iou_weight=1.0, kl_weight=0.5):
        """
        Args:
            gwd_weight: GWD损失权重
            prob_iou_weight: ProbIoU损失权重  
            kl_weight: KL散度损失权重
        """
        super().__init__()
        self.gwd_weight = gwd_weight
        self.prob_iou_weight = prob_iou_weight
        self.kl_weight = kl_weight
        
        self.gwd_loss = GaussianWassersteinDistance()
        self.prob_iou_loss = ProbIoULoss()
        self.kl_loss = KLDivergenceLoss()
    
    def forward(self, pred_rboxes, target_rboxes):
        """
        计算组合损失
        """
        total_loss = 0
        loss_dict = {}
        
        if self.gwd_weight > 0:
            gwd = self.gwd_loss(pred_rboxes, target_rboxes)
            total_loss += self.gwd_weight * gwd
            loss_dict['gwd'] = gwd
        
        if self.prob_iou_weight > 0:
            prob_iou = self.prob_iou_loss(pred_rboxes, target_rboxes)
            total_loss += self.prob_iou_weight * prob_iou
            loss_dict['prob_iou'] = prob_iou
        
        if self.kl_weight > 0:
            kl = self.kl_loss(pred_rboxes, target_rboxes)
            total_loss += self.kl_weight * kl
            loss_dict['kl'] = kl
        
        loss_dict['total'] = total_loss
        return total_loss, loss_dict

def test_obb_losses():
    """测试旋转框损失函数"""
    print("测试旋转框损失函数...")
    
    # 创建测试数据
    batch_size = 16
    
    # 预测框: (cx, cy, w, h, angle)
    pred_rboxes = torch.randn(batch_size, 5)
    pred_rboxes[..., :2] *= 50  # 中心点
    pred_rboxes[..., 2:4] = torch.abs(pred_rboxes[..., 2:4]) * 20 + 5  # 宽高 (正值)
    pred_rboxes[..., 4] = pred_rboxes[..., 4] * math.pi  # 角度 [-π, π]
    
    # 目标框 (加一些噪声)
    target_rboxes = pred_rboxes.clone()
    target_rboxes += torch.randn_like(target_rboxes) * 0.1
    target_rboxes[..., 2:4] = torch.abs(target_rboxes[..., 2:4])
    
    print(f"测试数据形状: {pred_rboxes.shape}")
    print(f"预测框范围: cx=[{pred_rboxes[..., 0].min():.2f}, {pred_rboxes[..., 0].max():.2f}], "
          f"cy=[{pred_rboxes[..., 1].min():.2f}, {pred_rboxes[..., 1].max():.2f}], "
          f"w=[{pred_rboxes[..., 2].min():.2f}, {pred_rboxes[..., 2].max():.2f}], "
          f"h=[{pred_rboxes[..., 3].min():.2f}, {pred_rboxes[..., 3].max():.2f}], "
          f"angle=[{pred_rboxes[..., 4].min():.2f}, {pred_rboxes[..., 4].max():.2f}]")
    
    # 测试各种损失函数
    print("\n1. 测试GWD损失:")
    gwd_loss = GaussianWassersteinDistance()
    gwd_value = gwd_loss(pred_rboxes, target_rboxes)
    print(f"   GWD Loss: {gwd_value.item():.6f}")
    
    print("\n2. 测试ProbIoU损失:")
    prob_iou_loss = ProbIoULoss()
    prob_iou_value = prob_iou_loss(pred_rboxes, target_rboxes)
    print(f"   ProbIoU Loss: {prob_iou_value.item():.6f}")
    
    print("\n3. 测试KL散度损失:")
    kl_loss = KLDivergenceLoss()
    kl_value = kl_loss(pred_rboxes, target_rboxes)
    print(f"   KL Divergence Loss: {kl_value.item():.6f}")
    
    print("\n4. 测试组合损失:")
    combined_loss = CombinedOBBLoss(gwd_weight=1.0, prob_iou_weight=2.0, kl_weight=0.5)
    total_loss, loss_dict = combined_loss(pred_rboxes, target_rboxes)
    print(f"   总损失: {total_loss.item():.6f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"   {key}: {value.item():.6f}")
    
    # 测试梯度
    print("\n5. 测试梯度计算:")
    pred_rboxes.requires_grad_(True)
    loss_value = combined_loss(pred_rboxes, target_rboxes)[0]
    loss_value.backward()
    
    if pred_rboxes.grad is not None:
        print(f"   梯度计算成功，梯度范数: {pred_rboxes.grad.norm().item():.6f}")
    else:
        print("   梯度计算失败")
    
    print("\n旋转框损失函数测试完成！")

if __name__ == "__main__":
    test_obb_losses()
