# model_custom.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Detect, Conv, C2f, SPPF
from ultralytics.utils.tal import make_anchors
import math

class DecoupledHead(nn.Module):
    """解耦检测头"""
    def __init__(self, nc=80, ch=256):
        super().__init__()
        self.nc = nc  # 类别数
        self.ch = ch  # 输入通道数
        
        # 分类分支
        self.cls_convs = nn.Sequential(
            Conv(ch, ch, 3),
            Conv(ch, ch, 3),
            Conv(ch, ch, 3)
        )
        self.cls_pred = nn.Conv2d(ch, nc, 1)
        
        # 回归分支  
        self.reg_convs = nn.Sequential(
            Conv(ch, ch, 3),
            Conv(ch, ch, 3),
            Conv(ch, ch, 3)
        )
        self.reg_pred = nn.Conv2d(ch, 4, 1)
        
        # 置信度分支
        self.obj_pred = nn.Conv2d(ch, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 分类层使用focal loss的初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
    
    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        
        cls_score = self.cls_pred(cls_feat)
        bbox_pred = self.reg_pred(reg_feat) 
        obj_score = self.obj_pred(reg_feat)
        
        return cls_score, bbox_pred, obj_score

class CustomDetect(Detect):
    """自定义检测层，支持P2层和解耦头"""
    
    def __init__(self, nc=80, ch=(), add_p2=True, decouple=True):
        """
        Args:
            nc: 类别数
            ch: 输入通道数 [P3, P4, P5] 或 [P2, P3, P4, P5]
            add_p2: 是否添加P2层
            decouple: 是否使用解耦头
        """
        self.add_p2 = add_p2
        self.decouple = decouple
        
        if add_p2:
            # 添加P2层通道
            ch = tuple([ch[0]] + list(ch))  # [P2, P3, P4, P5]
        
        # 调用父类初始化
        super().__init__(nc, ch)
        
        # 重新定义检测头
        if decouple:
            self.cv2 = nn.ModuleList()  # 清空原有的检测头
            self.cv3 = nn.ModuleList()
            
            # 为每个特征层创建解耦头
            for ch_i in ch:
                self.cv2.append(DecoupledHead(nc, ch_i))
                self.cv3.append(DecoupledHead(4, ch_i))  # bbox回归始终是4个值
        
        print(f"CustomDetect初始化完成:")
        print(f"  - P2层: {'启用' if add_p2 else '禁用'}")
        print(f"  - 解耦头: {'启用' if decouple else '禁用'}")
        print(f"  - 特征层数: {len(ch)}")
        print(f"  - 输入通道: {ch}")

    def forward(self, x):
        """前向传播"""
        if not isinstance(x, list):
            x = list(x)
        
        # 如果启用P2但输入只有3层，需要处理
        if self.add_p2 and len(x) == 3:
            # 这种情况下，x应该是[P2, P3, P4, P5]，但模型可能只输出了[P3, P4, P5]
            # 这里简单复制P3作为P2（实际使用时需要修改backbone）
            p2_fake = F.interpolate(x[0], scale_factor=2, mode='nearest')
            x = [p2_fake] + x
        
        shape = x[0].shape  # BCHW
        
        if self.decouple:
            # 解耦头前向传播
            outputs = []
            for i in range(self.nl):
                cls_score, bbox_pred, obj_score = self.cv2[i](x[i])
                
                # 合并输出 [batch, anchors, cls+4+1]
                b, _, h, w = cls_score.shape
                cls_score = cls_score.view(b, self.nc, -1).permute(0, 2, 1)
                bbox_pred = bbox_pred.view(b, 4, -1).permute(0, 2, 1)
                obj_score = obj_score.view(b, 1, -1).permute(0, 2, 1)
                
                output = torch.cat([bbox_pred, obj_score, cls_score], dim=2)
                outputs.append(output)
            
            if self.training:
                return outputs
            else:
                # 推理时合并所有层的输出
                return torch.cat(outputs, dim=1)
        else:
            # 使用原始的检测头
            return super().forward(x)

class FPN_P2(nn.Module):
    """带P2层的特征金字塔网络"""
    
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 侧向连接
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1, bias=False)
            )
        
        # 输出卷积
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels) + 1):  # +1 for P2
            self.fpn_convs.append(
                Conv(out_channels, out_channels, 3)
            )
        
        # P2生成
        self.p2_conv = Conv(out_channels, out_channels, 3)
    
    def forward(self, features):
        """
        Args:
            features: [C3, C4, C5] backbone特征
        Returns:
            [P2, P3, P4, P5] FPN特征
        """
        # 构建FPN
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[i]))
        
        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
        
        # 生成最终特征
        fpn_outs = []
        
        # P2: 从P3上采样生成
        p2 = F.interpolate(laterals[0], scale_factor=2, mode='nearest')
        p2 = self.p2_conv(p2)
        fpn_outs.append(p2)
        
        # P3, P4, P5
        for i, fpn_conv in enumerate(self.fpn_convs[1:]):
            fpn_outs.append(fpn_conv(laterals[i]))
        
        return fpn_outs

class EnhancedYOLOv8(nn.Module):
    """增强的YOLOv8模型"""
    
    def __init__(self, cfg_path=None, nc=80, add_p2=True, decouple=True):
        super().__init__()
        self.nc = nc
        self.add_p2 = add_p2
        self.decouple = decouple
        
        # 加载基础模型配置
        if cfg_path:
            import yaml
            with open(cfg_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        else:
            # 默认YOLOv8n配置
            self.cfg = self._get_default_config()
        
        # 构建网络
        self._build_network()
        
        print(f"EnhancedYOLOv8模型构建完成:")
        print(f"  - 类别数: {nc}")
        print(f"  - P2层: {'启用' if add_p2 else '禁用'}")
        print(f"  - 解耦头: {'启用' if decouple else '禁用'}")
    
    def _get_default_config(self):
        """默认配置"""
        return {
            'backbone': [
                [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
                [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
                [-1, 3, C2f, [128, True]],
                [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
                [-1, 6, C2f, [256, True]],
                [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
                [-1, 6, C2f, [512, True]],
                [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
                [-1, 3, C2f, [1024, True]],
                [-1, 1, SPPF, [1024, 5]],  # 9
            ]
        }
    
    def _build_network(self):
        """构建网络结构"""
        # 这里简化处理，实际应该根据cfg构建完整网络
        # 假设已有backbone特征 [C3:256, C4:512, C5:1024]
        
        if self.add_p2:
            # 添加FPN生成P2
            self.neck = FPN_P2([256, 512, 1024], 256)
            head_channels = [256, 256, 256, 256]  # [P2, P3, P4, P5]
        else:
            head_channels = [256, 512, 1024]  # [P3, P4, P5]
        
        # 检测头
        self.head = CustomDetect(
            nc=self.nc, 
            ch=head_channels, 
            add_p2=self.add_p2, 
            decouple=self.decouple
        )
    
    def forward(self, x):
        """前向传播"""
        # 这里需要实际的backbone实现
        # 简化版本，假设输入就是特征图
        if hasattr(self, 'neck'):
            features = self.neck(x)  # [P2, P3, P4, P5]
        else:
            features = x  # [P3, P4, P5]
        
        return self.head(features)

def create_custom_model(model_path=None, nc=80, add_p2=True, decouple=True):
    """
    创建自定义模型
    Args:
        model_path: 预训练模型路径
        nc: 类别数
        add_p2: 是否添加P2层
        decouple: 是否使用解耦头
    """
    from ultralytics import YOLO
    import yaml
    
    # 创建自定义配置文件
    config = {
        'nc': nc,
        'depth_multiple': 0.33,
        'width_multiple': 0.25,
        'max_channels': 1024,
        
        # backbone
        'backbone': [
            [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
            [-1, 3, C2f, [128, True]],
            [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
            [-1, 6, C2f, [256, True]],
            [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
            [-1, 6, C2f, [512, True]],
            [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
            [-1, 3, C2f, [1024, True]],
            [-1, 1, SPPF, [1024, 5]],  # 9
        ],
        
        # head
        'head': [
            [[4, 6, 9], 1, CustomDetect, [nc, [256, 512, 1024], add_p2, decouple]],
        ] if not add_p2 else [
            [[2, 4, 6, 9], 1, CustomDetect, [nc, [128, 256, 512, 1024], add_p2, decouple]],
        ]
    }
    
    # 保存配置文件
    config_path = 'custom_yolov8.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建模型
    if model_path and Path(model_path).exists():
        # 从预训练模型加载
        model = YOLO(model_path)
        # 替换检测头
        if hasattr(model.model, 'model') and hasattr(model.model.model[-1], 'f'):
            old_head = model.model.model[-1]
            new_head = CustomDetect(nc=nc, ch=old_head.ch, add_p2=add_p2, decouple=decouple)
            model.model.model[-1] = new_head
    else:
        # 从配置创建新模型
        model = YOLO(config_path)
    
    return model, config_path

def test_custom_model():
    """测试自定义模型"""
    print("测试自定义检测头...")
    
    # 创建测试输入
    batch_size = 2
    if True:  # 测试P2版本
        features = [
            torch.randn(batch_size, 128, 160, 160),  # P2/4
            torch.randn(batch_size, 256, 80, 80),    # P3/8  
            torch.randn(batch_size, 512, 40, 40),    # P4/16
            torch.randn(batch_size, 1024, 20, 20)    # P5/32
        ]
        head = CustomDetect(nc=6, ch=[128, 256, 512, 1024], add_p2=True, decouple=True)
    else:  # 测试标准版本
        features = [
            torch.randn(batch_size, 256, 80, 80),    # P3/8
            torch.randn(batch_size, 512, 40, 40),    # P4/16  
            torch.randn(batch_size, 1024, 20, 20)    # P5/32
        ]
        head = CustomDetect(nc=6, ch=[256, 512, 1024], add_p2=False, decouple=True)
    
    # 前向传播测试
    head.train()
    train_outputs = head(features)
    print(f"训练模式输出: {len(train_outputs)} 层")
    for i, out in enumerate(train_outputs):
        print(f"  层{i}: {out.shape}")
    
    head.eval() 
    eval_output = head(features)
    print(f"推理模式输出: {eval_output.shape}")
    
    print("自定义模型测试完成！")

if __name__ == "__main__":
    from pathlib import Path
    
    # 测试自定义检测头
    test_custom_model()
    
    # 创建完整的自定义模型示例
    print("\n创建完整自定义模型...")
    try:
        model, config_path = create_custom_model(
            model_path=None,  # 可以指定预训练模型路径
            nc=6,  # 6个类别
            add_p2=True,
            decouple=True
        )
        print(f"模型创建成功，配置文件: {config_path}")
        
        # 测试训练
        # model.train(data='custom_dataset.yaml', epochs=1, imgsz=640, batch=16)
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        print("请确保已安装ultralytics包: pip install ultralytics")
