# main.py
"""
低空视频多目标检测系统 - 主入口脚本

功能模块：
1. 数据增强 (augment_small_objects.py)
2. 切片推理 (tile_inference.py)  
3. 自定义模型 (model_custom.py)
4. 旋转框损失 (obb_loss.py)
5. 旋转框NMS (obb_nms.py)
6. 实验管理 (train_experiments.py)

使用方法:
    python main.py --help
"""

import argparse
import sys
from pathlib import Path
import subprocess
import os

def setup_environment():
    """设置环境和依赖检查"""
    print("检查环境和依赖...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'opencv-python',
        'numpy',
        'pyyaml',
        'albumentations',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {missing_packages}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("所有依赖包已安装")
    return True

def create_directory_structure():
    """创建项目目录结构"""
    dirs = [
        'dataset/images/train',
        'dataset/images/val', 
        'dataset/labels/train',
        'dataset/labels/val',
        'dataset_aug/images/train',
        'dataset_aug/labels/train',
        'experiments',
        'runs',
        'results',
        'configs',
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("项目目录结构已创建")

def create_dataset_yaml():
    """创建数据集配置文件"""
    dataset_yaml = """
# 低空视频多目标检测数据集配置

# 训练/验证/测试数据路径
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test  # 可选

# 类别数量
nc: 6

# 类别名称 (根据实际情况修改)
names:
  0: person      # 人员
  1: car         # 汽车
  2: truck       # 卡车
  3: bus         # 公交车
  4: motorbike   # 摩托车
  5: bicycle     # 自行车

# 数据集信息
dataset_info:
  description: "低空视频多目标检测数据集"
  version: "1.0"
  year: 2024
  contributor: "AI Competition Team"
  date_created: "2024-01-01"
"""
    
    with open('dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(dataset_yaml)
    
    print("数据集配置文件已创建: dataset.yaml")

def run_data_augmentation(args):
    """运行数据增强"""
    print("开始数据增强...")
    try:
        from augment_small_objects import main as augment_main
        augment_main()
        print("数据增强完成!")
    except Exception as e:
        print(f"数据增强失败: {e}")
        return False
    return True

def run_training(args):
    """运行模型训练"""
    print("开始模型训练...")
    try:
        from train_experiments import ExperimentManager
        
        manager = ExperimentManager(base_dir='experiments')
        
        # 创建训练配置
        train_config = {
            'model': args.model,
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'use_augmented_data': args.augment,
            'custom_model': args.custom_model,
            'add_p2': args.add_p2,
            'decouple_head': args.decouple,
        }
        
        config_path = manager.create_experiment_config('training', train_config)
        result = manager.run_experiment(str(config_path))
        
        print(f"训练完成! 结果: {result['status']}")
        if result['status'] == 'completed':
            metrics = result.get('metrics', {})
            print(f"mAP50: {metrics.get('mAP50', 0.0):.4f}")
            print(f"mAP50-95: {metrics.get('mAP50-95', 0.0):.4f}")
            print(f"模型保存路径: {result.get('model_path', 'unknown')}")
        
        return result['status'] == 'completed'
        
    except Exception as e:
        print(f"训练失败: {e}")
        return False

def run_inference(args):
    """运行推理"""
    print("开始推理...")
    try:
        from tile_inference import TileInference
        
        # 创建推理器
        tile_inference = TileInference(
            model_path=args.model,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        input_path = Path(args.input)
        output_dir = args.output or 'results/inference'
        
        if input_path.is_file():
            # 单张图像
            results = tile_inference.predict_single_image(
                str(input_path),
                visualize=True,
                save_path=f"{output_dir}/{input_path.stem}_result{input_path.suffix}"
            )
            print(f"检测到 {len(results.get('boxes', []))} 个目标")
            
        elif input_path.is_dir():
            # 批量处理
            tile_inference.predict_batch(
                input_dir=str(input_path),
                output_dir=output_dir
            )
        else:
            print(f"输入路径不存在: {args.input}")
            return False
        
        print("推理完成!")
        return True
        
    except Exception as e:
        print(f"推理失败: {e}")
        return False

def run_experiments(args):
    """运行实验管理"""
    print("开始实验管理...")
    try:
        from train_experiments import ExperimentManager, create_predefined_experiments
        
        manager = ExperimentManager(base_dir='experiments')
        
        if args.exp_mode == 'predefined':
            # 预定义实验
            experiments = create_predefined_experiments()
            config_files = []
            
            for exp_config in experiments:
                exp_config.update({
                    'data': args.data,
                    'epochs': args.epochs,
                    'batch': args.batch,
                    'device': args.device,
                })
                
                config_path = manager.create_experiment_config(exp_config['name'], exp_config)
                config_files.append(str(config_path))
            
            results = manager.run_experiments(config_files)
            print(f"预定义实验完成，共 {len(results)} 个实验")
            
        elif args.exp_mode == 'ablation':
            # 消融研究
            base_config = {
                'data': args.data,
                'epochs': args.epochs,
                'batch': args.batch,
                'model': args.model,
                'device': args.device,
            }
            
            variations = [
                {'name': 'baseline', 'description': '基线模型'},
                {'name': 'augment', 'description': '数据增强', 'use_augmented_data': True},
                {'name': 'p2', 'description': 'P2层', 'custom_model': True, 'add_p2': True},
                {'name': 'decouple', 'description': '解耦头', 'custom_model': True, 'decouple_head': True},
                {'name': 'combined', 'description': '组合改进', 'use_augmented_data': True, 'custom_model': True, 'add_p2': True, 'decouple_head': True},
            ]
            
            config_files = manager.create_ablation_study(base_config, variations)
            results = manager.run_experiments(config_files)
            print(f"消融研究完成，共 {len(results)} 个实验")
        
        return