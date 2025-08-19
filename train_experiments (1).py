# train_experiments.py
import os
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil
import subprocess
import argparse

# 导入自定义模块
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    HAS_ULTRALYTICS = True
except ImportError:
    print("警告: 未安装ultralytics，某些功能可能无法使用")
    HAS_ULTRALYTICS = False

try:
    from augment_small_objects import main as augment_main
    HAS_AUGMENT = True
except ImportError:
    print("警告: 无法导入数据增强模块")
    HAS_AUGMENT = False

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, base_dir: str = "experiments", log_level: str = "INFO"):
        """
        Args:
            base_dir: 实验基础目录
            log_level: 日志级别
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.log_level = log_level
        
        # 创建实验记录文件
        self.experiment_log = self.base_dir / "experiment_log.json"
        if not self.experiment_log.exists():
            self._save_json({}, self.experiment_log)
        
        print(f"实验管理器初始化完成，基础目录: {self.base_dir}")
    
    def _save_json(self, data: Dict, file_path: Path):
        """保存JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_json(self, file_path: Path) -> Dict:
        """加载JSON文件"""
        if not file_path.exists():
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_experiment_config(self, exp_name: str, config: Dict) -> Path:
        """
        创建实验配置文件
        Args:
            exp_name: 实验名称
            config: 配置字典
        Returns:
            配置文件路径
        """
        # 创建实验目录
        exp_dir = self.base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # 添加默认配置
        default_config = {
            "experiment_name": exp_name,
            "created_time": datetime.now().isoformat(),
            "description": f"实验: {exp_name}",
            
            # 模型配置
            "model": "yolov8n.pt",
            "custom_model": False,
            "add_p2": False,
            "decouple_head": False,
            
            # 数据配置
            "data": "dataset.yaml",
            "use_augmented_data": False,
            "augment_config": {
                "copy_paste": True,
                "mosaic": 0.5,
                "mixup": 0.1,
                "small_object_focus": True
            },
            
            # 训练配置
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "device": "0",
            "workers": 8,
            "patience": 50,
            "save_period": 10,
            
            # 优化器配置
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            
            # 损失函数配置
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 1.0,
            
            # 数据增强配置
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            
            # 验证配置
            "val": True,
            "split": "val",
            "save_json": True,
            "save_hybrid": False,
            "conf": 0.001,
            "iou": 0.6,
            "max_det": 300,
            "half": False,
            "dnn": False,
            "plots": True,
            
            # 推理配置
            "tile_inference": False,
            "tile_size": 1024,
            "overlap_ratio": 0.25,
        }
        
        # 合并用户配置
        final_config = {**default_config, **config}
        
        # 保存配置文件
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(final_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"实验配置已创建: {config_path}")
        return config_path
    
    def prepare_data(self, config: Dict) -> str:
        """
        准备训练数据
        Args:
            config: 实验配置
        Returns:
            数据配置文件路径
        """
        data_yaml = config.get("data", "dataset.yaml")
        
        # 如果需要使用增强数据
        if config.get("use_augmented_data", False) and HAS_AUGMENT:
            print("开始数据增强...")
            
            # 执行数据增强
            try:
                augment_main()
                print("数据增强完成")
                
                # 修改数据配置文件路径
                if Path("dataset_aug").exists():
                    # 创建增强数据的yaml配置
                    aug_data_yaml = "dataset_aug.yaml"
                    self._create_augmented_data_yaml(data_yaml, aug_data_yaml)
                    data_yaml = aug_data_yaml
                    
            except Exception as e:
                print(f"数据增强失败: {e}")
                print("使用原始数据进行训练")
        
        return data_yaml
    
    def _create_augmented_data_yaml(self, original_yaml: str, output_yaml: str):
        """创建增强数据的yaml配置文件"""
        try:
            # 读取原始配置
            with open(original_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 修改路径指向增强数据
            if 'train' in config:
                config['train'] = config['train'].replace('dataset/', 'dataset_aug/')
            if 'val' in config:
                # 验证集通常不增强，保持原样或复制到增强目录
                pass
            if 'test' in config:
                # 测试集通常不增强
                pass
            
            # 保存新配置
            with open(output_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"增强数据配置已创建: {output_yaml}")
            
        except Exception as e:
            print(f"创建增强数据配置失败: {e}")
            shutil.copy(original_yaml, output_yaml)
    
    def create_custom_model(self, config: Dict) -> Optional[str]:
        """
        创建自定义模型
        Args:
            config: 配置字典
        Returns:
            模型配置文件路径或None
        """
        if not config.get("custom_model", False):
            return None
        
        try:
            from model_custom import create_custom_model
            
            model, model_config_path = create_custom_model(
                model_path=config.get("model", "yolov8n.pt"),
                nc=config.get("nc", 80),
                add_p2=config.get("add_p2", False),
                decouple=config.get("decouple_head", False)
            )
            
            print(f"自定义模型已创建: {model_config_path}")
            return model_config_path
            
        except Exception as e:
            print(f"创建自定义模型失败: {e}")
            return None
    
    def run_experiment(self, config_path: str) -> Dict:
        """
        运行单个实验
        Args:
            config_path: 配置文件路径
        Returns:
            实验结果字典
        """
        if not HAS_ULTRALYTICS:
            raise ImportError("需要安装ultralytics包")
        
        print(f"\n{'='*50}")
        print(f"开始实验: {config_path}")
        print(f"{'='*50}")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        exp_name = config.get("experiment_name", "default")
        exp_dir = self.base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        try:
            # 1. 准备数据
            print("1. 准备训练数据...")
            data_yaml = self.prepare_data(config)
            
            # 2. 创建模型
            print("2. 创建模型...")
            model_path = config.get("model", "yolov8n.pt")
            
            # 如果需要自定义模型
            custom_model_path = self.create_custom_model(config)
            if custom_model_path:
                model_path = custom_model_path
            
            model = YOLO(model_path)
            
            # 3. 设置训练参数
            print("3. 配置训练参数...")
            train_args = {
                'data': data_yaml,
                'epochs': config.get('epochs', 100),
                'batch': config.get('batch', 16),
                'imgsz': config.get('imgsz', 640),
                'device': config.get('device', '0'),
                'workers': config.get('workers', 8),
                'patience': config.get('patience', 50),
                'save_period': config.get('save_period', 10),
                'project': str(exp_dir),
                'name': 'train',
                'exist_ok': True,
                
                # 优化器参数
                'optimizer': config.get('optimizer', 'SGD'),
                'lr0': config.get('lr0', 0.01),
                'lrf': config.get('lrf', 0.01),
                'momentum': config.get('momentum', 0.937),
                'weight_decay': config.get('weight_decay', 0.0005),
                'warmup_epochs': config.get('warmup_epochs', 3.0),
                'warmup_momentum': config.get('warmup_momentum', 0.8),
                'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),
                
                # 损失函数权重
                'box': config.get('box', 7.5),
                'cls': config.get('cls', 0.5),
                'dfl': config.get('dfl', 1.5),
                
                # 数据增强参数
                'hsv_h': config.get('hsv_h', 0.015),
                'hsv_s': config.get('hsv_s', 0.7),
                'hsv_v': config.get('hsv_v', 0.4),
                'degrees': config.get('degrees', 0.0),
                'translate': config.get('translate', 0.1),
                'scale': config.get('scale', 0.5),
                'shear': config.get('shear', 0.0),
                'perspective': config.get('perspective', 0.0),
                'flipud': config.get('flipud', 0.0),
                'fliplr': config.get('fliplr', 0.5),
                'mosaic': config.get('mosaic', 1.0),
                'mixup': config.get('mixup', 0.0),
                'copy_paste': config.get('copy_paste', 0.0),
                
                # 验证参数
                'val': config.get('val', True),
                'save_json': config.get('save_json', True),
                'plots': config.get('plots', True),
            }
            
            print(f"训练参数: {train_args}")
            
            # 4. 开始训练
            print("4. 开始训练...")
            results = model.train(**train_args)
            
            # 5. 验证模型
            print("5. 验证模型...")
            val_args = {
                'data': data_yaml,
                'device': config.get('device', '0'),
                'conf': config.get('conf', 0.001),
                'iou': config.get('iou', 0.6),
                'max_det': config.get('max_det', 300),
                'half': config.get('half', False),
                'save_json': config.get('save_json', True),
                'plots': config.get('plots', True),
                'project': str(exp_dir),
                'name': 'val',
                'exist_ok': True,
            }
            
            val_results = model.val(**val_args)
            
            # 6. 整理结果
            end_time = time.time()
            training_time = end_time - start_time
            
            # 提取关键指标
            metrics = {
                'training_time': training_time,
                'final_epoch': results.epoch if hasattr(results, 'epoch') else config['epochs'],
                'best_fitness': float(results.fitness) if hasattr(results, 'fitness') else 0.0,
            }
            
            # 从验证结果中提取指标
            if val_results and hasattr(val_results, 'results_dict'):
                val_metrics = val_results.results_dict
                metrics.update({
                    'mAP50': float(val_metrics.get('metrics/mAP50(B)', 0.0)),
                    'mAP50-95': float(val_metrics.get('metrics/mAP50-95(B)', 0.0)),
                    'precision': float(val_metrics.get('metrics/precision(B)', 0.0)),
                    'recall': float(val_metrics.get('metrics/recall(B)', 0.0)),
                })
            
            # 7. 保存实验结果
            experiment_result = {
                'experiment_name': exp_name,
                'config': config,
                'metrics': metrics,
                'status': 'completed',
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'training_time': training_time,
                'model_path': str(exp_dir / 'train' / 'weights' / 'best.pt'),
                'log_path': str(exp_dir / 'train'),
            }
            
            # 保存结果到实验目录
            result_path = exp_dir / 'result.json'
            self._save_json(experiment_result, result_path)
            
            # 更新全局实验日志
            experiment_log = self._load_json(self.experiment_log)
            experiment_log[exp_name] = experiment_result
            self._save_json(experiment_log, self.experiment_log)
            
            print(f"\n实验 '{exp_name}' 完成!")
            print(f"训练时间: {training_time:.2f}秒")
            print(f"mAP50: {metrics.get('mAP50', 0.0):.4f}")
            print(f"mAP50-95: {metrics.get('mAP50-95', 0.0):.4f}")
            print(f"结果已保存到: {result_path}")
            
            return experiment_result
            
        except Exception as e:
            # 记录失败的实验
            error_result = {
                'experiment_name': exp_name,
                'config': config,
                'status': 'failed',
                'error': str(e),
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'training_time': time.time() - start_time,
            }
            
            error_path = exp_dir / 'error.json'
            self._save_json(error_result, error_path)
            
            # 更新全局日志
            experiment_log = self._load_json(self.experiment_log)
            experiment_log[exp_name] = error_result
            self._save_json(experiment_log, self.experiment_log)
            
            print(f"\n实验 '{exp_name}' 失败: {e}")
            return error_result
    
    def run_experiments(self, config_files: List[str]) -> Dict[str, Dict]:
        """
        运行多个实验
        Args:
            config_files: 配置文件路径列表
        Returns:
            所有实验结果字典
        """
        print(f"开始运行 {len(config_files)} 个实验...")
        
        all_results = {}
        for i, config_file in enumerate(config_files):
            print(f"\n进度: {i+1}/{len(config_files)}")
            result = self.run_experiment(config_file)
            all_results[result['experiment_name']] = result
        
        # 生成实验对比报告
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: Dict[str, Dict]):
        """生成实验对比报告"""
        print(f"\n{'='*60}")
        print("实验对比报告")
        print(f"{'='*60}")
        
        # 创建对比表格
        headers = ['实验名称', '状态', 'mAP50', 'mAP50-95', '精确率', '召回率', '训练时间(s)']
        print(f"{'':15} {'':8} {'':8} {'':10} {'':8} {'':8} {'':10}")
        print(f"{headers[0]:15} {headers[1]:8} {headers[2]:8} {headers[3]:10} {headers[4]:8} {headers[5]:8} {headers[6]:10}")
        print("-" * 75)
        
        successful_experiments = []
        for exp_name, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'completed':
                metrics = result.get('metrics', {})
                mAP50 = metrics.get('mAP50', 0.0)
                mAP50_95 = metrics.get('mAP50-95', 0.0)
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                training_time = metrics.get('training_time', 0.0)
                
                print(f"{exp_name[:15]:15} {'完成':8} {mAP50:.4f}   {mAP50_95:.6f}   {precision:.4f}   {recall:.4f}   {training_time:.1f}")
                successful_experiments.append((exp_name, mAP50))
            else:
                error_msg = result.get('error', 'Unknown error')[:20]
                print(f"{exp_name[:15]:15} {'失败':8} {'N/A':8} {'N/A':10} {'N/A':8} {'N/A':8} {'N/A':10} - {error_msg}")
        
        print("-" * 75)
        
        # 找出最佳实验
        if successful_experiments:
            best_exp = max(successful_experiments, key=lambda x: x[1])
            print(f"\n最佳实验: {best_exp[0]} (mAP50: {best_exp[1]:.4f})")
        
        # 保存详细报告
        report_data = {
            'generated_time': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful_experiments': len(successful_experiments),
            'failed_experiments': len(results) - len(successful_experiments),
            'best_experiment': best_exp[0] if successful_experiments else None,
            'results': results
        }
        
        report_path = self.base_dir / 'comparison_report.json'
        self._save_json(report_data, report_path)
        print(f"详细报告已保存到: {report_path}")
    
    def create_ablation_study(self, base_config: Dict, variations: List[Dict]) -> List[str]:
        """
        创建消融研究实验
        Args:
            base_config: 基础配置
            variations: 变化配置列表
        Returns:
            配置文件路径列表
        """
        config_files = []
        
        for i, variation in enumerate(variations):
            # 合并配置
            exp_config = {**base_config, **variation}
            exp_name = f"ablation_{i+1:02d}_{variation.get('name', f'var_{i+1}')}"
            exp_config['experiment_name'] = exp_name
            exp_config['description'] = f"消融研究 - {variation.get('description', exp_name)}"
            
            # 创建配置文件
            config_path = self.create_experiment_config(exp_name, exp_config)
            config_files.append(str(config_path))
        
        print(f"消融研究配置已创建: {len(config_files)} 个实验")
        return config_files

def create_predefined_experiments() -> List[Dict]:
    """创建预定义实验配置"""
    experiments = []
    
    # 基线实验
    experiments.append({
        'name': 'baseline_yolov8n',
        'description': '基线YOLOv8n模型',
        'model': 'yolov8n.pt',
        'epochs': 100,
        'batch': 16,
        'custom_model': False,
    })
    
    # 数据增强实验
    experiments.append({
        'name': 'augmented_data',
        'description': '使用数据增强',
        'model': 'yolov8n.pt',
        'epochs': 100,
        'batch': 16,
        'use_augmented_data': True,
        'copy_paste': 0.3,
        'mosaic': 0.8,
        'mixup': 0.15,
    })
    
    # P2层实验
    experiments.append({
        'name': 'custom_p2_layer',
        'description': '添加P2检测层',
        'model': 'yolov8n.pt',
        'epochs': 100,
        'batch': 16,
        'custom_model': True,
        'add_p2': True,
        'decouple_head': False,
    })
    
    # 解耦头实验
    experiments.append({
        'name': 'decoupled_head',
        'description': '使用解耦检测头',
        'model': 'yolov8n.pt',
        'epochs': 100,
        'batch': 16,
        'custom_model': True,
        'add_p2': False,
        'decouple_head': True,
    })
    
    # 完整改进实验
    experiments.append({
        'name': 'full_improvements',
        'description': '所有改进措施',
        'model': 'yolov8n.pt',
        'epochs': 150,
        'batch': 16,
        'use_augmented_data': True,
        'custom_model': True,
        'add_p2': True,
        'decouple_head': True,
        'copy_paste': 0.3,
        'mosaic': 0.8,
        'mixup': 0.15,
        'lr0': 0.008,
        'warmup_epochs': 5.0,
    })
    
    return experiments

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO实验管理器')
    parser.add_argument('--mode', choices=['single', 'batch', 'ablation', 'predefined'], 
                       default='predefined', help='运行模式')
    parser.add_argument('--config', help='单个实验配置文件路径')
    parser.add_argument('--configs', nargs='+', help='多个实验配置文件路径')
    parser.add_argument('--base_dir', default='experiments', help='实验基础目录')
    parser.add_argument('--data', default='dataset.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    # 创建实验管理器
    manager = ExperimentManager(base_dir=args.base_dir)
    
    if args.mode == 'single':
        # 单个实验
        if not args.config:
            print("错误: 单个实验模式需要指定--config参数")
            return
        
        result = manager.run_experiment(args.config)
        print(f"实验结果: {result}")
        
    elif args.mode == 'batch':
        # 批量实验
        if not args.configs:
            print("错误: 批量实验模式需要指定--configs参数")
            return
        
        results = manager.run_experiments(args.configs)
        print(f"批量实验完成，共 {len(results)} 个实验")
        
    elif args.mode == 'ablation':
        # 消融研究
        base_config = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'model': 'yolov8n.pt',
        }
        
        variations = [
            {'name': 'baseline', 'description': '基线模型'},
            {'name': 'augment', 'description': '数据增强', 'use_augmented_data': True},
            {'name': 'p2', 'description': 'P2层', 'custom_model': True, 'add_p2': True},
            {'name': 'decouple', 'description': '解耦头', 'custom_model': True, 'decouple_head': True},
            {'name': 'all', 'description': '全部改进', 'use_augmented_data': True, 'custom_model': True, 'add_p2': True, 'decouple_head': True},
        ]
        
        config_files = manager.create_ablation_study(base_config, variations)
        results = manager.run_experiments(config_files)
        print(f"消融研究完成，共 {len(results)} 个实验")
        
    elif args.mode == 'predefined':
        # 预定义实验
        predefined_experiments = create_predefined_experiments()
        config_files = []
        
        for exp_config in predefined_experiments:
            # 添加命令行参数
            exp_config.update({
                'data': args.data,
                'epochs': args.epochs,
                'batch': args.batch,
            })
            
            config_path = manager.create_experiment_config(exp_config['name'], exp_config)
            config_files.append(str(config_path))
        
        results = manager.run_experiments(config_files)
        print(f"预定义实验完成，共 {len(results)} 个实验")

if __name__ == "__main__":
    # 示例用法
    if len(import sys) > 1 and sys.argv[1:]:
        main()
    else:
        print("YOLO实验管理器")
        print("=" * 50)
        print("使用方法:")
        print("1. 运行预定义实验:")
        print("   python train_experiments.py --mode predefined --data your_dataset.yaml")
        print("2. 消融研究:")
        print("   python train_experiments.py --mode ablation --data your_dataset.yaml --epochs 50")
        print("3. 单个实验:")
        print("   python train_experiments.py --mode single --config experiments/exp1/config.yaml")
        print("4. 批量实验:")
        print("   python train_experiments.py --mode batch --configs exp1.yaml exp2.yaml exp3.yaml")
        
        # 如果没有命令行参数，创建示例实验
        print("\n创建示例实验配置...")
        
        manager = ExperimentManager()
        
        # 创建一个简单的测试实验
        test_config = {
            'model': 'yolov8n.pt',
            'data': 'coco8.yaml',  # 使用COCO8小数据集进行测试
            'epochs': 3,
            'batch': 2,
            'imgsz': 640,
            'device': 'cpu',  # 使用CPU避免GPU依赖问题
        }
        
        config_path = manager.create_experiment_config('test_experiment', test_config)
        print(f"测试实验配置已创建: {config_path}")
        
        if HAS_ULTRALYTICS:
            print("可以使用以下命令运行测试实验:")
            print(f"python train_experiments.py --mode single --config {config_path}")
        else:
            print("请先安装ultralytics包: pip install ultralytics")
