"""
手部MCP关键点检测训练
代码进行划分再进行训练
直接使用写好的hand_keypoints.yaml配置文件
数据集处理到runs目录进行训练和测试
避免影响原始数据集
使用gpu进行训练
"""

import os
import torch
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# ========================================
# 1. 数据集处理到runs目录
# ========================================

def prepare_dataset_in_runs(source_root='data', runs_root='runs/dataset'):
    """将数据集复制到runs目录并划分"""
    print(f"\n{'='*50}")
    print("1. 准备数据集到runs目录")
    print('='*50)
    
    # 创建runs目录结构
    runs_path = Path(runs_root)
    runs_images = runs_path / 'images'
    runs_labels = runs_path / 'labels'
    
    runs_images.mkdir(parents=True, exist_ok=True)
    runs_labels.mkdir(parents=True, exist_ok=True)
    
    # 源数据路径
    source_path = Path(source_root)
    source_images = source_path / 'images'
    source_labels = source_path / 'yolo'
    
    if not source_images.exists():
        print(f"错误: 源图片目录不存在 {source_images}")
        return None, None, None
    
    if not source_labels.exists():
        print(f"错误: 源标签目录不存在 {source_labels}")
        return None, None, None
    
    # 获取所有图片文件
    image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
    print(f"找到 {len(image_files)} 张源图片")
    
    # 配对图片和标签
    all_pairs = []
    for img_path in image_files:
        txt_path = source_labels / f"{img_path.stem}.txt"
        if txt_path.exists():
            all_pairs.append((img_path, txt_path))
        else:
            print(f"警告: {img_path.stem} 缺少标签")
    
    print(f"有效图片-标签对: {len(all_pairs)}")
    
    # 随机打乱
    random.shuffle(all_pairs)
    
    # 复制所有文件到runs/images和runs/labels
    print("\n复制文件到runs目录...")
    for img_path, label_path in all_pairs:
        shutil.copy2(img_path, runs_images / img_path.name)
        shutil.copy2(label_path, runs_labels / label_path.name)
    
    print(f"已复制到: {runs_root}")
    
    # 划分训练集和验证集 (360训练 + 44验证)
    val_ratio = 44 / len(all_pairs) if len(all_pairs) > 0 else 0.1
    val_count = int(len(all_pairs) * val_ratio)
    if val_count < 1:
        val_count = 1
    
    val_pairs = all_pairs[:val_count]
    train_pairs = all_pairs[val_count:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_pairs)} 张")
    print(f"  验证集: {len(val_pairs)} 张")
    
    # 在runs目录下创建train和val子目录
    train_img_dir = runs_path / 'train' / 'images'
    train_label_dir = runs_path / 'train' / 'labels'
    val_img_dir = runs_path / 'val' / 'images'
    val_label_dir = runs_path / 'val' / 'labels'
    
    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 复制训练集文件
    print("\n创建训练集...")
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        shutil.copy2(label_path, train_label_dir / label_path.name)
    
    # 复制验证集文件
    print("创建验证集...")
    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        shutil.copy2(label_path, val_label_dir / label_path.name)
    
    print("✓ 数据集准备完成")
    
    return str(runs_path), train_pairs, val_pairs

# ========================================
# 2. 训练函数
# ========================================

def train_in_runs(config_path):
    """在runs目录中训练模型"""
    print(f"\n{'='*50}")
    print("2. 开始训练")
    print('='*50)
    
    # 训练参数（直接写入代码）
    epochs = 100
    batch_size = 8
    img_size = 320
    workers = 4
    patience = 30
    project = 'runs/train'
    name = 'hand_mcp'
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if 'cuda' in str(device):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 直接使用YOLO11s-pose模型
    print("使用 YOLO11s-pose")
    model = YOLO('models/yolo11s-pose.pt')
    
    # 训练参数
    train_args = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'workers': min(4, workers),
        'patience': patience,
        
        # 数据增强
        'mosaic': 1.0,
        'mixup': 0.1,
        'fliplr': 0.1,
        
        # 几何变换
        'degrees': 15.0,
        'translate': 0.1,
        'scale': 0.3,
        
        # 学习率
        'lr0': 0.01,
        'warmup_epochs': 3,
        
        # 正则化
        'label_smoothing': 0.1,
        
        # 输出
        'project': project,
        'name': name,
        'exist_ok': True,
        'verbose': True,
        'plots': True,
        'save': True,
        'save_period': 10,
    }
    
    print(f"\n训练参数:")
    print(f"  配置文件: {config_path}")
    print(f"  尺寸: {img_size}×{img_size}")
    print(f"  批次: {batch_size}")
    print(f"  轮次: {epochs}")
    print(f"  早停: {patience}")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(**train_args)
    
    return results

# ========================================
# 3. 验证和测试函数
# ========================================

def validate_and_test(config_path, val_pairs):
    """验证模型并在验证集上测试"""
    print(f"\n{'='*50}")
    print("3. 验证和测试")
    print('='*50)
    
    model_path = 'models/hand_mcp_best.pt'
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return
    
    # 加载模型
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 验证模型
    print("验证模型...")
    val_results = model.val(
        data=config_path,
        batch=8,
        imgsz=320,
        device=device,
        plots=True,
    )
    
    if hasattr(val_results, 'box'):
        print(f"  mAP50: {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
    
    # 随机抽取验证集图片进行测试
    if val_pairs:
        print(f"\n从验证集随机测试 (共{len(val_pairs)}张)...")
        
        # 随机选择1-3张图片
        test_count = min(3, len(val_pairs))
        test_samples = random.sample(val_pairs, test_count)
        
        for i, (img_path, _) in enumerate(test_samples):
            print(f"\n测试图片 {i+1}/{test_count}: {img_path.name}")
            
            # 进行预测
            results = model.predict(
                source=str(img_path),
                imgsz=320,
                conf=0.5,
                save=True,
                project='runs/test',
                name=f'test_{i+1}',
                exist_ok=True,
            )
            
            # 显示关键点信息
            if results and len(results) > 0:
                result = results[0]
                if result.keypoints is not None:
                    print(f"  检测到手部，关键点数量: {len(result.keypoints.xy)}")
    
    print(f"\n测试结果保存在: runs/test/")

# ========================================
# 4. 主函数
# ========================================

def main():
    print(f"{'='*60}")
    print("手部MCP关键点检测训练")
    print("（数据集处理到runs目录）")
    print(f"{'='*60}")
    
    # 检查源数据集
    source_dir = 'data'
    if not os.path.exists(source_dir):
        print(f"错误: 源数据集目录不存在 {source_dir}")
        print("请创建目录结构:")
        print("  datasets/")
        print("    ├── images/      # 图片数据")
        print("    └── yolo/        # 标签文件")
        return
    
    # 直接使用已有的配置文件
    config_path = os.path.join(source_dir, 'hand_keypoints.yaml')
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}")
        return
    print(f"使用已有配置文件: {config_path}")
    
    # 1. 准备数据集到runs目录
    runs_dataset_path, train_pairs, val_pairs = prepare_dataset_in_runs(
        source_root=source_dir,
        runs_root='runs/dataset'
    )
    
    if not runs_dataset_path:
        return
    
    # 2. 训练模型
    train_in_runs(config_path)
    
    # 3. 验证和测试
    validate_and_test(config_path, val_pairs)
    
    print(f"\n{'='*60}")
    print("训练流程完成!")
    print(f"{'='*60}")
    print("输出目录:")
    print(f"  数据集: {runs_dataset_path}")
    print(f"  训练结果: runs/train/hand_mcp/")
    print(f"  最佳模型: runs/train/hand_mcp/weights/best.pt")
    print(f"  测试结果: runs/test/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()