import os
import shutil
import random
import yaml
import argparse
from pathlib import Path
import numpy as np

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# 定义数据集根目录和类别
DATASET_ROOT = Path('data/samples')
OUTPUT_ROOT = Path('data/yolo_dataset')

# 6个细胞类别及其对应的ID
CLASSES = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'lymphocyte': 3,
    'monocyte': 4,
    'platelet': 5
}

# 训练、验证、测试集的比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 数据增强参数（用于训练配置文件）
DATA_AUGMENTATION = {
    'hsv_h': 0.015,  # 色调增强
    'hsv_s': 0.7,    # 饱和度增强
    'hsv_v': 0.4,    # 明度增强
    'degrees': 0.0,  # 旋转角度
    'translate': 0.1, # 平移比例
    'scale': 0.5,    # 缩放比例
    'shear': 0.0,    # 剪切角度
    'perspective': 0.0, # 透视变换
    'flipud': 0.0,   # 上下翻转概率
    'fliplr': 0.5,   # 左右翻转概率
    'mosaic': 0.4,   # 马赛克增强概率
    'mixup': 0.0     # 混合增强概率
}

def prepare_directory_structure():
    """准备输出目录结构"""
    directories = [
        OUTPUT_ROOT / 'images' / 'train',
        OUTPUT_ROOT / 'images' / 'val',
        OUTPUT_ROOT / 'images' / 'test',
        OUTPUT_ROOT / 'labels' / 'train',
        OUTPUT_ROOT / 'labels' / 'val',
        OUTPUT_ROOT / 'labels' / 'test'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def process_class(class_name, class_id):
    """处理单个类别的数据"""
    class_dir = DATASET_ROOT / class_name
    images_dir = class_dir / 'images'
    labels_dir = class_dir / 'labels'
    
    # 获取所有图像文件
    image_files = list(images_dir.glob('*'))
    random.shuffle(image_files)
    
    total_images = len(image_files)
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)
    
    # 划分数据集
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    datasets = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # 复制文件并转换标签
    for split, files in datasets.items():
        for image_file in files:
            # 复制图像文件
            shutil.copy(image_file, OUTPUT_ROOT / 'images' / split)
            
            # 处理对应的标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # 转换标签ID并保存
                with open(OUTPUT_ROOT / 'labels' / split / f"{image_file.stem}.txt", 'w') as f:
                    for line in lines:
                        # YOLO格式: class_id x_center y_center width height
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # 将原来的0类改为当前类别ID
                            parts[0] = str(class_id)
                            f.write(' '.join(parts) + '\n')
    
    print(f"处理完成 {class_name}: {total_images} 张图像")
    print(f"  - 训练集: {len(train_files)} 张")
    print(f"  - 验证集: {len(val_files)} 张")
    print(f"  - 测试集: {len(test_files)} 张")

def create_yaml_config():
    """创建YOLO数据集配置文件，包含数据增强参数"""
    config = {
        'path': str(OUTPUT_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASSES),
        'names': list(CLASSES.keys()),
        # 添加数据增强参数
        'hsv_h': DATA_AUGMENTATION['hsv_h'],
        'hsv_s': DATA_AUGMENTATION['hsv_s'],
        'hsv_v': DATA_AUGMENTATION['hsv_v'],
        'degrees': DATA_AUGMENTATION['degrees'],
        'translate': DATA_AUGMENTATION['translate'],
        'scale': DATA_AUGMENTATION['scale'],
        'shear': DATA_AUGMENTATION['shear'],
        'perspective': DATA_AUGMENTATION['perspective'],
        'flipud': DATA_AUGMENTATION['flipud'],
        'fliplr': DATA_AUGMENTATION['fliplr'],
        'mosaic': DATA_AUGMENTATION['mosaic'],
        'mixup': DATA_AUGMENTATION['mixup']
    }
    
    with open(OUTPUT_ROOT / 'cell_dataset.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"创建配置文件: {OUTPUT_ROOT / 'cell_dataset.yaml'}")

def create_training_config():
    """创建训练配置文件，包含早停和其他优化参数"""
    train_config = {
        # 模型设置
        'model': 'yolo11n.pt',  # 使用YOLOv11 n版本模型
        
        # 训练参数
        'epochs': 50,           # 最大训练轮数
        'batch': 4,             # 批次大小
        'imgsz': 640,           # 图像大小
        'patience': 10,         # 早停耐心值（10轮没有改进就停止）
        
        # 优化器设置
        'optimizer': 'AdamW',    # 优化器
        'lr0': 0.001,            # 初始学习率
        'lrf': 0.01,             # 最终学习率（lr0 * lrf）
        'momentum': 0.937,       # 动量
        'weight_decay': 0.0005,  # 权重衰减
        
        # 学习率调度
        'cos_lr': True,          # 使用余弦退火学习率
        
        # 评估指标
        'save': True,            # 保存模型
        'save_period': 1,        # 每轮保存一次
        'cache': True,           # 缓存图像以加速训练
        'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',  # 自动选择设备
        
        # 数据集配置
        'data': str(OUTPUT_ROOT / 'cell_dataset.yaml'),
        
        # 类别损失权重
        'cls': 1.0,              # 类别损失权重
        
        # 早停设置
        'early_stopping': {
            'enable': True,
            'monitor': 'val/box_loss',  # 监控验证集box loss
            'min_delta': 0.0001,        # 最小变化量
            'patience': 10              # 耐心值
        }
    }
    
    with open(OUTPUT_ROOT / 'train_config.yaml', 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"创建训练配置文件: {OUTPUT_ROOT / 'train_config.yaml'}")
    
    # 同时创建一个训练脚本 - 修复语法错误
    train_script = '''import os
import yaml
from ultralytics import YOLO

# 加载配置
with open('data/yolo_dataset/train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载模型
model = YOLO(config['model'])  # 使用yolo11n.pt模型

# 准备训练参数
train_args = {
    'data': config['data'],
    'epochs': config['epochs'],
    'batch': config['batch'],
    'imgsz': config['imgsz'],
    'patience': config['patience'],
    'optimizer': config['optimizer'],
    'lr0': config['lr0'],
    'lrf': config['lrf'],
    'momentum': config['momentum'],
    'weight_decay': config['weight_decay'],
    'cos_lr': config['cos_lr'],
    'save': config['save'],
    'save_period': config['save_period'],
    'cache': config['cache'],
    'device': config['device'],
    'cls': config.get('cls', 1.0)  # 添加类别损失权重
}

# 开始训练
print("开始训练模型...")
results = model.train(**train_args)

print("训练完成！")
print(f"最佳模型保存在: {results.best}")

# 评估模型
print("评估模型性能...")
eval_results = model.val()

# 导出模型
print("导出模型...")
model.export(format='onnx')
print("模型导出完成！")
'''
    
    with open('train_yolo.py', 'w') as f:
        f.write(train_script)
    
    print("创建训练脚本: train_yolo.py")

def generate_sequence_files():
    """生成训练、验证和测试的序列文件"""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = OUTPUT_ROOT / 'images' / split
        image_files = list(images_dir.glob('*'))
        
        # 生成序列文件
        with open(OUTPUT_ROOT / f'{split}.txt', 'w') as f:
            for image_file in image_files:
                f.write(str(image_file.absolute()) + '\n')
        
        print(f"生成序列文件: {OUTPUT_ROOT / f'{split}.txt'} ({len(image_files)} 行)")

def main():
    print("开始准备YOLO数据集...")
    
    # 设置随机种子以保证结果可复现
    set_seed(42)
    
    # 准备目录结构
    prepare_directory_structure()
    
    # 统计总图像数量
    total_images = 0
    for class_name in CLASSES:
        class_dir = DATASET_ROOT / class_name
        if class_dir.exists():
            images_dir = class_dir / 'images'
            img_count = len(list(images_dir.glob('*')))
            total_images += img_count
            print(f"发现类别 {class_name}: {img_count} 张图像")
    
    print(f"\n总图像数量: {total_images} 张")
    
    # 处理每个类别
    for class_name, class_id in CLASSES.items():
        # 检查类别目录是否存在
        if (DATASET_ROOT / class_name).exists():
            process_class(class_name, class_id)
        else:
            print(f"警告: {class_name} 类别目录不存在，跳过处理")
    
    # 创建配置文件
    create_yaml_config()
    
    # 创建训练配置文件
    create_training_config()
    
    # 生成序列文件
    generate_sequence_files()
    
    print("\n数据集准备完成!")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("\n使用方法:")
    print("1. 安装依赖: pip install ultralytics numpy")
    print("2. 生成数据集: python generate_yolo_dataset.py")
    print("3. 训练模型: python train_yolo.py")

if __name__ == '__main__':
    main()