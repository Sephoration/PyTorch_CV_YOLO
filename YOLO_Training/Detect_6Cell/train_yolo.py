import os
import yaml
import torch
from ultralytics import YOLO

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU设备:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        return True, gpu_count
    else:
        print("❌ 未检测到GPU，将使用CPU训练")
        print("请检查:")
        print("1. NVIDIA驱动是否安装")
        print("2. CUDA Toolkit是否安装")
        print("3. PyTorch是否支持CUDA")
        return False, 0

def main():
    print("=" * 60)
    print("YOLOv11 GPU训练配置")
    print("=" * 60)
    
    # 检查GPU
    gpu_available, gpu_count = check_gpu()
    
    # 加载配置
    with open('data/yolo_dataset/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 强制使用GPU（如果可用）
    if gpu_available:
        # 使用所有可用的GPU
        device = '0'  # 使用第一个GPU，或使用 '0,1,2' 使用多个GPU
        if gpu_count > 1:
            print(f"\n🎮 检测到多个GPU，建议使用: device='0,1' 来使用前两个GPU")
            # 可以改为 device = '0,1' 来使用多个GPU
    else:
        device = 'cpu'
    
    print(f"\n⚙️  将使用设备: {device}")
    
    # 加载模型
    model = YOLO(config['model'])
    
    # 准备训练参数 - 强制使用GPU
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
        'device': device,  # 这里强制使用GPU
        'cls': config.get('cls', 1.0),
        'workers': 4,  # 增加数据加载工作线程
        'amp': True,   # 启用混合精度训练（节省显存，加速训练）
    }
    
    # 显示训练信息
    print(f"\n📊 训练配置:")
    print(f"   模型: {config['model']}")
    print(f"   批次大小: {config['batch']}")
    print(f"   图像尺寸: {config['imgsz']}")
    print(f"   训练轮数: {config['epochs']}")
    print(f"   设备: {device}")
    print(f"   混合精度: {'启用' if train_args['amp'] else '禁用'}")
    
    # 检查显存
    if gpu_available:
        torch.cuda.empty_cache()  # 清空缓存
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"   GPU空闲显存: {free_mem / 1024**3:.2f} GB")
        
        # 预估显存需求
        estimated_vram = (config['batch'] * config['imgsz'] * config['imgsz'] * 3 * 4) / (1024**3) * 2  # 粗略估计
        print(f"   预估显存需求: {estimated_vram:.2f} GB")
        
        if free_mem / 1024**3 < estimated_vram:
            print(f"   ⚠️  显存可能不足，建议:")
            print(f"      1. 减小批次大小 (当前: {config['batch']})")
            print(f"      2. 减小图像尺寸 (当前: {config['imgsz']})")
            print(f"      3. 使用梯度累积")
    
    print("\n" + "=" * 60)
    
    # 开始训练
    print("🚀 开始GPU训练...")
    results = model.train(**train_args)
    
    print("\n✅ 训练完成!")
    print(f"最佳模型: {results.best}")
    
    # 评估
    print("\n📈 评估模型性能...")
    eval_results = model.val()
    
    print(f"\n🎯 最终性能:")
    print(f"   mAP50: {eval_results.box.map50:.4f}")
    print(f"   mAP50-95: {eval_results.box.map:.4f}")
    print(f"   精确率: {eval_results.box.precision:.4f}")
    print(f"   召回率: {eval_results.box.recall:.4f}")
    
    # 导出模型
    print("\n💾 导出模型...")
    export_path = model.export(format='onnx')
    print(f"✅ 模型导出完成: {export_path}")
    
    print("\n" + "=" * 60)
    print("🎉 GPU训练全部完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()