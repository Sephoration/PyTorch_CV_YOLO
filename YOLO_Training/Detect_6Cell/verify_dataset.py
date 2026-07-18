import yaml
from pathlib import Path

def verify_dataset():
    print("=== 数据集验证 ===\n")
    
    # 检查配置文件
    config_path = Path('data/yolo_dataset/cell_dataset.yaml')
    if not config_path.exists():
        print("❌ 配置文件不存在: yolo_dataset/cell_dataset.yaml")
        return
    
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("数据集配置:")
    print(f"- 类别数量: {config['nc']}")
    print(f"- 类别名称: {config['names']}")
    print(f"- 训练路径: {config['train']}")
    print(f"- 验证路径: {config['val']}")
    print(f"- 测试路径: {config['test']}")
    
    # 统计各数据集数量
    all_good = True
    for split in ['train', 'val', 'test']:
        images_dir = Path('data/yolo_dataset') / 'images' / split
        labels_dir = Path('data/yolo_dataset') / 'labels' / split
        
        image_count = len(list(images_dir.glob('*')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"\n{split.upper()}集:")
        print(f"  图片数量: {image_count}")
        print(f"  标签数量: {label_count}")
        
        if image_count == label_count:
            print(f"  ✅ 图片和标签数量匹配")
        else:
            print(f"  ❌ 警告: 图片和标签数量不匹配!")
            all_good = False
        
        # 检查标签格式
        if label_count > 0:
            label_files = list(labels_dir.glob('*.txt'))
            sample_label = label_files[0]
            with open(sample_label, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                            if all(0 <= x <= 1 for x in coords):
                                print(f"  ✅ 标签格式正确: class_id={class_id}")
                            else:
                                print(f"  ⚠️ 警告: 坐标值不在[0,1]范围内")
                                all_good = False
                        except ValueError:
                            print(f"  ❌ 错误: 标签格式无效")
                            all_good = False
                    else:
                        print(f"  ❌ 错误: 标签应该有5个值，但找到{len(parts)}个")
                        all_good = False
                else:
                    print(f"  ⚠️ 警告: 标签文件为空")
    
    # 检查训练配置
    train_config_path = Path('data/yolo_dataset/train_config.yaml')
    if train_config_path.exists():
        with open(train_config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        
        print(f"\n训练配置:")
        print(f"- 模型: {train_config['model']}")
        print(f"- 批次大小: {train_config['batch']}")
        print(f"- 图像大小: {train_config['imgsz']}")
        print(f"- 训练轮数: {train_config['epochs']}")
    
    # 检查训练脚本
    train_script_path = Path('train_yolo.py')
    if train_script_path.exists():
        print(f"\n✅ 训练脚本已创建: train_yolo.py")
    else:
        print(f"\n❌ 训练脚本不存在")
        all_good = False
    
    print(f"\n{'='*40}")
    if all_good:
        print("✅ 所有检查通过！可以开始训练。")
    else:
        print("⚠️  发现一些问题，请检查上述警告。")

if __name__ == '__main__':
    verify_dataset()