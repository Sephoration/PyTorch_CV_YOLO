import os
from ultralytics.data.utils import check_det_dataset

# 检查数据集配置
dataset_path = "D:/code/YOLO11-pose/dataset/triangle"
try:
    dataset = check_det_dataset(f"tr.yaml")
    print("✅ 数据集验证通过")
    print(f"训练图像数量: {len(dataset['train'])}")
    print(f"验证图像数量: {len(dataset['val'])}")
except Exception as e:
    print(f"❌ 数据集验证失败: {e}")