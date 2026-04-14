import os
import shutil
import random
from tqdm import tqdm

# 指定根目录路径
dataset_root_target = "./dataset/Triangle_labelme_25_1"      # 分配目标路径
dataset_root_src = "D:\code\YOLO11-pose\Triangle_labelme_25"  # dataset 来源
images_src = os.path.join(dataset_root_src, "images")
labels_src = os.path.join(dataset_root_src, "yolo")


# Target：创建 train 以及底下的 images, labels 文件夹
target_train = os.path.join(dataset_root_target, "train")
target_train_images = os.path.join(target_train, "images")
target_train_labels = os.path.join(target_train, "labels")
os.makedirs(target_train, exist_ok=True)
os.makedirs(target_train_images, exist_ok=True)
os.makedirs(target_train_labels, exist_ok=True)

# Target：创建 val 以及底下的 images, labels 文件夹
target_val = os.path.join(dataset_root_target, "val")
target_val_images = os.path.join(target_val, "images")
target_val_labels = os.path.join(target_val, "labels")
os.makedirs(target_val, exist_ok=True)
os.makedirs(target_val_images, exist_ok=True)
os.makedirs(target_val_labels, exist_ok=True)

test_frac = 0.2  # 测试集比例
random.seed(123) # 随机数种子，便于复现
img_paths = os.listdir(images_src)
random.shuffle(img_paths) # 随机打乱


# 产生训练集和测试集文件名列表
val_number = int(len(img_paths) * test_frac) # 测试集文件个数
train_files = img_paths[val_number:]         # 训练集文件名列表
val_files = img_paths[:val_number]           # 测试集文件名列表

print('数据集文件总数', len(img_paths))
print('训练集文件个数', len(train_files))
print('测试集文件个数', len(val_files))

# 处理训练集图片和标注档案
for each in tqdm(train_files):
    shutil.copy2(os.path.join(images_src, each), target_train_images)
    yolo_file = os.path.splitext(each)[0] + '.txt'
    shutil.copy2(os.path.join(labels_src, yolo_file), target_train_labels)

# 处理验证集图片和标注档案
for each in tqdm(val_files):
    shutil.copy2(os.path.join(images_src, each), target_val_images)
    yolo_file = os.path.splitext(each)[0] + '.txt'
    shutil.copy2(os.path.join(labels_src, yolo_file), target_val_labels)
