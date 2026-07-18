import os
import shutil
import random
from tqdm import tqdm


# 指定根目录路径
dataset_root_tapget = "./datasets/woods-4class-512-1"
dataset_root_src = "./datasets/woods-4class-origin-512"    #dataset 来源
images_src = os.path. join(dataset_root_src, "images")
labels_src = os.path. join(dataset_root_src, "labels")


# Target:创建 Labels 以及底下的 train, val 文件夹
target_labels = os.path. join(dataset_root_target, "labe ls")| I
target_labels_train = os.path.join(target_labels, "train")
target_labels_val = os.path.join(target_labels, "val")
os.makedirs(target_labels, exist_ok=True)
os.makedirs(target_labels_train, exist_ok=True)
os.makedirs(target_labels_val, exist_ok=True)
