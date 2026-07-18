import os

image_folder = "./datasets/Triangle_labelme_25/images"
label_folder = "./datasets/Triangle_labelme_25/yolo"

# 指定根目录路径
dataset_root_target = "./datasets/Triangle_25_1"      # 分配目标路径
dataset_root_src = "./datasets/Triangle_labelme_25"  # dataset 来源
images_src = os.path.join(dataset_root_src, "images")
labels_src = os.path.join(dataset_root_src, "yolo")
