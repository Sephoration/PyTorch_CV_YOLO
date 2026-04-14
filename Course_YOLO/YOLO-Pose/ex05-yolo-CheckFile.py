import os


image_folder = "D:\code\YOLO11-pose\Triangle_labelme_25\images"
label_folder = "D:\code\YOLO11-pose\Triangle_labelme_25\yolo"

# 获取 images 文件夹中的所有文件名（去除后缀）
image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder)]

# 获取 labels 文件夹中的所有文件名（去除后缀）
label_files = [os.path.splitext(file)[0] for file in os.listdir(label_folder)]

# 遍历 images 文件夹中的文件
for image_file in image_files:
    # 检查对应的 label 文件是否存在
    if image_file not in label_files:
        image_path = os.path.join(image_folder, image_file + ".jpg")
        # os.remove(image_path)  # 删除不匹配的图像文件
        print(f"Deleted image file: {image_file}.jpg")

# 遍历 labels 文件夹中的文件
for label_file in label_files:
    # 检查对应的 image 文件是否存在
    if label_file not in image_files:
        label_path = os.path.join(label_folder, label_file + ".txt")
        # os.remove(label_path)
        print(f"Deleted label file: {label_file}.txt")
