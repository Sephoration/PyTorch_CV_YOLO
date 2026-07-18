import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# 载入预训练模型
model = YOLO('yolo11s-pose.pt')

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 切换计算设备
model.to(device)
# model.cpu()  # CPU
# model.cuda() # GPU
# img_path = 'images/multi-person.jpeg'
img_path = './images/two_runners.jpg'

results = model(img_path)
# 预测框的所有类别(MS COCO数据集八十类)
print(results[0].names)

# 预测类别 ID
print(results[0].boxes.cls)

num_bbox = len(results[0].boxes.cls)
print('预测出 {} 个框'.format(num_bbox))

# 每个框的置信度
print(results[0].boxes.conf)

# 每个框的：左上角XY坐标、右下角XY坐标
print(results[0].boxes.xyxy)

bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
print(bboxes_xyxy)
