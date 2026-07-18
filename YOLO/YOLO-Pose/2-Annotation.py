import cv2
import numpy as np
import json

import matplotlib.pyplot as plt

img_path = "./images/1.jpg"
img_bgr = cv2.imread(img_path)
plt.imshow(img_bgr[:,:,::-1])
# plt.show()


# 载入labelme格式的json标注文件
labelme_path = "images/1_labelme.json"
with open(labelme_path, 'r', encoding='utf-8') as f:
    labelme = json.load(f)


bbox_color = (255, 129, 0)           # 框的颜色
bbox_thickness = 5                      # 框的线宽

# 框类别文字
bbox_labelstr = {
        'font_size':6,         # 字体大小
    'font_thickness':14,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-80,       # Y 方向，文字偏移距离，向下为正
}

for each_ann in labelme['shapes']:  # 遍历每一个标注
    if each_ann['shape_type'] == 'rectangle':  # 筛选出框标注
        # 框的类别
        bbox_label = each_ann['label']
        # 框的两点坐标
        bbox_keypoints = each_ann['points']
        bbox_keypoint_A_xy = bbox_keypoints[0]
        bbox_keypoint_B_xy = bbox_keypoints[1]
        # 左上角坐标
        bbox_top_left_x = int(min(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_top_left_y = int(min(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))
        # 右下角坐标
        bbox_bottom_right_x = int(max(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_bottom_right_y = int(max(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))

        # 画矩形：画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_top_left_x, bbox_top_left_y), (bbox_bottom_right_x, bbox_bottom_right_y),
                                bbox_color, bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label, (
            bbox_top_left_x + bbox_labelstr['offset_x'], bbox_top_left_y + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])

# 可视化
plt.imshow(img_bgr[:, :, ::-1])
plt.show()