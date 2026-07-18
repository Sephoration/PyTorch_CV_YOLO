# 脚本名称：8-2-Predict-Custom-Dataset-video-1.py
# 脚本功能：使用YOLOv11-pose模型进行手部MCP关键点检测
# 作者：用户
# 日期：2024-08-06
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# 设置中文字体（如果需要显示中文标签）
# 注意：如果不需要显示中文，可以注释掉这部分
# 设置环境变量来防止OpenCV的一些警告
os.environ['PYTHONWARNINGS'] = 'ignore'

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
# 使用绝对路径确保文件正确加载
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_pose.pt')
model = YOLO(model_path, task='pose')
print(f"模型加载成功: {model_path}")

# 关键点配置 - 基于hand_keypoints.yaml
keypoint_colors = {
    0: (0, 255, 0),    # MCP_1 - 绿色
    1: (255, 0, 0),    # MCP_2 - 蓝色
    2: (0, 0, 255),    # MCP_3 - 红色
    3: (255, 255, 0)   # MCP_4 - 青色
}

keypoint_names = {
    0: "MCP_1",
    1: "MCP_2", 
    2: "MCP_3",
    3: "MCP_4"
}

# 骨架连接配置
skeleton = [(0, 1), (1, 2), (2, 3)]
skeleton_color = (255, 165, 0)  # 橙色

# 框配置
box_color = (0, 255, 255)  # 黄色
box_thickness = 2

# 类别名称
class_name = "hand"

# 视频路径
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_generated_video.mp4')

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit()

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频信息: {width}x{height}, {fps} FPS")

# 处理视频的函数
def process_frame(frame):
    # 使用模型进行预测
    results = model(frame, device=device, verbose=False)
    
    # 获取检测结果
    result = results[0]
    
    # 绘制边界框和关键点
    for i, box in enumerate(result.boxes):
        # 绘制边界框
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # 添加类别标签
        conf = box.conf[0].item()
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
        
        # 获取关键点
        keypoints = result.keypoints.xy[i].cpu().numpy()
        
        # 绘制骨架连接
        for sk in skeleton:
            if len(keypoints) > max(sk):
                pt1 = tuple(map(int, keypoints[sk[0]]))
                pt2 = tuple(map(int, keypoints[sk[1]]))
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(frame, pt1, pt2, skeleton_color, 2)
        
        # 绘制关键点
        for j, kp in enumerate(keypoints):
            if j in keypoint_colors and kp[0] > 0 and kp[1] > 0:
                # 绘制关键点圆圈
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, keypoint_colors[j], -1)
                
                # 显示关键点名称
                if j in keypoint_names:
                    cv2.putText(frame, keypoint_names[j], 
                                (int(kp[0]) + 10, int(kp[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, keypoint_colors[j], 1)
    
    return frame

# 主循环
cv2.namedWindow("Hand MCP Keypoint Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理当前帧
    processed_frame = process_frame(frame)
    
    # 显示结果
    cv2.imshow("Hand MCP Keypoint Detection", processed_frame)
    
    # 按ESC键退出
    if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("视频处理完成")