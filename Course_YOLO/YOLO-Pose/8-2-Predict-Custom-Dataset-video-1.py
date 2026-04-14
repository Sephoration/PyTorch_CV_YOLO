import cv2
import torch
import time
import ctypes

from PIL import Image
from ultralytics import YOLO

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 载入预训练模型
model = YOLO('Triangle_215_yolo11s_pretrain.pt')
model = model.to(device)
# ------ OpenCV可视化 ------
# 框(rectangle)：颜色、线宽
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 6                   # 框的线宽
# 框类别文字
bbox_labelstr = {
    'font_size':4,         # 字体大小
    'font_thickness':10,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-80,        # Y 方向，文字偏移距离，向下为正
}
# 关键点 BGR 配色
kpt_color_map = {
    0:{'name':'angle_30', 'color':[255, 0, 0], 'radius':40},      # 30度角点
    1:{'name':'angle_60', 'color':[0, 255, 0], 'radius':40},      # 60度角点
    2:{'name':'angle_90', 'color':[0, 0, 255], 'radius':40},      # 90度角点
}
# 关键点类别文字
kpt_labelstr = {
    'font_size':4,             # 字体大小
    'font_thickness':10,       # 字体粗细
    'offset_x':30,             # X 方向，文字偏移距离，向右为正
    'offset_y':120,            # Y 方向，文字偏移距离，向下为正
}
# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[196, 75, 255], 'thickness':12},        # 30度角点-60度角点
    {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[180, 187, 28], 'thickness':12},        # 30度角点-90度角点
    {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[47,255, 173], 'thickness':12},         # 60度角点-90度角点
]

def process_frame(img_bgr):

    start_time = time.time()

    # 推理预测
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 pil
    results = model(img_pil, save=False, verbose=False)

    # ------ 目标检测框与关键点 ------
    # 目标检测框数量
    num_bbox = len(results[0].boxes.cls)

    # 转成整数的 numpy array
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('int32')
    bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('int32')
    # ------ 遍历每个框，划框與关键点 ------
    for idx in range(num_bbox):  # 遍历每个框
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别(对于关键点检测，只有一个类别)
        bbox_label = results[0].names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])

        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
    # 画该框的骨架连接
    for skeleton in skeleton_map:
        # 获取起始点坐标
        srt_kpt_id = skeleton['srt_kpt_id']        # {'srt_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':5}
        srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
        srt_kpt_y = bbox_keypoints[srt_kpt_id][1]

        # 获取终止点坐标
        dst_kpt_id = skeleton['dst_kpt_id']
        dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
        dst_kpt_y = bbox_keypoints[dst_kpt_id][1]

        # 获取骨架连接颜色
        skeleton_color = skeleton['color']

        # 获取骨架连接线宽
        skeleton_thickness = skeleton['thickness']

        # 画骨架连接
        img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                           thickness=skeleton_thickness)
    for kpt_id in kpt_color_map:
        # 获取该关键点的颜色、半径、XY坐标
        kpt_color = kpt_color_map[kpt_id]['color']
        kpt_radius = kpt_color_map[kpt_id]['radius']
        kpt_x = bbox_keypoints[kpt_id][0]
        kpt_y = bbox_keypoints[kpt_id][1]

        # 画圆：图片、XY坐标、半径、颜色、线宽(-1为填充)
        img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

        # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        kpt_label = str(kpt_id)  # 写关键点类别 ID(二选一)
        # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称(二选一)

        img_bgr = cv2.putText(img_bgr, kpt_label,
                              (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                              kpt_labelstr['font_thickness'])
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  {:.2f}'.format(FPS)  # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25, (255, 0, 255), 2)

    return img_bgr

if __name__ == '__main__':
    input_video = 'triangle_7.mp4'

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_video}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    win_name = "Keypoint Detection (q: quit, SPACE: pause)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # 先读一帧拿到视频尺寸
    success, frame_bgr = cap.read()
    if not success:
        raise RuntimeError("无法读取第一帧")
    h, w = frame_bgr.shape[:2]
    cv2.resizeWindow(win_name, w, h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第0帧
    # 计算屏幕中心并移动显示窗口
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    cv2.moveWindow(win_name, x, y)
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    paused = False
    while True:
        if not paused:
            success, frame_bgr = cap.read()

            # 播放到结尾：回到第 0 帧继续播
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            out_bgr = process_frame(frame_bgr)

            cv2.imshow(win_name, out_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
    cap.release()
    cv2.destroyAllWindows()



