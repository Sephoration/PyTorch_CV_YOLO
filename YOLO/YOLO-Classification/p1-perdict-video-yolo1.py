import time
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import torch
from ultralytics import YOLO

# ---------- 中文标签（务必与训练时类别顺序一致） ----------
idx_to_labels = {0: '电池', 1: '玻璃', 2: '金属', 3: '有机的', 4: '纸张', 5: '塑料'}
NUM_CLASSES = len(idx_to_labels)

# ---------- 基本设置 ----------
n = 2  # 显示前 n 个类别
IMG_SIZE = 224  # YOLO Classify 默认 224
font = ImageFont.truetype('SimHei.ttf', 20)
border_color = (255, 255, 255)
border_width = 2
# ---------- 载入 YOLO 分类模型（官方接口） ----------
# 注意：只加载一次，放在主流程外部，避免每帧重复初始化
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO("./yolo-cls/best.pt")
# 可选：固定输入尺寸 & 设备，减少每帧的开销
model.overrides['imgsz'] = IMG_SIZE
model.overrides['device'] = device
model.overrides['verbose'] = False  # 关闭冗余日志

# ---------- 处理帧函数（改为使用 YOLO 官方 predict） ----------
def process_frame(img_bgr):
    start_time = time.time()

    # 1) BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 2) YOLO 官方推理
    results = model.predict(source=img_rgb, imgsz=IMG_SIZE, save=False, verbose=False)
    r = results[0]
    probs = r.probs
    scores = probs.data.cpu().numpy().squeeze()   # [num_classes]

    # 3) 取前 n 个类别（确保长度匹配）
    if scores.shape[0] != NUM_CLASSES:
        # 安全兜底：若类别数不一致，截断或补齐
        scores = scores[:NUM_CLASSES] if scores.shape[0] > NUM_CLASSES else \
                 np.pad(scores, (0, NUM_CLASSES - scores.shape[0]), constant_values=0.0)
    top_idx = scores.argsort()[::-1][:n]

    # 4) PIL 叠字（中文）
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    for i, cls_id in enumerate(top_idx):
        cls_name = idx_to_labels.get(int(cls_id), str(cls_id))  # 用中文标签
        conf = float(scores[cls_id])
        text = f'{cls_name:<10} {conf:>.3f}'
        text_pos = (30, 60 + 25 * i)

        # 描边（八方向）
        for dx, dy in [(-border_width, 0), (border_width, 0), (0, -border_width), (0, border_width),
                       (-border_width, -border_width), (-border_width, border_width),
                       (border_width, -border_width), (border_width, border_width)]:
            draw.text((text_pos[0] + dx, text_pos[1] + dy), text, font=font, fill=border_color)
        # 正文（红色）
        draw.text(text_pos, text, font=font, fill=(255, 0, 0, 1))

    # 5) 回到 BGR，并叠加 FPS
    out_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    fps = 1.0 / (time.time() - start_time + 1e-9)
    out_bgr = cv2.putText(out_bgr, 'FPS  ' + str(int(fps)), (30, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4, cv2.LINE_AA)
    return out_bgr

# ------ 开启影片逐帧实时处理模板 ------
VIDEO_PATH = './videos/ex2-show-video.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

# 无限循环，直到break被触发
while True:
    # 获取画面
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第 0 帧重新播放q
        continue

    ## !!!处理帧函数
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('Classification', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()
