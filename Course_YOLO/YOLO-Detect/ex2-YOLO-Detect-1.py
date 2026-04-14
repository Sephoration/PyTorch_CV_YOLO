from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11n.pt')

# 推理（不设 imgsz，保持原始尺寸）
results = model.predict(
    source='./images/bus.jpg',  # 输入图片
    conf=0.5,                   # 置信度阈值
    show=True,                 # 不让 YOLO 内部显示
    save=True                  # 不保存，只显示
)
