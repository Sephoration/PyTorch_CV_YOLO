from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s-pose.pt")

    results = model.train(data="Triangle_25_1.yaml", epochs=10, imgsz=640,workers=2)