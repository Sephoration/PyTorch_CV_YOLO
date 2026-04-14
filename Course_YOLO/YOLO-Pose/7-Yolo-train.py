from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11s-pose.pt")

    results = model.train(data="tr.yaml", epochs=2, imgsz=640)

