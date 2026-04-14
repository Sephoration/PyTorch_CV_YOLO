from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./yolo-model/yolo11n.pt')
    # Train the model
    results = model.train(data='./coco128.yaml', epochs=2, imgsz=640, workers=0)
