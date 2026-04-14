from ultralytics import YOLO

# Load a model
model = YOLO('./yolo-model/yolo11n.pt')

# Train the model
results = model.train(data='./coco128.yaml', epochs=2, imgsz=224, workers=0)
