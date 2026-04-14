from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("./yolo-cls/best.pt")  # load a custom model
    results = model.predict(source='./images/GarbageClassifying/biological329.jpg',
			imgsz=224, save=True)