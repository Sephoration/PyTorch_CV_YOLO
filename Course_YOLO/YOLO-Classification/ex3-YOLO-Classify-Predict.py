from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("./yolo-model/yolo11s-cls.pt")

    print(model.names)
