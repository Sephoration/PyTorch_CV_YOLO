from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

if __name__ == '__main__':
    model = YOLO("./yolo-model/yolos-cls.pt")

    results = model.train(data='./datasets/garbage', epochs=10, imgsz=320)
