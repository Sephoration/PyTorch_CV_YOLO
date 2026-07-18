from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("./yolo-cls/best.pt")  # load a custom model

    # Validate the model
    # no arguments needed, dataset and settings remembered
    metrics = model("./images/GarbageClassification")
