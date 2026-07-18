from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./yolo-cls-pet9/best.pt")
    metrics = model.val()
    metrics.top1
    metrics.top5