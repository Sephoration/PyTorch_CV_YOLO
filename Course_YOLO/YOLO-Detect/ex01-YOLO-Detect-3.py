from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('./runs/detect/train6/weights/last.pt')

    # Train the model
    results = model.train(resume=True)
