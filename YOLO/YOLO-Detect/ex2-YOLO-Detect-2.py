from ultralytics import YOLO

model = YOLO('yolo11n.pt')   # pretrained YOLO11n model

results = model(['./images/bus.jpg', './images/zidane.jpg'])  # return a list of Results objects

i = 1
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()
    result.save(filename='result-{}.jpg'.format(i))
    i = i + 1
