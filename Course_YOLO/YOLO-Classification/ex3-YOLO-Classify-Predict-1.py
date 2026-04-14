from ultralytics import YOLO

if __name__ == '__main__':
        # Load a model
        model = YOLO("./yolo-cls/best.pt")  # load a custom model
        # 推理单张图片
        # results = model.predict(source='./images/GarbageClassification/battery5.jpg',
        #                     imgsz=224, save=True)
        results = model.predict(source='./images/GarbageClassification/',
                                imgsz=224, save=True,verbose=False)
        print(results)