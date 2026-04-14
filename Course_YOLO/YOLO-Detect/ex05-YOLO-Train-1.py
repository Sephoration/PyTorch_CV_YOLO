from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

results = model.train(data='./woods-4class-512-1.yaml',
                      batch=4, epochs=2, imgsz=640, workers=0)


"""train log
Starting training for 2 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/2     0.658G      2.907      6.847      1.768          3        640: 100% ━━━━━━━━━━━━ 105/105 2.9it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 2.2it/s 5.9s
                   all        104        189   0.000942     0.0935     0.0106    0.00205

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        2/2     0.721G      2.823      5.862       1.67          2        640: 100% ━━━━━━━━━━━━ 105/105 4.6it/s 23.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 3.4it/s 3.8s
                   all        104        189      0.513      0.204      0.131     0.0524

2 epochs completed in 0.019 hours.
Optimizer stripped from D:\code\detect\runs\detect\train4\weights\last.pt, 5.5MB
Optimizer stripped from D:\code\detect\runs\detect\train4\weights\best.pt, 5.5MB

Validating D:\code\detect\runs\detect\train4\weights\best.pt...
Ultralytics 8.3.213  Python-3.11.13 torch-2.8.0+cu126 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
YOLO11n summary (fused): 100 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 3.5it/s 3.7s
                   all        104        189      0.512      0.203      0.132     0.0519
                  knot         21         22      0.349      0.455      0.245      0.113
            oil streak         49         65      0.317      0.171      0.131     0.0465
             black dot         52         96      0.383      0.188      0.148     0.0479
                 crack          6          6          1          0    0.00213   0.000408
Speed: 0.3ms preprocess, 3.3ms inference, 0.0ms loss, 1.8ms postprocess per image
Results saved to D:\code\detect\ runs\detect\ train4
"""