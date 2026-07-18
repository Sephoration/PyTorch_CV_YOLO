import torch

ckpt = torch.load('./yolo-cls/best.pt', map_location='cpu', weights_only=False)
print(ckpt.keys())  # 一般会有 'model', 'optimizer', 'train_args' 等