import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2   # opencv-python
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw  # pillow

# 图像分类标签
n = 2           # 显示几个分类
# idx_to_labels = {0:'ants', 1:'bees'}
# ---------- 中文标签（务必与训练时类别顺序一致） ----------
idx_to_labels = {0: '电池', 1: '玻璃', 2: '金属', 3: '有机的', 4: '纸张', 5: '塑料'}
NUM_CLASSES = len(idx_to_labels)

# 图像预处理
# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop((224,224)),
                                     transforms.ToTensor()])

# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 20)
# 定义文字边框的颜色和偏移量（模拟边框）
border_color = (255, 255, 255)  # 白色边框
border_width = 2  # 边框宽度

# 载入预训练图像分类模型
ckpt = torch.load('./yolo-cls/best.pt', map_location='cpu', weights_only=False)
model = ckpt['model'].eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device).float()

# 处理帧函数
def process_frame(img):
    # 记录该帧开始处理的时间
    start_time = time.time()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)[0]  # 执行前向预测，得到所有类别的 logit 预测分数
    # pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    # top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    top_n = torch.topk(pred_logits, n)  # 执行前向预测，得到所有类别的 logit 预测分数
    # pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    # top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析置信度

    # 使用PIL绘制中文
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]]
        text = '{:<10} {:>.3f}'.format(pred_class, confs[i])
        text_position = (30, 60 + 25 * i)

        # 绘制边框（通过在八个方向分别绘制文字）
        for dx, dy in [(-border_width, 0), (border_width, 0), (0, -border_width), (0, border_width),
                       (-border_width, -border_width), (-border_width, border_width),
                       (border_width, -border_width), (border_width, border_width)]:
            draw.text((text_position[0] + dx, text_position[1] + dy), text, font=font, fill=border_color)

        # 文字坐标，中文字符串，字体，bgra颜色
        draw.text(text_position, text, font=font, fill=(255, 0, 0, 1))

    img = np.array(img_pil)  # PIL 转 array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB 转 BGR

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4,
                      cv2.LINE_AA)
    return img


# ------ 调用摄像头逐帧实时处理模板 ------
# 获取摄像头，传入0表示获取系统默认摄像头
# cap = cv2.VideoCapture(0)
# cap.open(0)    # 打开cap
# 开启影片
VIDEO_PATH = 'videos/ex2-show-video.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

# 无限循环，直到 break 被触发
while True:
    # 获取画面
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第0帧重新播放
        continue

    ## !!!处理帧函数
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('Classification', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()