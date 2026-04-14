from ultralytics import YOLO
from matplotlib import pyplot as plt
# 模型
model = YOLO('Triangle_215_yolo11s_pretrain.pt')
# 输入图像
img_path = './images/triangle_4.jpg'
# 推理预测
results = model(img_path,verbose=False)
# 4. 绘制检测结果
# Ultralytics 的 plot() 返回的是适合 RGB 显示的图像
rgb_image = results[0].plot()    # 这里其实可以把名字改成 image 或 rgb_image

# 5. 用 Matplotlib 显示（plt.imshow 期望 RGB）
plt.imshow(rgb_image)
plt.axis('off')  # 去掉坐标轴
plt.show()