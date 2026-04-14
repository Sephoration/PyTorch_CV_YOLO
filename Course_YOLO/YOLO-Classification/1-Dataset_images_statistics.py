import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

fruit = r'D:\code\yolo11Start\fruit81_full'
garbage = r'D:\code\yolo11Start\Garbage Classification'
pet = r'./datasets/pet9'
image_dir = pet

# 用于存储图像信息的列表
data = []
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']  # 常见的图像文件扩展名

# 遍历所有子目录和文件
for root, dirs, files in os.walk(image_dir):
    for file in tqdm(files):    # for file in files:
        file_path = os.path.join(root, file)
        # 检查文件是否为图像文件
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            try:
                # 手动读取文件，解决中文路径问题
                with open(file_path, 'rb') as f:
                    img_array = f.read()
                    img = cv2.imdecode(np.frombuffer(img_array, np.uint8), cv2.IMREAD_COLOR)

                if img is not None:
                    # 将图像信息添加到列表中
                    data.append({'类别': os.path.basename(root), '文件名': file, '图像宽': img.shape[1],
                                 '图像高': img.shape[0]})
                else:
                    print(file_path, '无法读取')
            except Exception as e:
                print(file_path, '读取错误:', e)

# 将列表转换为 DataFrame
df = pd.DataFrame(data)

# 打印或输出DataFrame
print(df)
x = df['图像宽']
y = df['图像高']

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(10,10))
# plt.figure(figsize=(12,12))
plt.scatter(x, y, c=z,  s=5, cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])

plt.tick_params(labelsize=15)

xy_max = max(max(df['图像宽']), max(df['图像高']))
plt.xlim(xmin=0, xmax=xy_max)
plt.ylim(ymin=0, ymax=xy_max)

plt.ylabel('height', fontsize=25)
plt.xlabel('width', fontsize=25)

plt.savefig('图像尺寸分布.pdf', dpi=120, bbox_inches='tight')

plt.show()
