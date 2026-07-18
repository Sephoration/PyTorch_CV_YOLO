# PyTorch CV & YOLO

计算机视觉、深度学习与 YOLO 目标检测课程项目。

---

## 目录结构

### Computer_Vision

MediaPipe 手部/姿态关键点检测、手势识别与 PySide6 桌面应用。

| 子目录 | 说明 |
|---|---|
| **MediaPipe_Learning** | 手部关键点检测、姿态估计、手指计数、瑜伽姿势分类 |
| **GestureImage** | 基于手势控制的图片浏览器 |
| **HandTracking** | 手部关键点检测基础与手指计数 |
| **PySide6-GUI** | 集成手势控制与 YOLO 追踪的桌面应用 |

### Deep_Learning

强化学习与 YOLO 目标追踪实践。

| 子目录 | 说明 |
|---|---|
| **RL_FrozenLake** | 强化学习入门：多臂老虎机、策略迭代、FrozenLake |
| **YOLO26Tracking** | 目标追踪：区域分类、轨迹绘制、变道检测、计数监控 |
| **Exp_Tracking** | YOLO 追踪完整流程与 GUI 集成 |

### YOLO

YOLO 全流程专项：图像分类、目标检测、姿态估计。

| 子目录 | 说明 |
|---|---|
| **YOLO-Classification** | 图像分类：训练、验证、预测 |
| **YOLO-Detect** | 目标检测：预测、数据集划分、训练 |
| **YOLO-Pose** | 姿态估计：标注转换、训练、推理 |
| **YOLO-PySide6-GUI** | 综合检测与追踪可视化界面 |
| **23_PyTorch_1** | PyTorch 基础：CIFAR-10 分类 |

### YOLO_Training

YOLO 模型训练项目：分类、检测、姿态估计。

| 子目录 | 说明 |
|---|---|
| **Classification_8Cell** | 细胞 8 分类训练 |
| **Detect_6Cell** | 细胞 6 类目标检测训练 |
| **Pose_hand** | 手部关键点姿态估计训练 |
| **.pt** | 训练产出的模型权重文件 |

---

## 环境依赖

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- MediaPipe
- PySide6
