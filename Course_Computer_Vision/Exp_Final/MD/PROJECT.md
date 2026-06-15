# 手势识别控制系统 — 开发记录

## 文件重命名（2026-06-14）

Python 模块名不能以数字开头，统一加 `p` 前缀：

```
1_extract_features.py      →  p1_extract_features.py
2_train_knn.py             →  p2_train_knn.py
3_main.py                  →  p3_main.py
3_gesture_worker.py        →  p3_gesture_worker.py
3_actions.py               →  p3_actions.py
3_hand_tracker.py          →  p3_hand_tracker.py
3_gesture_classifier.py    →  p3_gesture_classifier.py
```

---

## 参数优化（`p3_gesture_worker.py`）

| 参数 | 原值 | 新值 | 作用 |
|------|------|------|------|
| `LOCK_DURATION` | 1.5 | 0.9 | 锁定等待从 1.5 秒减到 0.9 秒 |
| `SMOOTH_WINDOW` | 9 | 7 | 平滑窗口从 9 帧减到 7 帧 |
| `UNLOCK_THRESHOLD` | — | 3（新增） | 连续 3 帧不同手势才解锁，防噪声误断 |
| fps 计算 | 每 15 帧 | 每秒按时间 | 更均匀 |

### 动作冷却（`p3_main.py`）

新增 `ACTION_COOLDOWN = 1.0` 秒，防手势闪回导致重复触发。

### 进度条每帧更新（`p3_gesture_worker.py`）

`lock_progress` 从检测帧计算改为每帧实时计算，进度条动画更平滑。

---

## 模型体系

### 当前 6 个模型

| 下拉选项 | 文件 | 特征模式 | 维度 | 说明 |
|----------|------|---------|------|------|
| `KNN` / `SVM` | `hand_gesture_knn/svm.pkl` | raw | 63 | 相机坐标减手腕 |
| `KNN_HAND` / `SVM_HAND` | `hand_gesture_knn/svm_hand.pkl` | hand | 63 | 2D 旋转对齐 |
| `KNN_HAND_V2` / `SVM_HAND_V2` | `hand_gesture_knn/svm_hand_v2.pkl` | hand_v2 | 63 | 3D 手掌坐标系 |

### hand_v2（3D 手掌坐标系）

用 4 个掌骨点构造 3D 正交基：

```
Y = wrist(0) → middle_mcp(9)      手掌长度方向
X = index_mcp(5) → pinky_mcp(17)  手掌宽度方向（Gram-Schmidt 正交化）
Z = X × Y                          手掌法向量
```

投影 21 点到局部坐标系，除以 `||Y||` 归一化。

#### 已知问题

- **掌心/手背翻转**：3D 坐标架已解决该问题，掌心 3 与手背 3 特征接近
- **0、7 易与 6 混淆**（未解决）：3D 投影弱化了手指张开程度信息

---

## CSV 数据集

| 文件 | 维度 | 说明 |
|------|------|------|
| `csv/hand_gesture_data.csv` | 63 | raw 特征 |
| `csv/hand_gesture_data_hand.csv` | 63 | hand 特征 |
| `csv/hand_gesture_data_hand_v2.csv` | 63 | hand_v2 特征 |

### 训练数据分布

`train/0` ~ `train/8` 共 9 类，每类约 124 张图片。

---

## 文件结构

```
Exp_Final/
├── train/0..8/           # 训练图像
├── csv/                  # 特征 CSV（3 个）
├── models/               # 模型文件（6 个 .pkl + hand_landmarker.task）
│
├── p1_extract_features.py   # 特征提取
├── p2_train_knn.py          # KNN/SVM 训练
├── p3_main.py               # 主界面（PySide6）
├── p3_gesture_worker.py     # 摄像头推理线程
├── p3_actions.py            # Windows API 操作映射
├── p3_hand_tracker.py       # MediaPipe 手部检测
├── p3_gesture_classifier.py # 分类器封装
└── README.md
```
