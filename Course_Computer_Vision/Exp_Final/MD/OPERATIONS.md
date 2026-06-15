# 手势识别控制系统 — 操作手册

## 一、快速启动

```powershell
cd C:\A_GitCode\PyTorch_CV_YOLO\Course_Computer_Vision\Exp_Final
python p4_main.py
```

GUI 启动后默认使用 `KNN_HAND` 模型（按字母序第一个）。在右侧「设置」下拉框可切换模型。

---

## 二、完整工作流

### 2.1 环境准备

```powershell
pip install PySide6 mediapipe opencv-python numpy scikit-learn pandas
# 可选（音量控制需要）
pip install pycaw comtypes
```

### 2.2 特征提取

从 `train/` 目录的训练图片中提取手部关键点特征，生成 4 个 CSV：

```powershell
python p1_extract_features.py
```

输出：
| CSV 文件 | 特征模式 | 维度 | 说明 |
|----------|---------|:----:|------|
| `csv/hand_gesture_data.csv` | raw | 63 | 原始坐标 |
| `csv/hand_gesture_data_hand.csv` | hand | 42 | 2D旋转对齐 |
| `csv/hand_gesture_data_hand_v2.csv` | hand_v2 | 42 | 2D正交基 |
| `csv/hand_gesture_data_dist.csv` | dist | 210 | 距离特征 |

提取过程会：
- 对每张图用 MediaPipe 检测 21 个手部关键点
- 按 4 种归一化方式生成特征向量
- **对 raw/hand/hand_v2**：额外做水平翻转生成镜像数据（训练集翻倍）
- **对 dist**：不做镜像（距离特征已几何不变，不需要）

### 2.3 模型训练

```powershell
python p2_train_knn.py
```

训练 8 个模型：
| 模型文件 | 特征 | 分类器 |
|---------|------|--------|
| `hand_gesture_knn.pkl` | raw | KNN k=5 |
| `hand_gesture_svm.pkl` | raw | SVM RBF |
| `hand_gesture_knn_hand.pkl` | hand | KNN k=5 |
| `hand_gesture_svm_hand.pkl` | hand | SVM RBF |
| `hand_gesture_knn_hand_v2.pkl` | hand_v2 | KNN k=5 |
| `hand_gesture_svm_hand_v2.pkl` | hand_v2 | SVM RBF |
| `hand_gesture_knn_dist.pkl` | dist | KNN k=5 |
| `hand_gesture_svm_dist.pkl` | dist | SVM RBF |

每个模型会输出：
- 测试集准确率（留出法 80/20 分割）
- 5 折交叉验证准确率
- 各类别召回率/精确率
- 混淆矩阵

### 2.4 启动 GUI

```powershell
python p4_main.py
```

---

## 三、项目文件结构

```
Exp_Final/
├── p4_main.py                 # 入口（极简）
├── p3_gui.py                  # GUI 界面（悬浮窗 + 主窗口）
├── p3_gesture_worker.py       # 摄像头推理线程
├── p3_gesture_classifier.py   # 分类器（线程安全 + 自动扫描模型）
├── p3_hand_tracker.py         # MediaPipe 手部检测
├── p3_actions.py              # Windows API 键盘/鼠标模拟
├── p1_extract_features.py     # 特征提取（训练用）
├── p2_train_knn.py            # 模型训练（训练用）
│
├── train/0..8/                # 训练图片（9 类手势）
├── csv/                       # 特征 CSV 文件
├── models/                    # .pkl 模型文件
│
├── README.md                  # 项目概述
├── MODELS.md                  # 模型原理说明
├── CHANGELOG.md               # 变更日志
├── PROJECT.md                 # 开发记录
└── 本文件                     # 操作手册
```

---

## 四、模型切换与接口

所有模型共享同一接口，通过 `GestureClassifier` 自动管理：

```python
from p3_gesture_classifier import GestureClassifier

clf = GestureClassifier()
# 自动扫描 models/ 下所有 .pkl
# 默认选择字母序第一个（KNN_HAND）

clf.predict(landmarks)          # → (类别, 置信度数组)
clf.switch("hand_gesture_knn_dist")  # 切换模型
clf.current                     # 当前模型 key
clf.available                   # 所有可用模型
```

命名规则（自动识别特征模式）：
- 文件名含 `_dist` → 距离特征（210 维）
- 文件名含 `_hand_v2` → 2D 正交基（42 维）
- 文件名含 `_hand` → 2D 旋转对齐（42 维）
- 其他 → 原始坐标（63 维）

---

## 五、四种特征模式对比

| 模式 | 维度 | 核心原理 | 旋转不变 | 翻转不变 |
|:----:|:----:|---------|:--------:|:--------:|
| raw | 63 | 手腕为原点，保留 xyz | ❌ | ❌ |
| hand | 42 | 旋转让中指根指向上方，丢 z | ✅ | ⚠️ |
| hand_v2 | 42 | 掌骨构建正交基，丢 z | ✅ | ⚠️ |
| dist | 210 | 全部成对点距离 | ✅ | ✅ |

**Dist 模式**用 21 个点之间 210 个欧氏距离作为特征。距离不随旋转、平移、翻转改变，KNN 看到的是纯粹的几何形状。这是解决手背翻转问题的最佳方案。

---

## 六、关键参数

| 参数 | 文件 | 默认值 | 说明 |
|------|------|:------:|------|
| `LOCK_DURATION` | `p3_gesture_worker.py` | 0.9s | 手势锁定等待时间 |
| `SMOOTH_WINDOW` | `p3_gesture_worker.py` | 7 | 平滑投票帧数 |
| `MIN_CONFIDENCE` | `p3_gesture_worker.py` | 0.25 | 最低置信度阈值 |
| `UNLOCK_THRESHOLD` | `p3_gesture_worker.py` | 3 | 连续不同手势解锁帧数 |
| `FRAME_SKIP` | `p3_gesture_worker.py` | 2 | 隔帧检测数 |
| `ACTION_COOLDOWN` | `p3_gui.py` | 1.0s | 动作触发冷却时间 |

---

## 七、常见问题

### Q: 摄像头打不开？
检查摄像头是否被其他程序占用。第一次运行 Windows 会弹出摄像头权限提示。

### Q: 模型切换后没生效？
切换模型后，画面右上角会显示当前模型名。确认显示的名称和下拉框一致。

### Q: 手背朝向识别错误？
切换到 **KNN_DIST** 或 **SVM_DIST** 模型。距离特征不受手心/手背翻转影响。

### Q: 检测速度慢？
MediaPipe 使用 320×240 内部推理分辨率，隔帧检测。正常 FPS 应在 20-30。

### Q: 退出卡顿？
已处理：`wait(3000)` 超时后 `terminate()` 兜底，不会卡死。
