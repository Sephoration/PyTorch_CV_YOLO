# 手势识别控制系统

基于 MediaPipe 21 点手部关键点 + KNN/SVM 分类器的实时手势识别控制系统，通过 Windows API 模拟键盘/鼠标输入，实现对电脑的隔空操控。

---

## 快速开始

```powershell
# 安装依赖
pip install PySide6 mediapipe opencv-python numpy scikit-learn pandas

# 启动 GUI
python p4_main.py
```

默认使用 `KNN_HAND` 模型（字母序第一个），右侧面板下拉框可切换模型。

如需从训练图片重新训练模型：

```powershell
python p1_extract_features.py    # 提取特征 → csv/
python p2_train_knn.py           # 训练模型 → models/
```

---

## 文件结构

```
Exp_Final/
├── p4_main.py                 # 入口（仅 16 行）
├── p3_gui.py                  # GUI 界面（PySide6）
├── p3_gesture_worker.py       # 摄像头推理线程
├── p3_gesture_classifier.py   # 分类器（自动扫描 models/）
├── p3_hand_tracker.py         # MediaPipe 手部检测
├── p3_actions.py              # Windows API 操作映射
├── p1_extract_features.py     # 特征提取（训练用）
├── p2_train_knn.py            # KNN/SVM 训练（训练用）
│
├── train/0..8/                # 9 类手势训练图片
├── csv/                       # 特征 CSV 文件
│   ├── hand_gesture_data.csv         # raw 特征
│   └── hand_gesture_data_*.csv       # hand / hand_v2 / dist
├── models/                    # .pkl 模型文件（共 8 个）
│
├── README.md                  # 本文件
└── 其他 .md                    # 历史文档
```

---

## 8 个功能模块

| 手势 | 功能 | 子操作 |
|:----:|------|--------|
| 1 | PPT 控制 | 下一页 / 上一页 / 开始放映 / 结束放映 |
| 2 | 媒体播放 | 播放暂停 / 下一首 / 上一首 / 音量+ / 音量- |
| 3 | 窗口管理 | 切换窗口 / 最小化 / 关闭 / 分屏左 / 分屏右 |
| 4 | 网页浏览 | 下滚 / 上滚 / 新标签 / 关标签 / 刷新 |
| 5 | 系统控制 | 锁屏 / 截图 / 任务视图 / 显示桌面 |
| 6 | 文件操作 | 新建文件夹 / 复制 / 粘贴 / 删除 / 重命名 |
| 7 | 输入辅助 | 全选 / 撤销 / 保存 / 查找 / 切换输入法 |
| 8 | 鼠标控制 | 左键 / 右键 / 双击 / 滚轮上 / 滚轮下 |
| 0 | 返回 | 从功能内返回主菜单 |

两层菜单：首页手势 1-8 进功能 → 子手势执行操作 → 手势 0 返回。

---

## 8 个模型

| 文件名 | 特征模式 | 维度 | 分类器 |
|--------|---------|:----:|:------:|
| `hand_gesture_knn.pkl` | raw | 63 | KNN k=5 |
| `hand_gesture_svm.pkl` | raw | 63 | SVM RBF |
| `hand_gesture_knn_hand.pkl` | hand | 42 | KNN k=5 |
| `hand_gesture_svm_hand.pkl` | hand | 42 | SVM RBF |
| `hand_gesture_knn_hand_v2.pkl` | hand_v2 | 42 | KNN k=5 |
| `hand_gesture_svm_hand_v2.pkl` | hand_v2 | 42 | SVM RBF |
| `hand_gesture_knn_dist.pkl` | dist | 210 | KNN k=5 |
| `hand_gesture_svm_dist.pkl` | dist | 210 | SVM RBF |

### 四种特征模式的原理

**raw（原始坐标系，63 维）**
以手腕为原点，按手腕到中指根距离缩放。保留 xyz。对旋转和翻转均敏感。

**hand（2D 旋转对齐，42 维）**
计算手腕到中指根的方向角，旋转整个手让中指根指向正上方，然后丢掉 z。对平面旋转不敏感，翻转（手背）时形状成镜像。实测效果最好。

**hand_v2（2D 手掌正交基，42 维）**
用 4 个掌骨点构造正交基：Y = 手腕→中指根，X = 食指根→小指根（Gram-Schmidt 正交化）。丢弃 z。数学上比 hand 更严谨，实测略低于 hand。

**dist（距离特征，210 维）**
计算 21 个点之间全部 210 个成对欧氏距离，按手腕到中指根归一化。平移、旋转、翻转完全不变——手心手背算出来一样。解决手背翻转问题的最佳方案。

### 接口

```python
classifier = GestureClassifier()           # 自动扫描 models/，默认选字母序第一个
classifier.predict(landmarks)              # → (pred, proba)
classifier.switch("hand_gesture_knn_dist") # 切换模型
classifier.current                         # 当前模型 key
classifier.available                       # 所有模型列表
```

系统根据文件名自动识别特征模式：
- 含 `_dist` → 距离（210 维）
- 含 `_hand_v2` → 正交基（42 维）
- 含 `_hand` → 旋转对齐（42 维）
- 其他 → 原始坐标（63 维）

---

## 关键参数

| 参数 | 位置 | 值 | 说明 |
|------|------|:--:|------|
| `LOCK_DURATION` | `gesture_worker.py` | 0.9s | 锁定等待时间 |
| `SMOOTH_WINDOW` | `gesture_worker.py` | 7 帧 | 平滑投票窗口 |
| `MIN_CONFIDENCE` | `gesture_worker.py` | 0.25 | 最低置信度 |
| `UNLOCK_THRESHOLD` | `gesture_worker.py` | 3 帧 | 解锁所需连续不同帧 |
| `FRAME_SKIP` | `gesture_worker.py` | 2 | 每 3 帧检测 1 次 |
| `ACTION_COOLDOWN` | `gui.py` | 1.0s | 动作触发冷却 |
| 检测分辨率 | 内部 | 320×240 | MediaPipe 推理分辨率 |
| 显示分辨率 | 显示 | 640×480 | 画面分辨率 |

---

## 数据流

```
摄像头 → MediaPipe 21点 → 归一化 → KNN/SVM → 平滑投票(7帧)
                                        ↓
              0.9s锁定机制 → 状态机(home/function) → Windows API 操作
```

稳定性机制：
- 置信度 < 0.25 的预测直接丢弃
- 平滑投票：7 帧内取众数
- 锁定：手势稳定 0.9s 才触发，连续 3 帧不同才解锁
- 动作冷却：1.0s 内不重复触发

---

## 性能优化

- 隔帧检测（每 3 帧检测 1 次）
- MediaPipe 推理分辨率 320×240
- 假图预热 MediaPipe + 分类器，消除首次伸手卡顿
- 后台线程预加载全部模型，切换零等待
- 手消失 0.3s 内保持锁定进度

---

## 依赖

| 包 | 用途 | 必选 |
|----|------|:----:|
| PySide6 | GUI | ✅ |
| mediapipe | 手部检测 | ✅ |
| opencv-python | 摄像头/图像 | ✅ |
| numpy | 数值计算 | ✅ |
| scikit-learn | KNN/SVM | ✅ |
| pandas | CSV 读取（训练用） | 仅训练 |
| pycaw | 音量控制 | 可选 |
| comtypes | 音量控制 | 可选 |
