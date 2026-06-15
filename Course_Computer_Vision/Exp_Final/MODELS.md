# 模型体系说明

本项目共 8 个 `.pkl` 模型，分为 4 种特征模式（Feature Mode），每种模式各有 KNN 和 SVM 两个版本。

---

## 一、模型总览

| 模型文件名 | 特征模式 | 特征维度 | 分类器 | 准确率 |
|-----------|---------|:--------:|:-----:|:-----:|
| `hand_gesture_knn.pkl` | raw | 63 | KNN k=5 | — |
| `hand_gesture_svm.pkl` | raw | 63 | SVM RBF | — |
| `hand_gesture_knn_hand.pkl` | hand | 42 | KNN k=5 | — |
| `hand_gesture_svm_hand.pkl` | hand | 42 | SVM RBF | — |
| `hand_gesture_knn_hand_v2.pkl` | hand_v2 | 42 | KNN k=5 | — |
| `hand_gesture_svm_hand_v2.pkl` | hand_v2 | 42 | SVM RBF | — |
| `hand_gesture_knn_dist.pkl` | dist | 210 | KNN k=5 | — |
| `hand_gesture_svm_dist.pkl` | dist | 210 | SVM RBF | — |

> 准确率见 `p2_train_knn.py` 训练输出的测试集结果。

---

## 二、四种特征模式的原理

### 1. Raw — 原始坐标系（63 维）

**归一化方式：**
1. 以手腕点（landmark 0）为原点
2. 按手腕到中指根（landmark 9）的距离缩放
3. 保留 x, y, z 三个坐标

```
raw 特征 = [(lm[i] - wrist) / scale]  for i in 0..20
维度 = 21 点 × 3 坐标 = 63
```

**特点：**
- ❌ 对手的旋转敏感
- ❌ 对手心/手背翻转敏感
- ❌ 对摄像头角度敏感

---

### 2. Hand — 2D 旋转对齐（42 维）

**归一化方式：**
1. 以手腕点（landmark 0）为原点
2. 计算手腕 → 中指根（landmark 9）的方向角
3. 将整个手旋转，让中指根指向正上方
4. 按中指根距离缩放
5. **只保留 x, y 坐标，丢掉 z**

```
angle = arctan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)
rot = -π/2 - angle
每个点：旋转(rot) → 缩放
维度 = 21 点 × 2 坐标 = 42
```

**特点：**
- ✅ 对手的平面旋转不敏感（旋转对齐）
- ✅ 丢掉了不稳定的 z 坐标
- ⚠️ 手背朝向时 2D 形状是掌心的镜像，仍会识别错误
- 🏆 **实测效果最好**

---

### 3. Hand V2 — 2D 手掌正交基（42 维）

**归一化方式：**
1. 用 4 个掌骨点构造 2D 正交基：
   - Y 轴：手腕(0) → 中指根(9)
   - X 轴：食指根(5) → 小指根(17)，对 Y 做 Gram-Schmidt 正交化
2. 将所有点投影到 (X, Y) 基上
3. 按 Y 轴长度缩放
4. **只保留 x, y 坐标**

```
y_axis = middle_mcp(9) - wrist(0)
x_raw = pinky_mcp(17) - index_mcp(5)
x_axis = x_raw - (x_raw·y_axis) * y_axis    # Gram-Schmidt
R = [x_axis, y_axis]
每个点：R.T @ (p - wrist) → 缩放
维度 = 21 点 × 2 坐标 = 42
```

**特点：**
- ✅ 对手的平面旋转不敏感
- ✅ 使用正交基比 Hand 的角度对齐更数学严谨
- ⚠️ 在实测中效果略低于 Hand 模式

---

### 4. Dist — 距离特征（210 维）⭐ 推荐

**归一化方式：**
1. 计算 21 个点之间所有成对欧氏距离（共 210 对）
2. 按手腕到中指根的距离归一化

```
for i in 0..20:
  for j in i+1..20:
    features.append(||lm[i] - lm[j]||)
每个特征 ÷ ||lm[0] - lm[9]||
维度 = 21×20/2 = 210
```

**特点：**
- ✅ **平移不变** — 不依赖任何原点
- ✅ **旋转不变** — 距离不随旋转改变
- ✅ **翻转不变** — 手心/手背的距离完全相同
- ✅ **尺度不变** — 按手的大小归一化
- ✅ KNN 学习的是"几何形状"，而非"坐标值"
- ⚠️ 维度较高（210），需要足够训练数据

---

## 三、分类器原理

### KNN（k=5, distance 加权）

```
Pipeline([
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
])
```

**原理：** 新样本与训练集中最相似的 5 个样本比较，按距离加权投票。距离越近的邻居权重越大。

**特点：**
- 非参数模型，不需要训练过程
- 对数据分布没有假设
- 维度高时（如 dist 210 维）需要足够样本

### SVM（RBF 核, C=10）

```
Pipeline([
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale", probability=True)
])
```

**原理：** 将特征映射到高维空间，寻找最大化类别间隔的超平面。RBF 核可以处理非线性决策边界。

**特点：**
- 对边界样本更鲁棒
- 高维特征下通常比 KNN 更稳定
- probability=True 支持输出置信度

---

## 四、接口说明

所有模型通过 `GestureClassifier` 统一管理，代码无需区分模型类型。

### 自动扫描

```python
classifier = GestureClassifier()
# 自动扫描 models/ 目录下所有 .pkl 文件
# 根据文件名自动推断特征模式（raw/hand/hand_v2/dist）
# 默认使用按字母序第一个模型
```

### 核心接口

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `predict(landmarks)` | MediaPipe 21 个 landmarks | `(pred, proba)` | 预测手势类别 0-8 + 置信度 |
| `switch(name)` | 模型 key（不带 .pkl） | 无 | 切换当前模型 |
| `normalize(landmarks)` | MediaPipe 21 个 landmarks | numpy array | 获取当前模型的特征向量 |
| `key_from_display(display)` | 显示名称（如 "KNN_HAND"） | key | 从下拉框文字找模型 key |

### 属性

| 属性 | 返回 | 说明 |
|------|------|------|
| `available` | `[key1, key2, ...]` | 所有可用模型列表 |
| `display_names` | `["KNN", "SVM", ...]` | 下拉框显示名称 |
| `current` | str | 当前模型 key |
| `meta` | dict | 所有模型的元信息 |

### 特征模式自动识别规则

文件名包含 → `_dist` → 模式 `dist`
文件名包含 → `_hand_v2` → 模式 `hand_v2`
文件名包含 → `_hand` → 模式 `hand`
其他 → 模式 `raw`

### 切换模型

```python
# GUI 下拉框切换
classifier.switch("hand_gesture_knn_dist")

# 或者通过显示名称
key = classifier.key_from_display("KNN_DIST")
classifier.switch(key)
```

---

## 五、训练流程

```powershell
# 1. 从训练图片提取特征（4 种模式同时生成）
python p1_extract_features.py

# 2. 训练全部 8 个模型
python p2_train_knn.py

# 3. 运行 GUI
python p4_main.py
```

每个 CSV 文件对应一种特征模式：

| CSV 文件 | 特征模式 | 维度 |
|----------|---------|:----:|
| `csv/hand_gesture_data.csv` | raw | 63 |
| `csv/hand_gesture_data_hand.csv` | hand | 42 |
| `csv/hand_gesture_data_hand_v2.csv` | hand_v2 | 42 |
| `csv/hand_gesture_data_dist.csv` | dist | 210 |

---

## 六、选择建议

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 掌心朝向、旋转少 | KNN_HAND | 实测最准，维度适中 |
| 需要抗手背翻转 | **KNN_DIST / SVM_DIST** | 距离特征完全几何不变 |
| 追求最高精度 | SVM_HAND_V2 或 SVM_DIST | SVM 对边界更鲁棒 |
| 实时性优先 | KNN 系列 | KNN 预测更快（无核函数计算） |
