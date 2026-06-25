# 变更日志 — 2026-06-15

## 架构重构

### GUI 分离
- **`p3_gui.py`** 新建：提取 `OverlayWindow` + `MainWindow` 全部 UI 代码
- **`p4_main.py`** 新建：极简入口，仅含导入与启动逻辑（16 行）
- **`p3_main.py`** 删除：原完整文件拆分后移除
- 配置常量 `FUNCTIONS`、`ACTION_COOLDOWN` 移入 `p3_gui.py`

### 当前文件结构
```
Exp_Final/
├── p4_main.py               # 入口（极简）
├── p3_gui.py                # GUI 界面（OverlayWindow + MainWindow）
├── p3_gesture_worker.py     # 摄像头推理线程
├── p3_gesture_classifier.py # 分类器封装
├── p3_hand_tracker.py       # MediaPipe 手部检测
├── p3_actions.py            # Windows API 操作映射
├── p2_train_knn.py          # KNN/SVM 训练
├── p1_extract_features.py   # 特征提取
├── CHANGELOG.md             # 本文件
├── README.md / PROJECT.md
├── train/ csv/ models/
```

## 稳定性修复

| 问题 | 修复 |
|------|------|
| `QThread: Destroyed while running` | `_stop()` 加 `wait(3000)` 安全等待，去掉 `terminate()`，确保 `cap.release()` 执行 |
| MediaPipe 警告刷屏 | `os.environ['GLOG_minloglevel'] = '2'` 屏蔽 C++ 日志 |
| 摄像头 90ms 卡顿 | 保留 `CAP_DSHOW`，保持稳定性 |

## UI 改进

### 悬浮窗（OverlayWindow）
- 固定宽度 180px，高度由内容自动包裹后冻结（`setFixedSize(sizeHint())`）
- 不再随内容变化变形
- 布局：绿色大字 → 说明 → 横线 → 锁定状态 → 操作列表
- `addStretch(1)` 将内容顶到上方，消除多余空白
- 内边距 12px，组件间距 4px，紧凑排版

### 主页面
- 状态与操作面板：`spacing=0` + `padding/margin:0` + HTML 紧排版
- 移除底部状态栏模型说明（悬浮窗已包含）

### 主页面状态与操作面板
- 采用 HTML `<p style='line-height:1.3'>` 消除行间距
- 所有 QLabel `padding:0; margin:0;`
- `back_hint` 功能合并入 `action_list`（原独立显示"0 返回主菜单"）

## 模型改进

- `p2_train_knn.py`：所有 KNN/SVM 模型加入 `StandardScaler` Pipeline，消除特征尺度偏差
- `p3_gesture_worker.py`：预热图从 JPG 文件改为 `np.zeros((240,320,3))` 假图，不依赖外部文件

## 控制台静默

- 移除 `[profiler]` 每 30 帧的耗时打印
- 移除 `[camera]` 摄像头信息打印
