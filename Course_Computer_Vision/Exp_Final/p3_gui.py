# gui.py
"""GUI：悬浮状态窗 + 主窗口"""
import time, threading, numpy as np
from PySide6.QtCore import Qt, Slot, QPoint
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QComboBox, QProgressBar, QCheckBox,
    QFrame,
)
from p3_gesture_classifier import GestureClassifier
from p3_gesture_worker import GestureWorker
from p3_actions import ACTION_MAP, execute as execute_action

ACTION_COOLDOWN = 1.0

FUNC_INFO = {
    1: ("PPT 控制", "幻灯片翻页"),
    2: ("媒体播放", "音乐/视频播放"),
    3: ("窗口管理", "窗口切换控制"),
    4: ("网页浏览", "浏览器控制"),
    5: ("系统控制", "锁屏/截图等"),
    6: ("文件操作", "新建/复制/粘贴/删除/重命名"),
    7: ("输入辅助", "全选/撤销/保存/查找/切换输入法"),
    8: ("鼠标控制", "左/右键/双击/滚轮"),
}

FUNCTIONS = {
    gid: {"name": name, "desc": desc, "sub": ACTION_MAP.get(gid, {})}
    for gid, (name, desc) in FUNC_INFO.items()
}


# ====================================================================
# 悬浮状态窗（右上角，始终置顶）
# ====================================================================
class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowDoesNotAcceptFocus |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.setFixedWidth(190)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget()
        self.container.setStyleSheet(
            "background: rgba(255,255,255,245); "
            "border: 1px solid #ccc; "
            "border-radius: 6px;"
        )
        cl = QVBoxLayout(self.container)
        cl.setContentsMargins(10, 4, 10, 4)
        cl.setSpacing(2)

        # 1. 绿色状态字体
        self.status_label = QLabel("等待手势")
        self.status_label.setStyleSheet(
            "color: #00c8a0; font-size: 16px; font-weight: bold; "
            "border: none; background: transparent; padding: 0; margin: 0;")
        cl.addWidget(self.status_label)

        # 2. 手势：标签
        self.mode_label = QLabel("手势：")
        self.mode_label.setStyleSheet(
            "color: #555; font-size: 11px; "
            "border: none; background: transparent; padding: 0; margin: 0;")
        cl.addWidget(self.mode_label)

        # 3. 分隔线（始终显示）
        self.sep_line = QFrame()
        self.sep_line.setFrameShape(QFrame.Shape.HLine)
        self.sep_line.setStyleSheet("background-color: #dcdcdc;")
        self.sep_line.setFixedHeight(1)
        cl.addWidget(self.sep_line)

        # 4. 可选手势列表
        self.gesture_list = QLabel("")
        self.gesture_list.setStyleSheet(
            "color: #333; font-size: 11px; "
            "border: none; background: transparent; padding: 0; margin: 0;")
        self.gesture_list.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        cl.addWidget(self.gesture_list)

        # 5. 锁定状态行（默认隐藏）
        self.lock_status = QLabel("")
        self.lock_status.setStyleSheet(
            "color: #cc8800; font-size: 10px; "
            "border: none; background: transparent; padding: 0; margin: 0;")
        self.lock_status.setVisible(False)
        cl.addWidget(self.lock_status)

        # 内容顶到上方
        cl.addStretch(1)

        main_layout.addWidget(self.container)

    def set_home(self):
        self.status_label.setText("首页")
        self.mode_label.setText("手势：")
        self.lock_status.setVisible(False)
        lines = [f"{gid} {fdef['name']}" for gid, fdef in FUNCTIONS.items()]
        self.gesture_list.setText("<p style='margin:0;padding:0;line-height:1.2'>" + "<br>".join(lines) + "</p>")

    def set_function(self, gid):
        fdef = FUNCTIONS.get(gid)
        if not fdef:
            return
        self.status_label.setText(fdef['name'])
        self.mode_label.setText("手势：")
        self.lock_status.setVisible(False)
        lines = [f"{sgid} {pair[0]}" for sgid, pair in fdef["sub"].items()]
        lines.append("0 返回主菜单")
        self.gesture_list.setText("<p style='margin:0;padding:0;line-height:1.2'>" + "<br>".join(lines) + "</p>")

    def set_executing(self, action_name):
        self.mode_label.setText(f"执行: {action_name}")
        self.lock_status.setVisible(False)

    def set_lock_status(self, pred, lock_progress, locked):
        self.lock_status.setVisible(True)
        if locked:
            self.lock_status.setText(f"✓ 手势 {pred} 已触发")
            self.lock_status.setStyleSheet(
                "color: #008844; font-size: 10px; border: none; background: transparent; font-weight: bold;")
        elif lock_progress > 0:
            self.lock_status.setText(f"手势 {pred} 锁定中 {int(lock_progress*100)}%")
            self.lock_status.setStyleSheet(
                "color: #cc8800; font-size: 10px; border: none; background: transparent; font-weight: bold;")
        elif pred is not None:
            self.lock_status.setText(f"手势 {pred}")
            self.lock_status.setStyleSheet(
                "color: #888; font-size: 10px; border: none; background: transparent;")
        else:
            self.lock_status.setVisible(False)


# ====================================================================
# 主窗口
# ====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手部识别控制系统")
        self.setMinimumSize(960, 540)
        self.resize(1020, 600)

        self.classifier = GestureClassifier()
        self.worker = None

        # 状态机
        self.state = "home"       # "home" | "function"
        self.current_func = None  # 当前功能编号
        self._last_locked = None  # 上一次锁定的手势（防重复触发）
        self._last_action_time = 0.0  # 上一次动作执行时间（配合冷却）

        self._init_ui()
        self._init_overlay()

        # 启动即开启摄像头
        self._start()

    def _init_overlay(self):
        """右上角悬浮窗"""
        self.overlay = OverlayWindow()
        self.overlay.set_home()
        self.overlay.show()
        self.overlay.raise_()

    def _position_overlay(self):
        """将悬浮窗放在当前屏幕右上角（支持多显示器）"""
        screen = self.screen().availableGeometry()
        ox = screen.right() - self.overlay.width() - 20
        oy = screen.top() + 20
        self.overlay.move(QPoint(ox, oy))
        self.overlay.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        self._position_overlay()
        self.overlay.raise_()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._position_overlay()

    # ==================================================================
    # UI
    # ==================================================================
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        title = QLabel("手势识别控制系统  |  MediaPipe 21点 + KNN/SVM")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title.setStyleSheet("color: #1a365d;")
        root.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(6)

        # ---- 左：摄像头 ----
        left = QVBoxLayout()
        left.setSpacing(0)
        self.video_label = QLabel("正在启动摄像头...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background: #111; color: #555; border: 2px solid #333; font-size: 15px;")
        left.addWidget(self.video_label, stretch=1)
        body.addLayout(left, stretch=1)

        # ---- 右：面板 ----
        right = QVBoxLayout()
        right.setSpacing(6)

        # ── 1. 识别（锁定进度条 + 手势/置信度） ──
        recog_group = QGroupBox("识别")
        rg = QVBoxLayout(recog_group)
        rg.setSpacing(4)

        self.lock_bar = QProgressBar()
        self.lock_bar.setRange(0, 100)
        self.lock_bar.setValue(0)
        self.lock_bar.setFixedHeight(18)
        self.lock_bar.setFormat("")
        self.lock_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #bbb; border-radius: 3px; text-align: center; font-size: 9px; } "
            "QProgressBar::chunk { background: #ffcc00; border-radius: 2px; }")
        rg.addWidget(self.lock_bar)

        # 手势 + 置信度（左右各一半，内容靠左，数值贴文字）
        num_row = QHBoxLayout()
        num_row.setSpacing(0)

        left_w = QWidget()
        left_lay = QHBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)
        left_lay.addWidget(QLabel("手势"))
        self.gesture_num = QLabel("-")
        self.gesture_num.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.gesture_num.setStyleSheet("color: #00c8a0;")
        left_lay.addWidget(self.gesture_num)
        left_lay.addStretch()
        num_row.addWidget(left_w, stretch=4)

        right_w = QWidget()
        right_lay = QHBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(QLabel("置信度"))
        self.conf_pct = QLabel("--")
        self.conf_pct.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.conf_pct.setStyleSheet("color: #555;")
        right_lay.addWidget(self.conf_pct)
        right_lay.addStretch()
        num_row.addWidget(right_w, stretch=6)
        rg.addLayout(num_row)

        # 检测耗时
        det_row = QHBoxLayout()
        det_row.setSpacing(8)
        det_row.addWidget(QLabel("检测耗时"))
        self.detect_ms = QLabel("--")
        self.detect_ms.setStyleSheet("color: #777; font-size: 11px;")
        det_row.addWidget(self.detect_ms)
        det_row.addStretch()
        rg.addLayout(det_row)

        # 分类耗时
        cls_row = QHBoxLayout()
        cls_row.setSpacing(8)
        cls_row.addWidget(QLabel("分类耗时"))
        self.classify_ms = QLabel("--")
        self.classify_ms.setStyleSheet("color: #777; font-size: 11px;")
        cls_row.addWidget(self.classify_ms)
        cls_row.addStretch()
        rg.addLayout(cls_row)

        right.addWidget(recog_group, stretch=1)

        # ── 2. 状态与操作（紧凑排列：标题→分隔线→列表，无多余空隙）──
        self.state_group = QGroupBox("状态与操作")
        sg = QVBoxLayout(self.state_group)
        sg.setSpacing(2)
        sg.setContentsMargins(8, 4, 6, 4)

        self.state_title = QLabel("首页")
        self.state_title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.state_title.setStyleSheet("color: #00c8a0; padding: 0; margin: 0;")
        sg.addWidget(self.state_title)

        self.state_desc = QLabel("选择功能")
        self.state_desc.setStyleSheet("color: #888; font-size: 10px; padding: 0; margin: 0;")
        sg.addWidget(self.state_desc)

        # 分隔线
        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.HLine)
        sep_line.setStyleSheet("background-color: #dcdcdc;")
        sep_line.setFixedHeight(1)
        sg.addWidget(sep_line)

        self.action_list = QLabel("")
        self.action_list.setStyleSheet("font-size: 11px; padding: 0; margin: 0;")
        self.action_list.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        sg.addWidget(self.action_list)

        self.back_hint = QLabel("")
        self.back_hint.setStyleSheet("color: #ff8844; font-size: 10px; padding: 0; margin: 0;")
        sg.addWidget(self.back_hint)
        # 内容顶到上方，不留空隙
        sg.addStretch(1)
        right.addWidget(self.state_group, stretch=2)

        # ── 3. 设置 ──
        settings_group = QGroupBox("设置")
        sl = QVBoxLayout(settings_group)
        sl.setSpacing(4)
        mr = QHBoxLayout()
        mr.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.classifier.display_names)
        self.model_combo.currentTextChanged.connect(self._on_model)
        mr.addWidget(self.model_combo)
        mr.addStretch()
        sl.addLayout(mr)
        self.chk_skeleton = QCheckBox("显示骨架")
        self.chk_skeleton.setChecked(True)
        self.chk_skeleton.stateChanged.connect(self._on_opt)
        sl.addWidget(self.chk_skeleton)
        self.chk_lockbar = QCheckBox("显示锁定条")
        self.chk_lockbar.setChecked(True)
        self.chk_lockbar.stateChanged.connect(self._on_opt)
        sl.addWidget(self.chk_lockbar)
        right.addWidget(settings_group, stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(240)
        body.addWidget(right_widget)
        root.addLayout(body, stretch=1)

        # 初始化状态卡内容
        self._update_state_card()

    # ==================================================================
    def _update_state_card(self):
        """根据当前状态更新状态功能卡"""
        if self.state == "home":
            self.state_title.setText("首页")
            self.state_desc.setText("选择一个功能进入")
            lines = [f"{gid} {fdef['name']}" for gid, fdef in FUNCTIONS.items()]
            self.action_list.setText("<p style='margin:0;padding:0;line-height:1.3'>" + "<br>".join(lines) + "</p>")
            self.back_hint.setText("")
        else:
            fdef = FUNCTIONS.get(self.current_func)
            if fdef:
                self.state_title.setText(fdef['name'])
                self.state_desc.setText(fdef['desc'])
                lines = [f"{sgid} {pair[0]}" for sgid, pair in fdef["sub"].items()]
                lines.append("0 返回主菜单")
                self.action_list.setText("<p style='margin:0;padding:0;line-height:1.3'>" + "<br>".join(lines) + "</p>")
                self.back_hint.setText("")

    # ==================================================================
    def _start(self):
        self.worker = GestureWorker(classifier=self.classifier)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()
        # 后台预加载全部模型，切换零等待
        threading.Thread(target=self.classifier.preload_all, daemon=True).start()

    def _stop(self):
        if hasattr(self, 'worker') and self.worker:
            self.worker._running = False
            self.worker.quit()
            if not self.worker.wait(3000):
                self.worker.terminate()
                self.worker.wait()
            self.worker = None
        self.video_label.clear()
        self.video_label.setText("摄像头已关闭")
        self.gesture_num.setText("-")
        self.conf_pct.setText("--")
        self.lock_bar.setValue(0)
        self.lock_bar.setFormat("")
        self.lock_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #bbb; border-radius: 3px; text-align: center; font-size: 9px; } "
            "QProgressBar::chunk { background: #ffcc00; border-radius: 2px; }")
        self.state = "home"
        self.current_func = None
        self._last_locked = None
        self._update_state_card()
        self.overlay.set_home()

    @Slot(np.ndarray)
    def _on_frame(self, frame):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(dict)
    def _on_result(self, data):
        pred = data.get("pred")
        locked = data.get("locked", False)
        lock_progress = data.get("lock_progress", 0.0)
        confidence = data.get("confidence", 0.0)

        # ---- 锁定进度条 ----
        self.lock_bar.setValue(int(lock_progress * 100))
        if locked:
            self.lock_bar.setFormat(f"✓ 手势 {pred}")
        elif lock_progress > 0:
            self.lock_bar.setFormat(f"{int(lock_progress*100)}%")
        else:
            self.lock_bar.setFormat("")

        # ---- 手势数字 + 置信度 ----
        if pred is not None and confidence > 0:
            self.gesture_num.setText(str(pred))
            self.gesture_num.setStyleSheet("color: #00c8a0;")
            self.conf_pct.setText(f"{confidence:.1%}")
            self.conf_pct.setStyleSheet("color: #555;")
        else:
            self.gesture_num.setText("-")
            self.gesture_num.setStyleSheet("color: #00c8a0;")
            self.conf_pct.setText("--")
            self.conf_pct.setStyleSheet("color: #555;")

        # ---- 检测/分类耗时 ----
        detect_ms = data.get("detect_ms", 0)
        classify_ms = data.get("classify_ms", 0)
        self.detect_ms.setText(f"{detect_ms} ms" if detect_ms else "--")
        self.classify_ms.setText(f"{classify_ms} ms" if classify_ms else "--")

        # ---- 更新悬浮窗锁定状态 ----
        self.overlay.set_lock_status(pred, lock_progress, locked)

        # ---- 锁定触发（含冷却期）----
        if locked and pred is not None and pred != self._last_locked:
            now = time.time()
            if now - self._last_action_time >= ACTION_COOLDOWN:
                self._last_action_time = now
                self._last_locked = pred
                self._handle_gesture(pred)
        elif not locked:
            self._last_locked = None

    def _handle_gesture(self, gid):
        if self.state == "home":
            if gid == 0:
                pass
            elif gid in FUNCTIONS:
                self.state = "function"
                self.current_func = gid
                self.overlay.set_function(gid)
                self._update_state_card()
                self._last_action_time = time.time()
        elif self.state == "function":
            if gid == 0:
                self.state = "home"
                self.current_func = None
                self.overlay.set_home()
                self._update_state_card()
                self._last_action_time = time.time()
            else:
                # 执行功能内子手势
                name = execute_action(self.current_func, gid)
                if name:
                    self.overlay.set_executing(name)

    def _on_model(self, display):
        key = self.classifier.key_from_display(display)
        if key:
            self.classifier.switch(key)

    def _on_opt(self):
        if self.worker:
            self.worker.show_skeleton = self.chk_skeleton.isChecked()
            self.worker.show_lock_bar = self.chk_lockbar.isChecked()

    def closeEvent(self, event):
        self.overlay.close()
        self._stop()
        event.accept()
