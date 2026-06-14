# main.py
"""手势识别控制系统 — 两级手势：首页选功能 → 功能内子手势 → 0 回首页"""
import sys, numpy as np
from PySide6.QtCore import Qt, Slot, QPoint
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QComboBox, QProgressBar, QCheckBox,
    QFrame,
)
from gesture_classifier import GestureClassifier
from gesture_worker import GestureWorker
from actions import ACTION_MAP, execute as execute_action

# 功能名称+描述（子手势定义在 actions.ACTION_MAP 里）
FUNC_INFO = {
    1: ("PPT 控制", "幻灯片翻页"),
    2: ("媒体播放", "音乐/视频播放"),
    3: ("窗口管理", "窗口切换控制"),
    4: ("网页浏览", "浏览器控制"),
    5: ("系统控制", "锁屏/截图等"),
    6: ("文件操作", "新建/复制/粘贴/删除/重命名"),
    7: ("输入辅助", "全选/撤销/保存/查找/切换输入法"),
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
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 内容容器
        self.container = QWidget()
        self.container.setStyleSheet(
            "background: rgba(20,20,30,220); border: 1px solid rgba(0,200,160,100); "
            "border-radius: 8px; padding: 6px 8px;"
        )
        cl = QVBoxLayout(self.container)
        cl.setSpacing(2)

        # 1. 手势锁定状态（最顶，演示显眼）
        self.lock_status = QLabel("")
        self.lock_status.setStyleSheet("color: #ffcc00; font-size: 11px; border: none; background: transparent; font-weight: bold;")
        self.lock_status.setWordWrap(True)
        cl.addWidget(self.lock_status)

        # 2. 功能标题
        self.mode_label = QLabel("首页")
        self.mode_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.mode_label.setStyleSheet("color: #00c8a0; border: none; background: transparent;")
        cl.addWidget(self.mode_label)

        # 状态描述
        self.status_label = QLabel("等待手势...")
        self.status_label.setStyleSheet("color: #ccc; font-size: 11px; border: none; background: transparent;")
        self.status_label.setWordWrap(True)
        cl.addWidget(self.status_label)

        # 3. 手势操作表
        self.gesture_list = QLabel("")
        self.gesture_list.setStyleSheet(
            "color: #aaa; font-size: 10px; border: none; background: transparent;"
        )
        self.gesture_list.setWordWrap(True)
        cl.addWidget(self.gesture_list)

        layout.addWidget(self.container)
        self.adjustSize()

    def set_home(self):
        self.mode_label.setText("首页")
        self.status_label.setText("选择功能")
        self.lock_status.setText("")
        tips = ""
        for gid, fdef in FUNCTIONS.items():
            tips += f"{gid} {fdef['name']}\n"
        tips += "0 主菜单"
        self.gesture_list.setText(tips.strip())
        self.adjustSize()

    def set_function(self, gid):
        fdef = FUNCTIONS.get(gid)
        if not fdef:
            return
        self.mode_label.setText(fdef['name'])
        self.status_label.setText(fdef['desc'])
        self.lock_status.setText("")
        tips = ""
        for sgid, pair in fdef["sub"].items():
            sname = pair[0]
            tips += f"{sgid} {sname}\n"
        tips += "0 返回主菜单"
        self.gesture_list.setText(tips.strip())
        self.adjustSize()

    def set_executing(self, action_name):
        self.status_label.setText(f"执行: {action_name}")
        self.adjustSize()

    def set_lock_status(self, pred, lock_progress, locked):
        """更新锁定状态行（演示时显示当前手势和进度）"""
        if locked:
            self.lock_status.setText(f"✓ 手势 {pred} 已触发")
            self.lock_status.setStyleSheet(
                "color: #00ff88; font-size: 11px; border: none; background: transparent; font-weight: bold;")
        elif lock_progress > 0:
            blocks = int(lock_progress * 8)
            bar = "█" * blocks + "░" * (8 - blocks)
            self.lock_status.setText(f"手势 {pred}  {bar} {int(lock_progress*100)}%")
            self.lock_status.setStyleSheet(
                "color: #ffcc00; font-size: 11px; border: none; background: transparent; font-weight: bold;")
        elif pred is not None:
            self.lock_status.setText(f"手势 {pred}")
            self.lock_status.setStyleSheet(
                "color: #aaa; font-size: 11px; border: none; background: transparent;")
        else:
            self.lock_status.setText("")
        self.adjustSize()


# ====================================================================
# 主窗口
# ====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别控制系统 — MediaPipe + KNN")
        self.setMinimumSize(960, 540)
        self.resize(1020, 600)

        self.classifier = GestureClassifier()
        self.worker = None

        # 状态机
        self.state = "home"       # "home" | "function"
        self.current_func = None  # 当前功能编号
        self._last_locked = None  # 上一次锁定的手势（防重复触发）

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
        """将悬浮窗放在屏幕右上角"""
        screen = QApplication.primaryScreen().availableGeometry()
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

        title = QLabel("手势识别控制系统  |  MediaPipe 21点 + KNN  |  粤台产业科技学院")
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

        # 手势 + 置信度（紧凑一行）
        num_row = QHBoxLayout()
        num_row.setSpacing(8)
        num_row.addWidget(QLabel("手势"))
        self.gesture_num = QLabel("-")
        self.gesture_num.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.gesture_num.setStyleSheet("color: #00c8a0;")
        num_row.addWidget(self.gesture_num)
        num_row.addStretch()
        num_row.addWidget(QLabel("置信度"))
        self.conf_pct = QLabel("--")
        self.conf_pct.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.conf_pct.setStyleSheet("color: #555;")
        num_row.addWidget(self.conf_pct)
        rg.addLayout(num_row)
        right.addWidget(recog_group)

        # ── 2. 状态与操作 ──
        state_group = QGroupBox("状态与操作")
        sg = QVBoxLayout(state_group)
        sg.setSpacing(4)
        sg.setContentsMargins(8, 6, 6, 6)  # 左8px统一缩进

        self.state_title = QLabel("首页")
        self.state_title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.state_title.setStyleSheet("color: #00c8a0;")
        sg.addWidget(self.state_title)

        # 分隔线
        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.HLine)
        sep_line.setFixedHeight(1)
        sg.addWidget(sep_line)

        self.action_list = QLabel("")
        self.action_list.setStyleSheet("font-size: 11px;")
        self.action_list.setWordWrap(True)
        sg.addWidget(self.action_list, stretch=1)

        self.back_hint = QLabel("")
        self.back_hint.setStyleSheet("color: #ff8844; font-size: 10px;")
        sg.addWidget(self.back_hint)
        right.addWidget(state_group, stretch=1)

        # ── 3. 设置 ──
        settings_group = QGroupBox("设置")
        sl = QVBoxLayout(settings_group)
        sl.setSpacing(4)
        mr = QHBoxLayout()
        mr.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([m.upper() for m in self.classifier.available])
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
        right.addWidget(settings_group)

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
            tips = ""
            for gid, fdef in FUNCTIONS.items():
                tips += f"{gid}    {fdef['name']}\n"
            self.action_list.setText(tips.strip())
            self.back_hint.setText("")
        else:
            fdef = FUNCTIONS.get(self.current_func)
            if fdef:
                self.state_title.setText(fdef['name'])
                tips = ""
                for sgid, pair in fdef["sub"].items():
                    tips += f"{sgid}    {pair[0]}\n"
                self.action_list.setText(tips.strip())
                self.back_hint.setText("0  返回主菜单")

    # ==================================================================
    def _start(self):
        self.worker = GestureWorker()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.result_ready.connect(self._on_result)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop(); self.worker = None
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

        # ---- 更新悬浮窗锁定状态 ----
        self.overlay.set_lock_status(pred, lock_progress, locked)

        # ---- 锁定触发 ----
        if locked and pred is not None and pred != self._last_locked:
            self._last_locked = pred
            self._handle_gesture(pred)
        elif not locked:
            self._last_locked = None

    def _handle_gesture(self, gid):
        """根据当前状态处理手势（0 返回主菜单）"""
        if self.state == "home":
            if gid == 0:
                pass  # 已在主菜单
            elif gid in FUNCTIONS:
                # 进入功能
                self.state = "function"
                self.current_func = gid
                self.overlay.set_function(gid)
                self._update_state_card()
        elif self.state == "function":
            if gid == 0:
                # 返回主菜单
                self.state = "home"
                self.current_func = None
                self.overlay.set_home()
                self._update_state_card()
            else:
                # 执行功能内子手势
                name = execute_action(self.current_func, gid)
                if name:
                    self.overlay.set_executing(name)

    def _on_model(self, name):
        self.classifier.switch(name.lower())

    def _on_opt(self):
        if self.worker:
            self.worker.show_skeleton = self.chk_skeleton.isChecked()
            self.worker.show_lock_bar = self.chk_lockbar.isChecked()

    def closeEvent(self, event):
        self.overlay.close()
        self._stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
