import os
import cv2
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTabWidget, QGroupBox, QComboBox,
    QDoubleSpinBox, QSlider, QCheckBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QStatusBar, QFrame, QSizePolicy,
)

from core.tracker import VehicleTracker, PROJECT_ROOT


class VideoWorker(QThread):
    change_pixmap_raw = Signal(QImage)
    change_pixmap_proc = Signal(QImage)
    update_stats = Signal(dict)
    update_count = Signal(dict)
    update_fps = Signal(float)
    frame_processed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracker = VehicleTracker()
        self.cap = None
        self._running = False
        self._paused = False
        self.video_path = None

    def open_video(self, path):
        self.video_path = path
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        return self.cap.isOpened()

    def run(self):
        self._running = True
        while self._running:
            if self._paused:
                self.msleep(30)
                continue
            if self.cap is None or not self.cap.isOpened():
                self.msleep(100)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._running = False
                break

            h, w, ch = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_raw.emit(qimg)

            processed, info = self.tracker.process(frame)

            rgb2 = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            qimg2 = QImage(rgb2.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_proc.emit(qimg2)

            self.update_stats.emit(info)
            self.update_count.emit(self.tracker.counts)
            self.update_fps.emit(self.tracker.fps)
            self.frame_processed.emit(self.tracker.frame_count)

            self.msleep(30)

        if self.cap:
            self.cap.release()
            self.cap = None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        self._paused = False

    @property
    def is_paused(self):
        return self._paused

    @property
    def is_running(self):
        return self._running


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = VideoWorker(self)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.setWindowTitle("东莞理工 - 粤台产业科技学院 - 车辆检测智能监控系统")
        self.resize(1360, 920)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addLayout(self._build_header())
        root.addLayout(self._build_video_area(), 1)
        root.addLayout(self._build_controls())

        self._init_status_bar()

    def _build_header(self):
        layout = QHBoxLayout()

        self.logo_label = QLabel("车辆检测\n监控系统")
        self.logo_label.setFixedSize(150, 150)
        self.logo_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2c3e50, stop:1 #1a252f);
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self.logo_label.setAlignment(Qt.AlignCenter)

        right = QVBoxLayout()

        title = QLabel("东莞理工 - 粤台产业科技学院 - 计算机科学与技术(跨境电商)")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; padding: 4px 8px;")

        subtitle = QLabel("车辆检测智能监控系统  |  YOLO深度学习  |  期末项目")
        subtitle.setStyleSheet("font-size: 12px; color: #7f8c8d; padding: 0 8px 4px 8px;")

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background: #fafafa;
            }
            QTabBar::tab {
                padding: 8px 20px;
                margin: 2px 1px;
                font-size: 13px;
                border: 1px solid #dcdcdc;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                background: #ecf0f1;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #d5dbdb;
            }
        """)

        self.tabs.addTab(self._build_tab_config(), "检测配置")
        self.tabs.addTab(self._build_tab_stats(), "统计信息")
        self.tabs.addTab(self._build_tab_zone(), "区域设置")

        right.addWidget(title)
        right.addWidget(subtitle)
        right.addWidget(self.tabs, 1)

        layout.addWidget(self.logo_label)
        layout.addLayout(right, 1)

        return layout

    def _build_tab_config(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        g1 = QGroupBox("模型与参数")
        g1_layout = QGridLayout(g1)

        g1_layout.addWidget(QLabel("检测模型:"), 0, 0)
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(200)
        self._scan_models()
        g1_layout.addWidget(self.cmb_model, 0, 1, 1, 2)

        g1_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        g1_layout.addWidget(self.spin_conf, 1, 1)

        g1_layout.addWidget(QLabel("计数线位置:"), 2, 0)
        self.slider_count_line = QSlider(Qt.Horizontal)
        self.slider_count_line.setRange(10, 90)
        self.slider_count_line.setValue(60)
        self.label_count_line_val = QLabel("60%")
        g1_layout.addWidget(self.slider_count_line, 2, 1)
        g1_layout.addWidget(self.label_count_line_val, 2, 2)

        g2 = QGroupBox("检测目标")
        g2_layout = QVBoxLayout(g2)

        self.chk_person = QCheckBox("行人 (person)")
        self.chk_car = QCheckBox("小汽车 (car)")
        self.chk_car.setChecked(True)
        self.chk_bus = QCheckBox("公交车 (bus)")
        self.chk_bus.setChecked(True)
        self.chk_truck = QCheckBox("卡车 (truck)")
        self.chk_truck.setChecked(True)

        g2_layout.addWidget(self.chk_person)
        g2_layout.addWidget(self.chk_car)
        g2_layout.addWidget(self.chk_bus)
        g2_layout.addWidget(self.chk_truck)

        layout.addWidget(g1)
        layout.addWidget(g2)
        layout.addStretch()

        return w

    def _build_tab_stats(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        info_label = QLabel("车辆计数统计")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")

        self.count_table = QTableWidget(3, 4)
        self.count_table.setHorizontalHeaderLabels(["类别", "进入 (IN)", "驶出 (OUT)", "总计"])
        self.count_table.setVerticalHeaderLabels(["小汽车", "公交车", "卡车"])
        self.count_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.count_table.setSelectionMode(QTableWidget.NoSelection)
        self.count_table.horizontalHeader().setStretchLastSection(True)
        self.count_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        cls_map = {'car': 0, 'bus': 1, 'truck': 2}
        self._count_table_items = {}
        for cls_name, row in cls_map.items():
            display_name = {'car': '小汽车', 'bus': '公交车', 'truck': '卡车'}[cls_name]
            self.count_table.setItem(row, 0, QTableWidgetItem(display_name))
            for col in range(1, 4):
                item = QTableWidgetItem("0")
                item.setTextAlignment(Qt.AlignCenter)
                self.count_table.setItem(row, col, item)

        self.count_table.setMaximumHeight(140)

        layout.addWidget(info_label)
        layout.addWidget(self.count_table)

        layout.addWidget(QLabel("FPS与帧数"))
        self.label_fps_display = QLabel("FPS: 0.0  |  帧数: 0  |  入侵: 0")
        self.label_fps_display.setStyleSheet("font-size: 13px; padding: 4px; background: #ecf0f1; border-radius: 4px;")
        layout.addWidget(self.label_fps_display)

        layout.addWidget(QLabel("区域入侵记录"))
        self.intrusion_log = QTextEdit()
        self.intrusion_log.setReadOnly(True)
        self.intrusion_log.setMaximumHeight(150)
        self.intrusion_log.setPlaceholderText("入侵记录将显示在这里...")
        layout.addWidget(self.intrusion_log)

        layout.addStretch()

        return w

    def _build_tab_zone(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        g = QGroupBox("敏感区域坐标 (比例 0.0 ~ 1.0)")
        gl = QGridLayout(g)

        gl.addWidget(QLabel("左上:"), 0, 0)
        self.lbl_tl = QLabel("(0.30, 0.35)")
        self.lbl_tl.setStyleSheet("font-family: Consolas; padding: 2px 6px; background: #f0f0f0; border-radius: 3px;")
        gl.addWidget(self.lbl_tl, 0, 1)

        gl.addWidget(QLabel("右上:"), 1, 0)
        self.lbl_tr = QLabel("(0.70, 0.35)")
        self.lbl_tr.setStyleSheet("font-family: Consolas; padding: 2px 6px; background: #f0f0f0; border-radius: 3px;")
        gl.addWidget(self.lbl_tr, 1, 1)

        gl.addWidget(QLabel("右下:"), 2, 0)
        self.lbl_br = QLabel("(0.90, 0.85)")
        self.lbl_br.setStyleSheet("font-family: Consolas; padding: 2px 6px; background: #f0f0f0; border-radius: 3px;")
        gl.addWidget(self.lbl_br, 2, 1)

        gl.addWidget(QLabel("左下:"), 3, 0)
        self.lbl_bl = QLabel("(0.10, 0.85)")
        self.lbl_bl.setStyleSheet("font-family: Consolas; padding: 2px 6px; background: #f0f0f0; border-radius: 3px;")
        gl.addWidget(self.lbl_bl, 3, 1)

        note = QLabel("区域为四边形区域，车辆进入该区域将触发入侵警报。\n坐标值为相对画面宽高的比例，范围 0.0 ~ 1.0。")
        note.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 4px;")
        gl.addWidget(note, 4, 0, 1, 2)

        self.chk_snapshot = QCheckBox("自动保存入侵截图")
        self.chk_snapshot.setChecked(True)
        gl.addWidget(self.chk_snapshot, 5, 0, 1, 2)

        self.btn_reset_zone = QPushButton("重置区域为默认")
        gl.addWidget(self.btn_reset_zone, 6, 0, 1, 2)

        layout.addWidget(g)

        g2 = QGroupBox("显示开关")
        g2l = QVBoxLayout(g2)

        self.chk_zone_overlay = QCheckBox("显示区域覆盖层")
        self.chk_zone_overlay.setChecked(True)
        g2l.addWidget(self.chk_zone_overlay)

        self.chk_zone_alerts = QCheckBox("显示入侵警报文字")
        self.chk_zone_alerts.setChecked(True)
        g2l.addWidget(self.chk_zone_alerts)

        layout.addWidget(g2)
        layout.addStretch()

        return w

    def _build_video_area(self):
        layout = QHBoxLayout()
        layout.setSpacing(8)

        left_panel = QWidget()
        left_panel.setStyleSheet("QWidget { background: #f8f9fa; border-radius: 6px; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)

        left_header = QHBoxLayout()
        self.btn_hide_left = QPushButton("◀ 隐藏")
        self.btn_hide_left.setFixedWidth(70)
        self.btn_hide_left.setStyleSheet("QPushButton { background: #3498db; color: white; border: none; border-radius: 4px; padding: 4px 8px; } QPushButton:hover { background: #2980b9; }")
        left_title = QLabel("原始输入视频流")
        left_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        left_header.addWidget(self.btn_hide_left)
        left_header.addWidget(left_title)
        left_header.addStretch()

        self.view_raw = QLabel()
        self.view_raw.setMinimumSize(480, 360)
        self.view_raw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_raw.setStyleSheet("""
            QLabel {
                background: #1a1a2e;
                border: 3px solid #3498db;
                border-radius: 6px;
                color: #95a5a6;
                font-size: 18px;
            }
        """)
        self.view_raw.setAlignment(Qt.AlignCenter)
        self.view_raw.setText("原始视频\n\n请点击「打开视频」加载视频文件")

        left_layout.addLayout(left_header)
        left_layout.addWidget(self.view_raw, 1)

        right_panel = QWidget()
        right_panel.setStyleSheet("QWidget { background: #f8f9fa; border-radius: 6px; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)

        right_header = QHBoxLayout()
        right_title = QLabel("算法检测结果")
        right_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        self.btn_toggle_view = QPushButton("切换 ▶")
        self.btn_toggle_view.setFixedWidth(80)
        self.btn_toggle_view.setStyleSheet("QPushButton { background: #e74c3c; color: white; border: none; border-radius: 4px; padding: 4px 8px; } QPushButton:hover { background: #c0392b; }")
        right_header.addWidget(right_title)
        right_header.addStretch()
        right_header.addWidget(self.btn_toggle_view)

        self.view_processed = QLabel()
        self.view_processed.setMinimumSize(480, 360)
        self.view_processed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_processed.setStyleSheet("""
            QLabel {
                background: #1a1a2e;
                border: 3px solid #e74c3c;
                border-radius: 6px;
                color: #95a5a6;
                font-size: 18px;
            }
        """)
        self.view_processed.setAlignment(Qt.AlignCenter)
        self.view_processed.setText("检测结果\n\n打开视频后将显示YOLO检测效果")

        right_layout.addLayout(right_header)
        right_layout.addWidget(self.view_processed, 1)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #bdc3c7;")

        layout.addWidget(left_panel, 1)
        layout.addWidget(separator)
        layout.addWidget(right_panel, 1)

        return layout

    def _build_controls(self):
        layout = QHBoxLayout()
        layout.setSpacing(6)

        btn_style = """
            QPushButton {
                padding: 6px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
        """

        self.btn_open = QPushButton("📂 打开视频")
        self.btn_open.setStyleSheet(btn_style + "QPushButton { background: #2c3e50; color: white; } QPushButton:hover { background: #34495e; }")

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setStyleSheet(btn_style + "QPushButton { background: #27ae60; color: white; } QPushButton:hover { background: #2ecc71; }")

        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setStyleSheet(btn_style + "QPushButton { background: #f39c12; color: white; } QPushButton:hover { background: #e67e22; }")

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setStyleSheet(btn_style + "QPushButton { background: #e74c3c; color: white; } QPushButton:hover { background: #c0392b; }")

        self.btn_reset = QPushButton("⟳ 重置计数")
        self.btn_reset.setStyleSheet(btn_style + "QPushButton { background: #8e44ad; color: white; } QPushButton:hover { background: #9b59b6; }")

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #bdc3c7; margin: 0 4px;")

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #bdc3c7; margin: 0 4px;")

        self.chk_boxes = QCheckBox("检测框")
        self.chk_boxes.setChecked(True)
        self.chk_trails = QCheckBox("轨迹线")
        self.chk_trails.setChecked(True)
        self.chk_count_line = QCheckBox("计数线")
        self.chk_count_line.setChecked(True)
        self.chk_zone_display = QCheckBox("敏感区域")
        self.chk_zone_display.setChecked(True)

        self.label_total = QLabel("车辆总计: 0")
        self.label_total.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50; padding: 0 8px;")

        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_reset)
        layout.addWidget(sep)
        layout.addWidget(self.chk_boxes)
        layout.addWidget(self.chk_trails)
        layout.addWidget(self.chk_count_line)
        layout.addWidget(self.chk_zone_display)
        layout.addWidget(sep2)
        layout.addWidget(self.label_total)
        layout.addStretch()

        return layout

    def _init_status_bar(self):
        self.setStatusBar(QStatusBar(self))
        self.statusBar().setStyleSheet("QStatusBar { background: #ecf0f1; border-top: 1px solid #bdc3c7; padding: 2px 8px; }")

        self.status_label = QLabel("就绪 - 请打开视频文件开始检测")
        self.status_label.setStyleSheet("color: #2c3e50;")

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("font-weight: bold; color: #27ae60; padding: 0 8px;")

        self.frame_label = QLabel("帧: 0")
        self.frame_label.setStyleSheet("color: #7f8c8d; padding: 0 8px;")

        footer_school = QLabel("粤台产业科技学院 - 计算机视觉研究中心")
        footer_school.setStyleSheet("color: #7f8c8d; font-size: 11px;")

        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.frame_label)
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().addPermanentWidget(footer_school)

    def _scan_models(self):
        self.cmb_model.clear()
        candidates = [
            PROJECT_ROOT / "models",
            PROJECT_ROOT.parent / "Exp_Final" / "models",
        ]
        added = set()
        for d in candidates:
            if d.exists():
                for f in sorted(d.glob("*.pt")):
                    if f.name not in added:
                        added.add(f.name)
                        label = f"[{d.parent.name}] {f.name}" if d.parent.name != "1" else f.name
                        self.cmb_model.addItem(label, str(f))
        if self.cmb_model.count() == 0:
            self.cmb_model.addItem("未找到模型文件（请放入 models/ 目录）", "")

    def _connect_signals(self):
        self.worker.change_pixmap_raw.connect(self._update_raw_view)
        self.worker.change_pixmap_proc.connect(self._update_processed_view)
        self.worker.update_stats.connect(self._update_stats)
        self.worker.update_count.connect(self._update_counts)
        self.worker.update_fps.connect(self._update_fps)
        self.worker.frame_processed.connect(self._on_frame_processed)

        self.btn_open.clicked.connect(self._open_video)
        self.btn_play.clicked.connect(self._play_video)
        self.btn_pause.clicked.connect(self._pause_video)
        self.btn_stop.clicked.connect(self._stop_video)
        self.btn_reset.clicked.connect(self._reset_counts)
        self.btn_hide_left.clicked.connect(self._toggle_left_panel)
        self.btn_toggle_view.clicked.connect(self._toggle_view)
        self.btn_reset_zone.clicked.connect(self._reset_zone)

        self.spin_conf.valueChanged.connect(self._on_conf_changed)
        self.slider_count_line.valueChanged.connect(self._on_count_line_changed)

        self.chk_boxes.stateChanged.connect(lambda v: setattr(self.worker.tracker, 'show_boxes', self.chk_boxes.isChecked()))
        self.chk_trails.stateChanged.connect(lambda v: setattr(self.worker.tracker, 'show_trails', self.chk_trails.isChecked()))
        self.chk_count_line.stateChanged.connect(lambda v: setattr(self.worker.tracker, 'show_count_line', self.chk_count_line.isChecked()))
        self.chk_zone_display.stateChanged.connect(lambda v: setattr(self.worker.tracker, 'show_zone', self.chk_zone_display.isChecked()))
        self.chk_snapshot.stateChanged.connect(lambda v: setattr(self.worker.tracker, 'snapshot_enabled', self.chk_snapshot.isChecked()))
        self.chk_zone_overlay.stateChanged.connect(self._on_zone_overlay_changed)

        self.chk_person.stateChanged.connect(self._update_target_classes)
        self.chk_car.stateChanged.connect(self._update_target_classes)
        self.chk_bus.stateChanged.connect(self._update_target_classes)
        self.chk_truck.stateChanged.connect(self._update_target_classes)

    def _on_conf_changed(self, val):
        self.worker.tracker.conf_threshold = val

    def _on_count_line_changed(self, val):
        ratio = val / 100.0
        self.worker.tracker.count_line_y_ratio = ratio
        self.label_count_line_val.setText(f"{val}%")

    def _on_zone_overlay_changed(self, val):
        self.worker.tracker.show_zone = self.chk_zone_overlay.isChecked()

    def _update_target_classes(self):
        classes = []
        if self.chk_person.isChecked():
            classes.append('person')
        if self.chk_car.isChecked():
            classes.append('car')
        if self.chk_bus.isChecked():
            classes.append('bus')
        if self.chk_truck.isChecked():
            classes.append('truck')
        if not classes:
            classes = ['car', 'bus', 'truck']
        self.worker.tracker.target_classes = classes

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*.*)"
        )
        if not path:
            return
        self._stop_video()
        ok = self.worker.open_video(path)
        if ok:
            self.status_label.setText(f"已加载: {Path(path).name}")
            self.view_raw.setText("加载中...")
            self.view_processed.setText("加载中...")
        else:
            self.status_label.setText(f"无法打开视频: {Path(path).name}")

    def _play_video(self):
        if self.worker.cap is None or not self.worker.cap.isOpened():
            self.status_label.setText("请先打开视频文件")
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.status_label.setText("继续播放")
        elif not self.worker.is_running:
            self.worker.start()
            self.status_label.setText("正在播放...")
        else:
            self.status_label.setText("已在播放中")

    def _pause_video(self):
        if self.worker.is_running and not self.worker.is_paused:
            self.worker.pause()
            self.status_label.setText("已暂停")

    def _stop_video(self):
        self.worker.stop()
        self.worker.wait(500)
        self.view_raw.clear()
        self.view_raw.setText("原始视频\n\n请点击「打开视频」加载视频文件")
        self.view_processed.clear()
        self.view_processed.setText("检测结果\n\n打开视频后将显示YOLO检测效果")
        self.status_label.setText("已停止")
        self.fps_label.setText("FPS: 0.0")
        self.frame_label.setText("帧: 0")

    def _reset_counts(self):
        self.worker.tracker.reset()
        self._update_counts({
            'car': {'in': 0, 'out': 0},
            'bus': {'in': 0, 'out': 0},
            'truck': {'in': 0, 'out': 0},
        })
        self.intrusion_log.clear()
        self.status_label.setText("计数已重置")

    def _toggle_left_panel(self):
        visible = self.view_raw.isVisible()
        self.view_raw.setVisible(not visible)
        self.btn_hide_left.setText("▶ 显示" if not visible else "◀ 隐藏")

    def _toggle_view(self):
        raw_pix = self.view_raw.pixmap()
        proc_pix = self.view_processed.pixmap()
        if raw_pix:
            self.view_processed.setPixmap(raw_pix)
        if proc_pix:
            self.view_raw.setPixmap(proc_pix)

    def _reset_zone(self):
        h = self.worker.tracker.frame_h or 720
        w = self.worker.tracker.frame_w or 1280
        self.worker.tracker.set_default_zone(w, h)
        self.lbl_tl.setText("(0.30, 0.35)")
        self.lbl_tr.setText("(0.70, 0.35)")
        self.lbl_br.setText("(0.90, 0.85)")
        self.lbl_bl.setText("(0.10, 0.85)")
        self.status_label.setText("敏感区域已重置为默认")

    def _update_raw_view(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.view_raw.setPixmap(pix.scaled(
            self.view_raw.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_processed_view(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.view_processed.setPixmap(pix.scaled(
            self.view_processed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_stats(self, info):
        total = 0
        for cls_name in ['car', 'bus', 'truck']:
            c = self.worker.tracker.counts[cls_name]
            total += c['in'] + c['out']
        self.label_total.setText(f"车辆总计: {total}")

    def _update_counts(self, counts):
        row_map = {'car': 0, 'bus': 1, 'truck': 2}
        grand_total = 0
        for cls_name, data in counts.items():
            r = row_map.get(cls_name)
            if r is None:
                continue
            in_val = data.get('in', 0)
            out_val = data.get('out', 0)
            total = in_val + out_val
            grand_total += total
            self.count_table.item(r, 1).setText(str(in_val))
            self.count_table.item(r, 2).setText(str(out_val))
            self.count_table.item(r, 3).setText(str(total))
        self.label_total.setText(f"车辆总计: {grand_total}")

    def _update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.label_fps_display.setText(
            f"FPS: {fps:.1f}  |  帧数: {self.worker.tracker.frame_count}  |  "
            f"入侵: {len(self.worker.tracker.intruded_ids)}"
        )

    def _on_frame_processed(self, frame_count):
        self.frame_label.setText(f"帧: {frame_count}")

    def closeEvent(self, event):
        self._stop_video()
        event.accept()
