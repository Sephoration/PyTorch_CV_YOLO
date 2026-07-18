# gui/main_window.py
import os
import sys
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QSlider,
                               QTabWidget, QVBoxLayout, QWidget, QSizePolicy,
                               QGroupBox, QSpinBox, QFileDialog, QApplication,
                               QCheckBox, QGridLayout, QMessageBox)

from qthreads.base_worker import BaseWorker
from qthreads.base_worker import SourceType
from qthreads.tasks.hand_volume_task import HandVolumeTask
from qthreads.tasks.yoga_predict_task import YogaPredictTask
from qthreads.tasks.yolo_trajectory_task import YoloTrajectoryTask

FingerCountTask = None
PPTControlTask = None


class VisionWorkstationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("计算机视觉综合实验工作站 - 选项卡流动版")
        self.resize(1360, 920)
        self.center()  # 新增：讓視窗在啟動時自動置中

        self.v_thread = BaseWorker(
            source="videos/ComputerVision-1.mp4",
            source_type=SourceType.VIDEO_FILE
        )
        self.init_ui()

        self.v_thread.raw_frame_signal.connect(self.update_raw_view)
        self.v_thread.processed_frame_signal.connect(self.update_processed_view)
        self.v_thread.data_signal.connect(self.update_business_data)

        self.v_thread.start()

    def center(self):
        """🎯 計算螢幕可用空間的中心點，並將視窗移至該處"""
        qr = self.frameGeometry()  # 取得視窗本身的幾何大小
        cp = QApplication.primaryScreen().availableGeometry().center()  # 取得目前主螢幕的中心點
        qr.moveCenter(cp)  # 將視窗的中心點對齊螢幕的中心點
        self.move(qr.topLeft())  # 移動視窗到對齊後的左上角座標

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ----------------------------------------------------
        # 布局 A: 【顶部】复合控制头
        # ----------------------------------------------------
        top_header_layout = QHBoxLayout()
        top_header_layout.setSpacing(20)

        self.logo_label = QLabel()
        self.logo_label.setFixedSize(150, 150)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("border: 1px solid #CCCCCC; background: #FAFAFA;")
        right_header_box = QVBoxLayout()
        right_header_box.setSpacing(8)
        title_label = QLabel("计算机视觉研究中心")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setStyleSheet("color: #1A365D;")
        right_header_box.addWidget(title_label)

        self.param_tabs = QTabWidget()
        self.param_tabs.setFont(QFont("Microsoft YaHei", 10))

        # ==================== Tab 0: MediaPipe 全局参数 ====================
        self.mp_global_tab = QWidget()
        mp_global_layout = QHBoxLayout(self.mp_global_tab)

        self.col1_box = QGroupBox("1. 基础模式加载")
        col1_form = QFormLayout(self.col1_box)
        self.combo_mp_category = QComboBox()
        self.combo_mp_category.addItems(["手部跟踪 (Hand)", "姿态评估 (Pose)"])
        self.combo_mp_category.currentIndexChanged.connect(self.on_mp_category_changed)
        self.combo_running_mode = QComboBox()
        self.combo_running_mode.addItems(["VIDEO (实时视频流)", "IMAGE (静态单帧)"])
        self.combo_running_mode.currentIndexChanged.connect(self.on_mp_global_params_changed)
        col1_form.addRow("模型大类选择:", self.combo_mp_category)
        col1_form.addRow("模型运行模式:", self.combo_running_mode)
        mp_global_layout.addWidget(self.col1_box, stretch=1)

        self.col2_box = QGroupBox("2. 专属属性微调")
        col2_form = QFormLayout(self.col2_box)
        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 10)
        self.spin_max_targets.setValue(1)
        self.spin_max_targets.valueChanged.connect(self.on_mp_global_params_changed)
        self.combo_pose_complexity = QComboBox()
        self.combo_pose_complexity.addItems(["Lite (轻量快速)", "Full (平衡版)", "Heavy (高精度版)"])
        self.combo_pose_complexity.currentIndexChanged.connect(self.on_mp_global_params_changed)
        col2_form.addRow("最大侦测对象上限:", self.spin_max_targets)
        col2_form.addRow("姿态模型复杂度:", self.combo_pose_complexity)
        self.combo_pose_complexity.setEnabled(False)
        mp_global_layout.addWidget(self.col2_box, stretch=1)

        self.col3_box = QGroupBox("3. 模型置信度控制")
        col3_form = QFormLayout(self.col3_box)
        self.slider_detect_con = QSlider(Qt.Horizontal)
        self.slider_detect_con.setRange(0, 100)
        self.slider_detect_con.setValue(50)
        self.slider_detect_con.valueChanged.connect(self.on_mp_global_params_changed)
        self.label_detect_con_val = QLabel("0.50")
        detect_layout = QHBoxLayout()
        detect_layout.addWidget(self.slider_detect_con)
        detect_layout.addWidget(self.label_detect_con_val)

        self.slider_presence_con = QSlider(Qt.Horizontal)
        self.slider_presence_con.setRange(0, 100)
        self.slider_presence_con.setValue(50)
        self.slider_presence_con.valueChanged.connect(self.on_mp_global_params_changed)
        self.label_presence_con_val = QLabel("0.50")
        presence_layout = QHBoxLayout()
        presence_layout.addWidget(self.slider_presence_con)
        presence_layout.addWidget(self.label_presence_con_val)

        self.slider_track_con = QSlider(Qt.Horizontal)
        self.slider_track_con.setRange(0, 100)
        self.slider_track_con.setValue(50)
        self.slider_track_con.valueChanged.connect(self.on_mp_global_params_changed)
        self.label_track_con_val = QLabel("0.50")
        track_layout = QHBoxLayout()
        track_layout.addWidget(self.slider_track_con)
        track_layout.addWidget(self.label_track_con_val)

        col3_form.addRow("检测置信度 (Detection):", detect_layout)
        col3_form.addRow("存在置信度 (Presence):", presence_layout)
        col3_form.addRow("追踪置信度 (Tracking):", track_layout)
        mp_global_layout.addWidget(self.col3_box, stretch=1.5)

        self.param_tabs.addTab(self.mp_global_tab, "MediaPipe 参数配置")

        # ==================== Tab 1: 手势音量控制 ====================
        self.vol_tab = QWidget()
        vol_layout = QVBoxLayout(self.vol_tab)
        self.vol_tips = QLabel(
            "💡 手势音量控制指南：\n"
            "   伸出单手，通过拇指食指指尖拉伸距离即可实时控制Windows系统扬声器音量。\n"
            "   本模块基于自主设计的 [HandModule] 驱动运行，请在下方手动进行模型加载部署。"
        )
        self.vol_tips.setStyleSheet("color: #555555; font-style: italic;")
        vol_layout.addWidget(self.vol_tips)
        self.btn_load_volume_model = QPushButton("🚀 加载手势音量控制模型 (Gesture Volume Control Model)")
        self.btn_load_volume_model.setStyleSheet(
            "QPushButton { background-color: #005577; color: white; padding: 10px; border-radius: 5px; }")
        self.btn_load_volume_model.clicked.connect(self.on_load_volume_model_clicked)
        vol_layout.addWidget(self.btn_load_volume_model)
        vol_layout.addStretch()
        self.param_tabs.addTab(self.vol_tab, "手势音量控制")

        # ==================== Tab 2: 手势数字识别 ====================
        self.count_tab = QWidget()
        self.param_tabs.addTab(self.count_tab, "手势数字识别")

        # ==================== Tab 3: PPT 手势翻页 ====================
        self.ppt_tab = QWidget()
        self.param_tabs.addTab(self.ppt_tab, "PPT 手势翻页")

        # ==================== Tab 4: 瑜珈姿势检测 ====================
        self.yoga_tab = QWidget()
        self.init_yoga_tab_ui()
        self.param_tabs.addTab(self.yoga_tab, "瑜珈姿势检测")

        # ==================== Tab 5: YOLO 参数配置 ====================
        self.yolo_tab = QWidget()
        self.init_yolo_tab_ui()
        self.param_tabs.addTab(self.yolo_tab, "YOLO 参数配置")

        # ==================== 🎯 新增 Tab 6: 目标追踪 ====================
        self.yolo_track_tab = QWidget()
        self.init_yolo_track_tab_ui()  # 呼叫专属的 UI 构建函数
        self.param_tabs.addTab(self.yolo_track_tab, "目标追踪")

        right_header_box.addWidget(self.param_tabs)
        top_header_layout.addWidget(self.logo_label)
        top_header_layout.addLayout(right_header_box)
        main_layout.addLayout(top_header_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        main_layout.addWidget(line)

        # ----------------------------------------------------
        # 布局 B: 【下方】双窗口完美对齐对齐与单窗口机制
        # ----------------------------------------------------
        video_layout = QHBoxLayout()
        video_layout.setSpacing(15)

        self.left_video_widget = QWidget()
        self.left_video_box = QVBoxLayout(self.left_video_widget)
        self.left_video_box.setContentsMargins(0, 0, 0, 0)
        self.left_video_box.setSpacing(8)

        left_title_layout = QHBoxLayout()

        self.btn_hide_left = QPushButton("👁️ 隐藏原始画面")
        self.btn_hide_left.setFixedWidth(130)
        self.btn_hide_left.setStyleSheet(
            "QPushButton { background-color: #DDDDDD; border-radius: 4px; padding: 4px; } QPushButton:hover { background-color: #CCCCCC; }")
        self.btn_hide_left.clicked.connect(self.on_hide_left_clicked)

        self.left_title = QLabel(" 原始输入视讯流 (Source Video)")
        self.left_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        left_title_layout.addWidget(self.btn_hide_left)
        left_title_layout.addWidget(self.left_title)
        left_title_layout.addStretch()

        self.view_raw = QLabel("等待视频流输入...")
        self.view_raw.setAlignment(Qt.AlignCenter)
        self.view_raw.setStyleSheet("background-color: #1E1E1E; border: 2px solid #333333; color: #FFFFFF;")
        self.view_raw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view_raw.setMinimumSize(320, 240)

        self.left_video_box.addLayout(left_title_layout)
        self.left_video_box.addWidget(self.view_raw)

        video_layout.addWidget(self.left_video_widget, stretch=1)

        self.right_video_box = QVBoxLayout()
        self.right_video_box.setContentsMargins(0, 0, 0, 0)
        self.right_video_box.setSpacing(8)
        right_title_layout = QHBoxLayout()

        self.btn_toggle_view = QPushButton("👁️ 开启原始画面")
        self.btn_toggle_view.setFixedWidth(130)
        self.btn_toggle_view.setStyleSheet(
            "QPushButton { background-color: #DDDDDD; border-radius: 4px; padding: 4px; } QPushButton:hover { background-color: #CCCCCC; }")
        self.btn_toggle_view.clicked.connect(self.on_toggle_view_clicked)

        self.right_title = QLabel(" 算法检测结果 (Processed Output)")
        self.right_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        right_title_layout.addWidget(self.btn_toggle_view)
        right_title_layout.addWidget(self.right_title)
        right_title_layout.addStretch()

        self.view_processed = QLabel("等待算法启动...")
        self.view_processed.setAlignment(Qt.AlignCenter)
        self.view_processed.setStyleSheet("background-color: #1E1E1E; border: 2px solid #005577; color: #FFFFFF;")
        self.view_processed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view_processed.setMinimumSize(320, 240)

        self.right_video_box.addLayout(right_title_layout)
        self.right_video_box.addWidget(self.view_processed)

        video_layout.addLayout(self.right_video_box, stretch=1)
        main_layout.addLayout(video_layout, stretch=1)

        # ----------------------------------------------------
        # 布局 C: 页脚签名
        # ----------------------------------------------------
        footer_layout = QHBoxLayout()
        self.label_status_msg = QLabel("系统状态: 神经底座就绪，正在接收媒体流...")
        self.label_status_msg.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self.label_status_msg.setStyleSheet("color: #005577;")
        school_label = QLabel("计算机视觉研究中心")
        school_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        school_label.setStyleSheet("color: #1A365D;")
        footer_layout.addWidget(self.label_status_msg)
        footer_layout.addStretch()
        footer_layout.addWidget(school_label)
        main_layout.addLayout(footer_layout)

        self.left_video_widget.hide()
        self.v_thread.set_raw_stream_enabled(False)

        # ----------------------------------------------------
        # 🎯 UX 优化：利用 Emoji 直觉区分开发进度
        # ----------------------------------------------------
        self.param_tabs.setTabText(0, "✅ MediaPipe 参数配置")
        self.param_tabs.setTabText(1, "✅ 手势音量控制")
        self.param_tabs.setTabText(2, "🚧 手势数字识别")
        self.param_tabs.setTabText(3, "🚧 PPT 手势翻页")
        self.param_tabs.setTabText(4, "✅ 瑜珈姿势检测")
        self.param_tabs.setTabText(5, "✅ YOLO 参数配置")
        self.param_tabs.setTabText(6, "✅ 目标追踪")  # 状态更新为已实作

        self.param_tabs.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(0)

    # ====================================================================
    # 🎯 子页面 UI 构建区
    # ====================================================================
    def init_yoga_tab_ui(self):
        layout = QVBoxLayout()
        instruction = QLabel("请点选”选择文件夹”按钮来设定检测文件夹，"
                             "或者点选”选择视频”按钮来设定检测视频，"
                             "最后请执行”瑜珈姿势检测”按钮。")
        layout.addWidget(instruction)
        btn_layout = QHBoxLayout()
        # 🎨 美化资料夹按钮 (森林绿)
        self.btn_select_folder = QPushButton("📂 选择文件夹")
        self.btn_select_folder.setStyleSheet(
            "QPushButton { background-color: #2E8B57; color: white; padding: 10px; border-radius: 5px; font-weight: bold; } "
            "QPushButton:hover { background-color: #3CB371; }"
        )

        # 🎨 美化影片按钮 (温暖橘)
        self.btn_select_video = QPushButton("🎞️ 选择视频")
        self.btn_select_video.setStyleSheet(
            "QPushButton { background-color: #D35400; color: white; padding: 10px; border-radius: 5px; font-weight: bold; } "
            "QPushButton:hover { background-color: #E67E22; }"
        )

        # 🎨 美化启动按钮 (深海蓝)
        self.btn_start_yoga = QPushButton("🚀 瑜珈姿势检测")
        self.btn_start_yoga.setStyleSheet(
            "QPushButton { background-color: #005577; color: white; padding: 10px; border-radius: 5px; font-weight: bold; } "
            "QPushButton:hover { background-color: #006688; }"
        )

        self.btn_select_folder.clicked.connect(self.on_select_folder_clicked)
        self.btn_select_video.clicked.connect(self.on_select_video_clicked)
        self.btn_start_yoga.clicked.connect(self.on_start_yoga_clicked)

        btn_layout.addWidget(self.btn_select_folder)
        btn_layout.addWidget(self.btn_select_video)
        btn_layout.addWidget(self.btn_start_yoga)

        layout.addLayout(btn_layout)

        self.label_folder_path = QLabel("尚未选择文件夹")
        self.label_video_path = QLabel("尚未选择视频")
        layout.addWidget(self.label_folder_path)
        layout.addWidget(self.label_video_path)

        self.yoga_tab.setLayout(layout)

    def init_yolo_tab_ui(self):
        """构建 Tab 5: YOLO 参数配置"""
        layout = QHBoxLayout(self.yolo_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # 📦 区块一：基础模型配置 (移除了视频选择)
        col1_box = QGroupBox("1. 基础模型配置")
        col1_form = QFormLayout(col1_box)

        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.addItems(["yolo26n.pt (轻量极速版)", "yolo26s.pt (标准精确版)"])
        self.combo_yolo_model.setCurrentIndex(1)  # 默认使用 S 版

        col1_form.addRow("模型版本:", self.combo_yolo_model)
        layout.addWidget(col1_box, stretch=1)

        # 📦 区块二：引擎推理超参数
        col2_box = QGroupBox("2. 引擎推理超参数")
        col2_form = QFormLayout(col2_box)

        self.slider_yolo_conf = QSlider(Qt.Horizontal)
        self.slider_yolo_conf.setRange(1, 100)
        self.slider_yolo_conf.setValue(25)  # 预设 0.25
        self.slider_yolo_conf.valueChanged.connect(self.on_yolo_params_changed)
        self.label_yolo_conf_val = QLabel("0.25")

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.slider_yolo_conf)
        conf_layout.addWidget(self.label_yolo_conf_val)

        self.slider_yolo_iou = QSlider(Qt.Horizontal)
        self.slider_yolo_iou.setRange(1, 100)
        self.slider_yolo_iou.setValue(70)  # 预设 0.70
        self.slider_yolo_iou.valueChanged.connect(self.on_yolo_params_changed)
        self.label_yolo_iou_val = QLabel("0.70")

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(self.slider_yolo_iou)
        iou_layout.addWidget(self.label_yolo_iou_val)

        col2_form.addRow("置信度 (Conf):", conf_layout)
        col2_form.addRow("交并比 (IoU):", iou_layout)
        layout.addWidget(col2_box, stretch=1)

        # 📦 区块三：目标类别过滤器
        col3_box = QGroupBox("3. 目标类别过滤器")
        col3_layout = QVBoxLayout(col3_box)

        chk_layout = QGridLayout()
        self.chk_person = QCheckBox("行人 (person)")
        self.chk_car = QCheckBox("汽车 (car)")
        self.chk_bus = QCheckBox("公交车 (bus)")
        self.chk_truck = QCheckBox("卡车 (truck)")
        self.chk_motorcycle = QCheckBox("摩托车 (motorcycle)")
        self.chk_bicycle = QCheckBox("自行车 (bicycle)")

        # 预设勾选交通主要元素
        self.chk_person.setChecked(True)
        self.chk_car.setChecked(True)
        self.chk_bus.setChecked(True)
        self.chk_truck.setChecked(True)

        chk_layout.addWidget(self.chk_person, 0, 0)
        chk_layout.addWidget(self.chk_car, 0, 1)
        chk_layout.addWidget(self.chk_bus, 1, 0)
        chk_layout.addWidget(self.chk_truck, 1, 1)
        chk_layout.addWidget(self.chk_motorcycle, 2, 0)
        chk_layout.addWidget(self.chk_bicycle, 2, 1)

        # 绑定点击事件，实现热更新
        for chk in [self.chk_person, self.chk_car, self.chk_bus, self.chk_truck, self.chk_motorcycle, self.chk_bicycle]:
            chk.stateChanged.connect(self.on_yolo_params_changed)

        col3_layout.addLayout(chk_layout)
        col3_layout.addStretch()
        layout.addWidget(col3_box, stretch=1.5)

    def init_yolo_track_tab_ui(self):
        """🎯 构建 Tab 6: 目标追踪的业务控制面板"""
        layout = QHBoxLayout(self.yolo_track_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # 📦 区块一：视觉特效控制 (加入视频选择平衡高度)
        col1_box = QGroupBox("1. 视觉特效控制")
        col1_layout = QVBoxLayout(col1_box)

        self.chk_show_trail = QCheckBox("启用追踪轨迹 (Trail) 渲染")
        self.chk_show_trail.setChecked(True)  # 默认打开画线
        self.chk_show_trail.stateChanged.connect(self.on_yolo_track_params_changed)

        self.btn_select_yolo_video = QPushButton("选择测试视频")
        self.btn_select_yolo_video.setStyleSheet(
            "QPushButton { background-color: #D35400; color: white; padding: 10px; border-radius: 5px; font-weight: bold; } "
            "QPushButton:hover { background-color: #E67E22; }"
        )
        self.btn_select_yolo_video.clicked.connect(self.on_select_yolo_video_clicked)
        self.label_yolo_video_path = QLabel("尚未选择视频 (默认使用内置流)")
        self.label_yolo_video_path.setStyleSheet("color: #666666; font-size: 11px;")
        self.label_yolo_video_path.setWordWrap(True)

        col1_layout.addWidget(self.chk_show_trail)
        col1_layout.addSpacing(10)
        col1_layout.addWidget(self.btn_select_yolo_video)
        col1_layout.addWidget(self.label_yolo_video_path)
        col1_layout.addStretch()
        layout.addWidget(col1_box, stretch=1)

        # 📦 区块二：轨迹业务逻辑微调
        col2_box = QGroupBox("2. 轨迹业务逻辑微调")
        col2_form = QFormLayout(col2_box)

        # 轨迹保留长度 Slider
        self.slider_trail_length = QSlider(Qt.Horizontal)
        self.slider_trail_length.setRange(1, 150)
        self.slider_trail_length.setValue(50)
        self.slider_trail_length.valueChanged.connect(self.on_yolo_track_params_changed)
        self.label_trail_length_val = QLabel("50")

        length_layout = QHBoxLayout()
        length_layout.addWidget(self.slider_trail_length)
        length_layout.addWidget(self.label_trail_length_val)

        # 防丢帧容忍度 Slider
        self.slider_lost_threshold = QSlider(Qt.Horizontal)
        self.slider_lost_threshold.setRange(5, 60)
        self.slider_lost_threshold.setValue(20)
        self.slider_lost_threshold.valueChanged.connect(self.on_yolo_track_params_changed)
        self.label_lost_threshold_val = QLabel("20")

        lost_layout = QHBoxLayout()
        lost_layout.addWidget(self.slider_lost_threshold)
        lost_layout.addWidget(self.label_lost_threshold_val)

        col2_form.addRow("轨迹保留长度 (帧):", length_layout)
        col2_form.addRow("防丢帧容忍度 (帧):", lost_layout)
        layout.addWidget(col2_box, stretch=1.5)

        # 📦 区块三：启动引擎
        col3_box = QGroupBox("3. 启动引擎")
        col3_layout = QVBoxLayout(col3_box)

        tips = QLabel("注意：启动前请确保已经在【YOLO 参数配置】\n页签中设定好目标过滤类别。")
        tips.setStyleSheet("color: #555555; font-style: italic;")

        self.btn_start_yolo_track = QPushButton("🚀 启动 YOLO 目标追踪模块")
        self.btn_start_yolo_track.setStyleSheet(
            "QPushButton { background-color: #005577; color: white; padding: 10px; border-radius: 5px; font-weight: bold; }")
        self.btn_start_yolo_track.clicked.connect(self.on_start_yolo_track_clicked)

        col3_layout.addWidget(tips)
        col3_layout.addWidget(self.btn_start_yolo_track)
        col3_layout.addStretch()
        layout.addWidget(col3_box, stretch=1)

    def on_tab_changed(self, index):
        if not hasattr(self, "label_status_msg"): return

        if index in [1, 2, 3]:
            self.combo_mp_category.blockSignals(True)
            self.combo_running_mode.blockSignals(True)
            self.combo_mp_category.setCurrentIndex(0)
            self.combo_running_mode.setCurrentIndex(0)
            self.combo_mp_category.blockSignals(False)
            self.combo_running_mode.blockSignals(False)
            self.combo_pose_complexity.setEnabled(False)
            self.on_mp_global_params_changed()

        elif index == 4:
            self.combo_mp_category.blockSignals(True)
            self.combo_running_mode.blockSignals(True)
            self.combo_mp_category.setCurrentIndex(1)
            self.combo_mp_category.blockSignals(False)
            self.combo_running_mode.blockSignals(False)
            self.combo_pose_complexity.setEnabled(True)
            self.on_mp_global_params_changed()

        if index == 0:
            self.label_status_msg.setText("系统状态: MediaPipe 参数审查中")
        elif index == 1:
            self.label_status_msg.setText("系统状态: 切换到【手势音量控制】配置页。请点击「加载模型」开启服务")
        elif index == 2:
            self.label_status_msg.setText("系统状态: 切换到【手势数字识别】。请等待后续开发...")
        elif index == 3:
            self.label_status_msg.setText("系统状态: 切换到【PPT手势翻页】。请等待后续开发...")
        elif index == 4:
            self.label_status_msg.setText("系统状态: 切换到【瑜珈姿势检测】配置页。等待挂载 Pose 算法外挂")
        elif index == 5:
            self.label_status_msg.setText("系统状态: YOLO 基础参数设定就绪")
        elif index == 6:
            self.label_status_msg.setText("系统状态: 切换到【目标追踪】。请选择视频并点击启动模块测试")

    # ====================================================================
    # 🎯 YOLO 专属槽函数区 (含目标追踪事件)
    # ====================================================================
    @Slot()
    def on_select_yolo_video_clicked(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 测试视频",
            "./videos/", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.yolo_video_path = video_path
            file_name = os.path.basename(video_path)
            self.label_yolo_video_path.setText(f"已选: {file_name}")

    def get_yolo_target_classes(self):
        """收集面板上所有打勾的类别名称"""
        obj_list = []
        if self.chk_person.isChecked(): obj_list.append('person')
        if self.chk_car.isChecked(): obj_list.append('car')
        if self.chk_bus.isChecked(): obj_list.append('bus')
        if self.chk_truck.isChecked(): obj_list.append('truck')
        if self.chk_motorcycle.isChecked(): obj_list.append('motorcycle')
        if self.chk_bicycle.isChecked(): obj_list.append('bicycle')
        return obj_list

    def on_yolo_params_changed(self):
        """Tab 5：YOLO 底层参数 UI 数值更新 & 热更新至后台追踪器"""
        conf_val = self.slider_yolo_conf.value() / 100.0
        iou_val = self.slider_yolo_iou.value() / 100.0
        self.label_yolo_conf_val.setText(f"{conf_val:.2f}")
        self.label_yolo_iou_val.setText(f"{iou_val:.2f}")

        # 如果当前正在运行的是 YoloTrajectoryTask，则触发后台的参数热更新
        if hasattr(self.v_thread, 'current_task') and isinstance(self.v_thread.current_task, YoloTrajectoryTask):
            # 🎯 修复点：直接呼叫当前 Task 的 update_special_params，绕过 BaseWorker
            self.v_thread.current_task.update_special_params({
                "detection_con": conf_val,
                "iou": iou_val,
                "obj_list": self.get_yolo_target_classes()
            })

    def on_yolo_track_params_changed(self):
        """Tab 6：目标追踪面板控制数值更新 & 热更新"""
        trail_len = self.slider_trail_length.value()
        lost_thresh = self.slider_lost_threshold.value()
        self.label_trail_length_val.setText(str(trail_len))
        self.label_lost_threshold_val.setText(str(lost_thresh))

        # 传递给后台运行中的任务
        if hasattr(self.v_thread, 'current_task') and isinstance(self.v_thread.current_task, YoloTrajectoryTask):
            # 🎯 修复点：直接呼叫当前 Task 的 update_special_params，绕过 BaseWorker
            self.v_thread.current_task.update_special_params({
                "show_trail": self.chk_show_trail.isChecked(),
                "trail_length": trail_len,
                "lost_threshold": lost_thresh
            })

    @Slot()
    def on_start_yolo_track_clicked(self):
        """正式装载目标追踪引擎，结合 Tab 5 和 Tab 6 的参数发车！"""
        if YoloTrajectoryTask is None:
            self.label_status_msg.setText("系统状态: 启动失败！[YoloTrajectoryTask] 未找到或代码异常。")
            return

        # 1. 切入指定的媒体视频源
        if hasattr(self, 'yolo_video_path'):
            self.v_thread.change_media_source(new_source=self.yolo_video_path, new_type=SourceType.VIDEO_FILE)
        else:
            # 修改了没选视频时的提示用语
            self.label_status_msg.setText("系统提示: 您没有选择测试视频，将继续使用当前媒体流。")

        # 2. 强制开启双画面显示 (如果左侧原图被关了的话)
        if not self.left_video_widget.isVisible():
            self.left_video_widget.show()
            self.btn_toggle_view.setText("👁️ 隐藏原始画面")
            self.v_thread.set_raw_stream_enabled(True)

        # 3. 实例化 Task 并一站式注入全套超参数
        task = YoloTrajectoryTask()

        # 解析下拉选单，提取出模型档名 (把 "yolo26n.pt (轻量极速版)" 切出 "yolo26n.pt")
        selected_model_text = self.combo_yolo_model.currentText()
        task.model_name = selected_model_text.split(" ")[0]

        # 注入 Tab 5 的引擎全局参数
        task.detection_con = self.slider_yolo_conf.value() / 100.0
        task.iou = self.slider_yolo_iou.value() / 100.0
        task.obj_list = self.get_yolo_target_classes()

        # 注入 Tab 6 的目标追踪特性参数
        task.show_trail = self.chk_show_trail.isChecked()
        task.trail_length = self.slider_trail_length.value()
        task.lost_threshold = self.slider_lost_threshold.value()

        # 4. 把填装好参数的 Task 送进后台引爆
        self.v_thread.switch_task(task)
        self.label_status_msg.setText("系统状态: 🚀 YOLO 目标追踪任务加载成功，引擎启动！")

    # ====================================================================
    # 🎯 瑜珈及原始槽函数区 (已完成防呆与状态机修复)
    # ====================================================================
    @Slot()
    def on_select_folder_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择瑜伽测试集文件夹",
            "./datasets/")
        if folder_path:
            self.label_folder_path.setText(f"已选择: {folder_path}")
            self.yoga_folder_path = folder_path
            self.yoga_active_mode = 'folder'  # 🎯 记录最后一次的选择意图

    @Slot()
    def on_select_video_clicked(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择瑜伽测试视频",
            "./videos/",
            "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.label_video_path.setText(f"已选择: {os.path.basename(video_path)}")
            self.yoga_video_path = video_path
            self.yoga_active_mode = 'video'  # 🎯 记录最后一次的选择意图

    @Slot()
    def on_start_yoga_clicked(self):
        # 1. 弹窗提示防呆
        if getattr(self, 'yoga_active_mode', None) is None:
            QMessageBox.warning(self, "操作提示", "请先点击「选择文件夹」或「选择视频」来设定测试来源！")
            self.label_status_msg.setText("系统状态: 启动失败，请先选择文件夹或视频！")
            return

        # 2. 根据最后一次的意图执行 (不会互相清除文字，直接认最后一次点的模式)
        if self.yoga_active_mode == 'folder':
            source = self.yoga_folder_path
            source_type = SourceType.IMAGE_FOLDER
        else:  # self.yoga_active_mode == 'video'
            source = self.yoga_video_path
            source_type = SourceType.VIDEO_FILE

        self.v_thread.change_media_source(new_source=source, new_type=source_type)
        task = YogaPredictTask()
        self.v_thread.switch_task(task)
        self.v_thread.set_raw_stream_enabled(self.left_video_widget.isVisible())

        mode_text = "文件夹" if self.yoga_active_mode == 'folder' else "视频"
        self.label_status_msg.setText(f"系统状态: 🚀 正在运行瑜珈姿势检测 ({mode_text}模式)...")

    @Slot()
    def on_hide_left_clicked(self):
        self.left_video_widget.hide()
        self.btn_toggle_view.setText("👁️ 开启原始画面")
        self.v_thread.set_raw_stream_enabled(False)

    @Slot()
    def on_toggle_view_clicked(self):
        if self.left_video_widget.isVisible():
            self.left_video_widget.hide()
            self.btn_toggle_view.setText("👁️ 开启原始画面")
            self.v_thread.set_raw_stream_enabled(False)
        else:
            self.left_video_widget.show()
            self.btn_toggle_view.setText("👁️ 隐藏原始画面")
            self.v_thread.set_raw_stream_enabled(True)

    @Slot(QImage)
    def update_raw_view(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.view_raw.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view_raw.setPixmap(scaled_pixmap)

    @Slot(QImage)
    def update_processed_view(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.view_processed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view_processed.setPixmap(scaled_pixmap)

    @Slot(dict)
    def update_business_data(self, data):
        if "status" in data:
            self.label_status_msg.setText(f"业务反馈: {data['status']}")

    @Slot()
    def on_load_volume_model_clicked(self):
        if HandVolumeTask is None:
            self.label_status_msg.setText("系统状态: 加载失败！[HandVolumeTask] 导入异常。")
            return
        if self.combo_mp_category.currentIndex() != 0:
            self.label_status_msg.setText("系统错误: 当前为【手势音量控制】，模型大类必须选择 [手部跟踪 (Hand)]！")
            return
        if self.v_thread.source != 0:
            self.v_thread.change_media_source(new_source=0, new_type=SourceType.CAMERA)
        if not self.left_video_widget.isVisible():
            self.left_video_widget.show()
            self.btn_toggle_view.setText("👁️ 隐藏原始画面")
            self.v_thread.set_raw_stream_enabled(True)

        current_task = self.v_thread.current_task
        if isinstance(current_task, HandVolumeTask):
            self.label_status_msg.setText("系统状态: [HandModule] 已经运行中，无需重新加载")
            return

        task = HandVolumeTask()
        self.sync_mp_global_params(task)
        self.v_thread.switch_task(task)
        self.label_status_msg.setText("系统状态: 经过安全审查，最新 [HandModule] 参数同步并实例化成功！")

    def on_mp_global_params_changed(self):
        detect_con_val = self.slider_detect_con.value() / 100.0
        presence_con_val = self.slider_presence_con.value() / 100.0
        track_con_val = self.slider_track_con.value() / 100.0
        self.label_detect_con_val.setText(f"{detect_con_val:.2f}")
        self.label_presence_con_val.setText(f"{presence_con_val:.2f}")
        self.label_track_con_val.setText(f"{track_con_val:.2f}")

    def sync_mp_global_params(self, task_instance):
        task_instance.max_targets = self.spin_max_targets.value()
        task_instance.running_mode = "VIDEO" if self.combo_running_mode.currentIndex() == 0 else "IMAGE"
        task_instance.detection_con = self.slider_detect_con.value() / 100.0
        task_instance.presence_con = self.slider_presence_con.value() / 100.0
        task_instance.tracking_con = self.slider_track_con.value() / 100.0

    def on_mp_category_changed(self, index):
        if index == 0:
            self.combo_pose_complexity.setEnabled(False)
        else:
            self.combo_pose_complexity.setEnabled(True)
        self.on_mp_global_params_changed()

    def closeEvent(self, event):
        self.v_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = VisionWorkstationGUI()
    window.show()
    sys.exit(app.exec())