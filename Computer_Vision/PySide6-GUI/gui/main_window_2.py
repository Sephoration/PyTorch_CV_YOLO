# gui/main_window.py
import sys
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QSlider,
                               QTabWidget, QVBoxLayout, QWidget, QSizePolicy,
                               QGroupBox, QSpinBox, QApplication)

# 1. 导入后台线程发动机（总调度底座）
from qthreads.base_worker import BaseWorker

# 2. 导入我们刚刚写好的简单视讯处理卡带
from qthreads.tasks.simple_video_task import SimpleVideoTask

try:
    from qthreads.tasks.hand_volume_task import HandVolumeTask
except ImportError as e:
    print(f"【静态提示】HandVolumeTask 未挂载: {e}")
    HandVolumeTask = None

FingerCountTask = None
PPTControlTask = None


class VisionWorkstationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("计算机视觉综合实验工作站 - 选项卡流动版(学生教学模版)")
        self.resize(1360, 920)

        # ====================================================================
        # 教学挖空点 1：原本在这里实例化的后台执行引擎已拔除
        # 后续课程需要带学生在这里手写接通发动机：self.v_thread = BaseWorker(...)
        # ====================================================================

        self.init_ui()

        # ====================================================================
        # 教学挖空点 2：三条专用跨线程信号电缆已全部去除
        # 后续课程教到信号时，带学生在这里手写绑定：
        # self.v_thread.raw_frame_signal.connect(self.update_raw_view)
        # self.v_thread.processed_frame_signal.connect(self.update_processed_view)
        # self.v_thread.data_signal.connect(self.update_business_data)
        # ====================================================================

        # ====================================================================
        # 教学挖空点 3：原本在末尾自动隐藏左视窗并开局 start() 的逻辑已去除
        # 保持双窗口初始完全对齐呈现，处于静态等待状态
        # ====================================================================

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ----------------------------------------------------
        # 布局 A: 【顶部】复合控制头（大 LOGO 与 调参选项卡）
        # ----------------------------------------------------
        top_header_layout = QHBoxLayout()
        top_header_layout.setSpacing(20)

        # 左侧：正方形大 LOGO 区域
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
        self.spin_max_targets.setValue(2)
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
        self.slider_detect_con.setValue(75)
        self.slider_detect_con.valueChanged.connect(self.on_mp_global_params_changed)
        self.label_detect_con_val = QLabel("0.75")
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
        self.slider_presence_con.setEnabled(False)
        mp_global_layout.addWidget(self.col3_box, stretch=1.5)
        self.param_tabs.addTab(self.mp_global_tab, "MediaPipe 全局参数")

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
        self.btn_load_volume_model = QPushButton("🚀 加载手部侦测追踪模型 (HandModule)")
        self.btn_load_volume_model.setStyleSheet(
            "QPushButton { background-color: #005577; color: white; padding: 10px; border-radius: 5px; }")
        self.btn_load_volume_model.clicked.connect(self.on_load_volume_model_clicked)
        vol_layout.addWidget(self.btn_load_volume_model)
        vol_layout.addStretch()
        self.param_tabs.addTab(self.vol_tab, "手势音量控制")

        # 其他未开发的标签页
        self.count_tab = QWidget()
        self.param_tabs.addTab(self.count_tab, "手势数字识别")
        self.ppt_tab = QWidget()
        self.param_tabs.addTab(self.ppt_tab, "PPT 手势翻页")
        self.yolo_tab = QWidget()
        self.param_tabs.addTab(self.yolo_tab, "YOLO 深度学习配置")

        right_header_box.addWidget(self.param_tabs)
        top_header_layout.addWidget(self.logo_label)
        top_header_layout.addLayout(right_header_box)
        main_layout.addLayout(top_header_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        main_layout.addWidget(line)

        # ----------------------------------------------------
        # 布局 B: 【下方】左右双窗口显示容器
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
        self.view_raw.setScaledContents(True)

        self.left_video_box.addLayout(left_title_layout)
        self.left_video_box.addWidget(self.view_raw)
        video_layout.addWidget(self.left_video_widget)

        self.right_video_box = QVBoxLayout()
        self.right_video_box.setContentsMargins(0, 0, 0, 0)
        self.right_video_box.setSpacing(8)
        right_title_layout = QHBoxLayout()

        self.btn_toggle_view = QPushButton("👁️ 隐藏原始画面")  # 开局对称
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
        self.view_processed.setScaledContents(True)

        self.right_video_box.addLayout(right_title_layout)
        self.right_video_box.addWidget(self.view_processed)
        video_layout.addLayout(self.right_video_box)

        main_layout.addLayout(video_layout, stretch=1)

        # ----------------------------------------------------
        # 布局 C: 【底部】页脚签名
        # ----------------------------------------------------
        footer_layout = QHBoxLayout()
        self.label_status_msg = QLabel("系统状态: 界面初始化完成，等待线程引擎挂载...")
        self.label_status_msg.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self.label_status_msg.setStyleSheet("color: #005577;")
        school_label = QLabel("计算机视觉研究中心")
        school_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        school_label.setStyleSheet("color: #1A365D;")
        footer_layout.addWidget(self.label_status_msg)
        footer_layout.addStretch()
        footer_layout.addWidget(school_label)
        main_layout.addLayout(footer_layout)

        self.param_tabs.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(0)

    # ==========================================
    # 核心槽函数（留白待学生后续补齐核心控制逻辑）
    # ==========================================
    @Slot()
    def on_hide_left_clicked(self):
        """【学生填空】控制隐藏左侧原生视窗"""
        # 静态隐藏基础示范
        self.left_video_widget.hide()
        self.btn_toggle_view.setText("👁️ 开启原始画面")

    @Slot()
    def on_toggle_view_clicked(self):
        """【学生填空】单双视窗联动及后端数据流开关控制"""
        if self.left_video_widget.isVisible():
            self.left_video_widget.hide()
            self.btn_toggle_view.setText("👁️ 开启原始画面")
        else:
            self.left_video_widget.show()
            self.btn_toggle_view.setText("👁️ 隐藏原始画面")

    def on_tab_changed(self, index):
        """【学生填空】Tab页签切换响应，用于动态热插拔算法任务"""
        if not hasattr(self, "label_status_msg"): return
        if index == 0:
            self.label_status_msg.setText("系统状态: MediaPipe 参数审查中")
        elif index == 1:
            self.label_status_msg.setText("系统状态: 请点击「加载模型」部署手部组件")

    @Slot()
    def on_load_volume_model_clicked(self):
        """【学生填空】手动载入模型，触发媒体源热热切与算法卡带注入"""
        print("点击了加载手部模型按钮（等待学生编写线程接通、change_media_source(0) 及 switch_task(task) 逻辑）")

    def on_mp_category_changed(self, index):
        if index == 0:
            self.combo_pose_complexity.setEnabled(False)
            self.slider_presence_con.setEnabled(False)
        else:
            self.combo_pose_complexity.setEnabled(True)
            self.slider_presence_con.setEnabled(True)
        self.on_mp_global_params_changed()

    def sync_mp_global_params(self, task_instance):
        """【学生填空】运行时超参数动态同步契约"""
        pass

    def on_mp_global_params_changed(self):
        """【学生填空】参数面板数值更新，触发多线程底层参数改写"""
        detect_con_val = self.slider_detect_con.value() / 100.0
        presence_con_val = self.slider_presence_con.value() / 100.0
        track_con_val = self.slider_track_con.value() / 100.0
        self.label_detect_con_val.setText(f"{detect_con_val:.2f}")
        self.label_presence_con_val.setText(f"{presence_con_val:.2f}")
        self.label_track_con_val.setText(f"{track_con_val:.2f}")

    # ==========================================
    # 跨线程数据流渲染接收槽（保留给学生后续做信号绑定联通用）
    # ==========================================
    @Slot(QImage)
    def update_raw_view(self, q_img):
        self.view_raw.setPixmap(QPixmap.fromImage(q_img))

    @Slot(QImage)
    def update_processed_view(self, q_img):
        self.view_processed.setPixmap(QPixmap.fromImage(q_img))

    @Slot(dict)
    def update_business_data(self, data):
        if "status" in data:
            self.label_status_msg.setText(f"业务反馈: {data['status']}")

    def closeEvent(self, event):
        # 提醒学生释放线程
        print("主窗口正在关闭...")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = VisionWorkstationGUI()
    window.show()
    sys.exit(app.exec())