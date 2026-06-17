import sys
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QSlider, QTabWidget, QVBoxLayout,
                               QWidget, QSizePolicy)


# =====================================================================
# 【单元一教学提示】此时尚未引入 qthreads 文件夹，先将算法推理线程骨架注释掉
# =====================================================================
"""
class VideoProcessingThread(QThread):
    raw_frame_signal = Signal(QImage)
    processed_frame_signal = Signal(QImage)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.confidence_threshold = 0.5

    def run(self):
        print("视觉核心推理线程已启动...")
        while self.is_running:
            img_raw = np.zeros((480, 640, 3), dtype=np.uint8)
            img_raw[:, :] = [40, 44, 52]

            img_processed = np.zeros((480, 640, 3), dtype=np.uint8)
            img_processed[:, :] = [30, 60, 50]

            q_img_raw = QImage(img_raw.data, 640, 480, 640 * 3, QImage.Format_RGB888)
            q_img_processed = QImage(img_processed.data, 640, 480, 640 * 3, QImage.Format_RGB888)

            self.raw_frame_signal.emit(q_img_raw)
            self.processed_frame_signal.emit(q_img_processed)
            self.msleep(33)

    def set_parameters(self, conf):
        self.confidence_threshold = conf
        print(f"后台线程：置信度已更新为: {self.confidence_threshold}")

    def stop(self):
        self.is_running = False
        self.wait()
"""

# ==========================================
# 2. 前端核心：布局方案 4（经典底部控制台版）
# ==========================================
class VisionWorkstationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("计算机视觉综合实验工作站 - 布局方案 4 (教学范例版)")
        self.resize(1360, 900)

        # 1. 【单元一屏蔽】暂不实例化后端视觉处理线程
        # self.v_thread = VideoProcessingThread()

        # 2. 构建前端 UI 静态界面
        self.init_ui()

        # 3. 【单元一静态测试】直接加载本地图片来展示 UI 成果与拉伸适配
        self.load_static_test_images()

        # 4. 【单元一屏蔽】暂不绑定多线程信号和启动线程
        # self.v_thread.raw_frame_signal.connect(self.update_raw_view)
        # self.v_thread.processed_frame_signal.connect(self.update_processed_view)
        # self.v_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主垂直布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ----------------------------------------------------
        # 布局 A: 【最上方】双路视频流对比区域
        # ----------------------------------------------------
        video_layout = QHBoxLayout()
        video_layout.setSpacing(15)

        # 左路：原始视频
        self.left_video_box = QVBoxLayout()
        self.left_title = QLabel("原始输入视讯流 (Source Video)")
        self.left_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.view_raw = QLabel("等待视频流输入...")
        self.view_raw.setAlignment(Qt.AlignCenter)
        self.view_raw.setStyleSheet("background-color: #1E1E1E; border: 2px solid #333333; color: #FFFFFF;")
        self.view_raw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view_raw.setScaledContents(True)
        self.left_video_box.addWidget(self.left_title)
        self.left_video_box.addWidget(self.view_raw)

        # 右路：算法后处理输出
        self.right_video_box = QVBoxLayout()
        self.right_title = QLabel("算法检测结果 (Processed Output)")
        self.right_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.view_processed = QLabel("等待算法启动...")
        self.view_processed.setAlignment(Qt.AlignCenter)
        self.view_processed.setStyleSheet("background-color: #1E1E1E; border: 2px solid #005577; color: #FFFFFF;")
        self.view_processed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view_processed.setScaledContents(True)
        self.right_video_box.addWidget(self.right_title)
        self.right_video_box.addWidget(self.view_processed)

        video_layout.addLayout(self.left_video_box)
        video_layout.addLayout(self.right_video_box)

        # 让视频画面最大化扩展，占据上半部分核心空间
        main_layout.addLayout(video_layout, stretch=1)

        # 中间水平装饰分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # ----------------------------------------------------
        # 布局 B: 【最下方】复合控制底部面板（QTabWidget 在上，标题在下）
        # ----------------------------------------------------
        bottom_panel_layout = QHBoxLayout()
        bottom_panel_layout.setSpacing(20)

        # 底面板左侧：正方形大 LOGO 区域
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(140, 140)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("border: 1px solid #CCCCCC; background: #FAFAFA;")

        logo_pixmap = QPixmap("images/GuangDong-TaiWan.png")
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(self.logo_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
        else:
            self.logo_label.setText("LOGO\n未找到")

        # 底面板右侧：垂直控制箱
        right_controls_box = QVBoxLayout()
        right_controls_box.setSpacing(12)

        # 参数配置选项卡组件
        self.param_tabs = QTabWidget()
        self.param_tabs.setFont(QFont("Microsoft YaHei", 10))

        # 选项卡 1: YOLO 模型配置
        self.yolo_tab = QWidget()
        yolo_layout = QFormLayout(self.yolo_tab)
        yolo_layout.setContentsMargins(10, 10, 10, 10)
        self.combo_yolo_type = QComboBox()
        self.combo_yolo_type.addItems(["YOLO26-目标侦测", "YOLO26-关键点", "YOLO26-目标追踪"])
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self.on_conf_changed)
        yolo_layout.addRow("教学任务:", self.combo_yolo_type)
        yolo_layout.addRow("置信度阈值 (Conf):", self.slider_conf)

        # 选项卡 2: MediaPipe 配置
        self.mp_tab = QWidget()
        mp_layout = QFormLayout(self.mp_tab)
        mp_layout.setContentsMargins(10, 10, 10, 10)
        self.combo_mp_type = QComboBox()
        self.combo_mp_type.addItems(["MediaPipe-手势识别", "MediaPipe-静态姿势分类"])
        self.btn_calibrate = QPushButton("静态姿态基准校准 (Calibrate)")
        mp_layout.addRow("组件任务:", self.combo_mp_type)
        mp_layout.addRow("校准动作:", self.btn_calibrate)

        self.param_tabs.addTab(self.yolo_tab, "YOLO 深度学习系列配置")
        self.param_tabs.addTab(self.mp_tab, "MediaPipe 管道系列配置")
        right_controls_box.addWidget(self.param_tabs)

        # 中心名称作为底部页脚签名
        title_label = QLabel("粤台产业科技学院 - 计算机视觉研究中心")
        title_label.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        title_label.setStyleSheet("color: #1A365D; padding-top: 2px;")
        right_controls_box.addWidget(title_label)

        # 组合底部的 LOGO 和 右侧控制区
        bottom_panel_layout.addWidget(self.logo_label)
        bottom_panel_layout.addLayout(right_controls_box)

        main_layout.addLayout(bottom_panel_layout)

    # ==========================================
    # 【单元一新增】用于静态展示效果的图片加载函数
    # ==========================================
    def load_static_test_images(self):
        """
        在没有接入后台多线程流时，加载本地测试图，
        向学生直观展示自适应响应式布局的最终呈现。
        """
        raw_test = QPixmap("images/source_test.jpg")
        processed_test = QPixmap("images/result_test.jpg")

        if not raw_test.isNull():
            self.view_raw.setPixmap(raw_test)
        if not processed_test.isNull():
            self.view_processed.setPixmap(processed_test)

    # ==========================================
    # 3. 控制层：状态响应槽函数
    # ==========================================
    @Slot(QImage)
    def update_raw_view(self, q_img):
        self.view_raw.setPixmap(QPixmap.fromImage(q_img))

    @Slot(QImage)
    def update_processed_view(self, q_img):
        self.view_processed.setPixmap(QPixmap.fromImage(q_img))

    def on_conf_changed(self, value):
        float_conf = value / 100.0
        # 【单元一调试提示】由于后台线程此时未实例化，此处仅做界面事件监听打印
        print(f"UI 调试：当前方案四的滑块值已改变为: {float_conf}")

    def closeEvent(self, event):
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = VisionWorkstationGUI()
    window.show()
    sys.exit(app.exec())