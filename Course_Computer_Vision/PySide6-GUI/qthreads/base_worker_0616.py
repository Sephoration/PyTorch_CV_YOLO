# qthreads/base_worker.py
import random
import time
import cv2
import threading  # 🎯 新增：导入系统级线程互斥锁
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from .tasks.base_task import BaseCVTask
from enum import Enum

class SourceType(Enum):
    CAMERA = 0
    VIDEO_FILE = 1
    IMAGE_FOLDER = 2

class BaseWorker(QThread):
    raw_frame_signal = Signal(QImage)
    processed_frame_signal = Signal(QImage)
    data_signal = Signal(dict)

    def __init__(self, source=0, source_type=SourceType.CAMERA):
        super().__init__()
        self.source = source
        self.source_type = source_type
        self.cap = None
        self.image_paths = []  # 如果是 IMAGE_FOLDER，这里存所有图片路径
        self.current_image_idx = 0

        self.fps_delay = 0.03  # 控制播放速度的煞車變數 (預設 0.03 秒)
        self.is_running = True
        self.enable_raw_stream = True
        self.source_changed = False
        self.current_task = BaseCVTask()

    def _init_media_source(self):
        """根据媒体类型进行适配初始化"""
        if self.source_type == SourceType.CAMERA:
            self.cap = cv2.VideoCapture(self.source)
            self.fps_delay = 0.03  # 攝影機通常抓 30fps
        elif self.source_type == SourceType.VIDEO_FILE:
            self.cap = cv2.VideoCapture(self.source)
            # 自動讀取影片檔案的真實 FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.fps_delay = 1.0 / fps  # 例如 30fps 就是 1/30 = 0.033秒
            else:
                self.fps_delay = 0.03
        elif self.source_type == SourceType.IMAGE_FOLDER:
            # 扫描活页夹内所有图片
            path_obj = Path(self.source)
            exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
            self.image_paths = []
            for ext in exts:
                self.image_paths.extend(path_obj.rglob(ext))
            self.image_paths = [str(p) for p in self.image_paths]
            random.shuffle(self.image_paths)  # 隨機打亂
            self.current_image_idx = 0
            # 靜態圖片我們給它 0.8 秒的停頓，方便看清楚檢測結果
            self.fps_delay = 0.8

    def switch_task(self, task_instance):
        if hasattr(self.current_task, "close"):
            try:
                self.current_task.close()
            except Exception as e:
                print(f"【后台提示】关闭旧模型任务异常: {e}")

        self.current_task = task_instance
        print(f"【后台提示】已成功切换视觉算法插件为: {task_instance.__class__.__name__}")

    def change_media_source(self, new_source, new_type=SourceType.CAMERA):
        """支持在运行时无缝切换媒体源和类型"""
        self.source = new_source
        self.source_type = new_type
        self.source_changed = True  # 立起 Flag，通知主循环该换挡了
        print(f"【后台提示】媒体源准备热切为 -> {new_source}, 类型 -> {new_type.name}")

    def set_raw_stream_enabled(self, enabled: bool):
        self.enable_raw_stream = enabled

    # def _open_video_capture(self, source):
    #     """辅助函数：根据媒体源类型，自动匹配最安全的 OpenCV 驱动后端"""
    #     if isinstance(source, int):
    #         print(f"【雷达诊断】正在拉起 Windows 物理相机硬件, ID = {source}...")
    #         return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    #     else:
    #         print(f"【雷达诊断】正在加载本地视频文件, 路径 = {source}...")
    #         return cv2.VideoCapture(source)

    def _cv2_to_qimage(self, frame):
        """统一的图像转换辅助函数"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        return QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

    def run(self):
        self._init_media_source()

        while self.is_running:
            # ==================================================
            # ⚡ 核心新增：在每一帧循环前，检查是否需要“换挡”
            # ==================================================
            if self.source_changed:
                if self.cap:
                    self.cap.release()  # 物理释放旧的媒体流
                self._init_media_source() # 重新初始化新的媒体流
                self.source_changed = False
                continue # 换挡完毕，跳过这一帧，重新开始下一轮循环

            frame = None

            # 1. 媒体读取策略
            if self.source_type in [SourceType.CAMERA, SourceType.VIDEO_FILE]:
                if self.cap and self.cap.isOpened():
                    success, frame = self.cap.read()
                    if not success:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
            elif self.source_type == SourceType.IMAGE_FOLDER and self.image_paths:
                frame = cv2.imread(self.image_paths[self.current_image_idx])
                self.current_image_idx = (self.current_image_idx + 1) % len(self.image_paths)

            # 2. 推理与信号发射
            if frame is not None:
                # 发送原始画面
                if self.enable_raw_stream:
                    self.raw_frame_signal.emit(self._cv2_to_qimage(frame))

                # 执行任务 (Task)
                if self.current_task:
                    p_frame, info_dict = self.current_task.process(frame)

                    # 发送处理后画面
                    self.processed_frame_signal.emit(self._cv2_to_qimage(p_frame))

                    # 发送处理后的数据 (如果存在)
                    if info_dict:
                        self.data_signal.emit(info_dict)

                # ==================================================
                # 🛑 核心新增：統一幀率控制 (FPS Throttling)
                # ==================================================
                time.sleep(self.fps_delay)

        # 3. 资源彻底释放 (确保在 is_running 变为 False 后执行)
        if self.cap:
            self.cap.release()
        if hasattr(self.current_task, "close"):
            self.current_task.close()

    def stop(self):
        self.is_running = False
        self.wait()