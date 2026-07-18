# qthreads/base_worker.py
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .tasks.base_task import BaseCVTask


class BaseWorker(QThread):
    raw_frame_signal = Signal(QImage)
    processed_frame_signal = Signal(QImage)
    data_signal = Signal(dict)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.is_running = True
        self.enable_raw_stream = True
        self.source_changed = False  # 媒体源切换标志
        self.current_task = BaseCVTask()

    def switch_task(self, task_instance):
        if hasattr(self.current_task, "close"):
            try:
                self.current_task.close()
            except Exception as e:
                print(f"关闭旧模型任务异常: {e}")

        self.current_task = task_instance
        print(f"后台线程：已成功切换视觉算法插件为: {task_instance.__class__.__name__}")

    def change_media_source(self, new_source):
        """支持在运行时无缝把镜头从 MP4 切换到 WebCam(0)"""
        self.camera_id = new_source
        self.source_changed = True
        print(f"后台线程：媒体源已准备切换为 -> {new_source}")

    def set_raw_stream_enabled(self, enabled: bool):
        self.enable_raw_stream = enabled

    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        while self.is_running:
            # 动态热切换相机或视频源
            if self.source_changed:
                cap.release()
                cap = cv2.VideoCapture(self.camera_id)
                self.source_changed = False

            success, frame = cap.read()
            if not success:
                # 影片读到结尾时自动回放 (Loop)
                if isinstance(self.camera_id, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self.msleep(10)
                continue

            if self.enable_raw_stream:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                q_img_raw = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.raw_frame_signal.emit(q_img_raw.copy())

            processed_frame, info_dict = self.current_task.process(frame.copy())

            p_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            p_h, p_w, p_ch = p_frame_rgb.shape
            q_img_processed = QImage(p_frame_rgb.data, p_w, p_h, p_ch * p_w, QImage.Format_RGB888)
            self.processed_frame_signal.emit(q_img_processed.copy())

            if info_dict:
                self.data_signal.emit(info_dict)

            self.msleep(30)

        cap.release()
        if hasattr(self.current_task, "close"):
            self.current_task.close()

    def update_global_params(self, max_targets, detect_con, presence_con, track_con, pose_complexity, running_mode):
        self.current_task.max_targets = max_targets
        self.current_task.detection_con = detect_con
        self.current_task.presence_con = presence_con
        self.current_task.tracking_con = track_con
        self.current_task.pose_complexity = pose_complexity
        self.current_task.running_mode = running_mode

        if hasattr(self.current_task, "update_global"):
            self.current_task.update_global(max_targets, detect_con, presence_con, track_con, pose_complexity,
                                            running_mode)

    def update_task_params(self, param_dict):
        if hasattr(self.current_task, "update_special_params"):
            self.current_task.update_special_params(param_dict)

    def set_parameters(self, conf):
        if hasattr(self.current_task, "update_params"):
            self.current_task.update_params(conf)

    def stop(self):
        self.is_running = False
        self.wait()