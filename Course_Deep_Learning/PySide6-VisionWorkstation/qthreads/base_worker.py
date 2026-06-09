# qthreads/base_worker.py
"""BaseWorker — QThread 异步视觉工作线程（PPT 第二部分核心）

架构设计（PPT Slide 14-19）：
  - 前端（GUI 线程）：按钮交互、画面渲染 —— 不能阻塞（60 FPS ≈ 16.6ms/帧）
  - 后端（工作线程）：AI 推理（YOLO/MediaPipe 20~50ms/帧）
  - 通讯桥梁：Signal & Slot（Qt 唯一线程安全的数据传递方式）
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QImage


class BaseWorker(QThread):
    """异步视觉工作线程。

    Signals:
        raw_frame_signal(QImage):       原始帧信号 → 左画面
        processed_frame_signal(QImage):  处理后帧信号 → 右画面
        data_signal(dict):              业务数据信号 → 状态栏/统计面板
    """

    raw_frame_signal = Signal(QImage)
    processed_frame_signal = Signal(QImage)
    data_signal = Signal(dict)

    def __init__(self, src=0, task=None, parent=None):
        super().__init__(parent)
        self._src = src                 # 视频源：0=摄像头 / 文件路径
        self._task = task               # 当前算法卡带（BaseCVTask 实例）
        self._running = False
        self._mutex = QMutex()
        self._frame_skip = 1            # 跳帧计数，性能调节用

    def switch_task(self, task):
        """热切换算法卡带（PPT 模组化设计关键）"""
        with QMutexLocker(self._mutex):
            if self._task:
                self._task.release_model()
            self._task = task
            if self._task:
                self._task.load_model()

    def switch_source(self, src):
        """动态切换视频源"""
        self._src = src

    def stop(self):
        """安全停止线程"""
        self._running = False
        self.wait()

    # ---------- 图像转换管道（PPT Slide 核心） ----------
    @staticmethod
    def cvframe_to_qimage(bgr_frame: np.ndarray) -> QImage:
        """OpenCV BGR → QImage（RGB888）"""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        return QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    def run(self):
        """主循环"""
        self._running = True
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            self.data_signal.emit({"status": f"无法打开视频源: {self._src}"})
            self._running = False
            return

        frame_count = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                # 视频文件播放完毕 → 自动重播（PPT 要求）
                if isinstance(self._src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1

            # 【线程安全】深拷贝后发出原始帧
            raw_copy = frame.copy()
            q_img_raw = self.cvframe_to_qimage(raw_copy)
            # 再次拷贝避免 Signal 传递时的缓冲区竞争
            self.raw_frame_signal.emit(q_img_raw.copy())

            # 若有算法卡带 → 处理帧
            with QMutexLocker(self._mutex):
                task = self._task

            if task:
                try:
                    processed, info = task.process(frame)
                except Exception as e:
                    processed = frame.copy()
                    info = {"status": f"Task error: {e}"}

                q_img_proc = self.cvframe_to_qimage(processed)
                self.processed_frame_signal.emit(q_img_proc.copy())
                self.data_signal.emit(info)
            else:
                # 无任务时，左右画面一致
                self.processed_frame_signal.emit(q_img_raw.copy())
                self.data_signal.emit({"status": "未加载算法任务"})

            # 控制帧率
            self.msleep(30)  # ~33 FPS

        cap.release()
