# qthreads/tasks/simple_video_task.py
"""SimpleVideoTask — 灰度转换示例任务（PPT 教学示例）"""

import cv2
import numpy as np
from qthreads.tasks.base_task import BaseCVTask


class SimpleVideoTask(BaseCVTask):
    """将输入帧转为灰度图的简单演示任务。"""

    def process(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 转回 BGR 三通道保持显示兼容
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        info = {"status": "Grayscale conversion", "channels": 1}
        return gray_bgr, info
