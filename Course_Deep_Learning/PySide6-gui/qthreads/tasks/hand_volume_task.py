# qthreads/tasks/hand_volume_task.py
"""HandVolumeTask — 手势音量控制任务"""

import cv2
import numpy as np
from qthreads.tasks.base_task import BaseCVTask


class HandVolumeTask(BaseCVTask):
    """手势音量控制：基于 HandModule 检测手部，计算拇指-食指间距控制音量。"""

    def __init__(self):
        super().__init__()
        self.hand_module = None

    def load_model(self):
        try:
            from utils.HandModule import HandModule
            self.hand_module = HandModule()
        except Exception as e:
            print(f"HandModule 加载失败: {e}")
            self.hand_module = None

    def release_model(self):
        self.hand_module = None

    def process(self, frame: np.ndarray):
        if self.hand_module:
            try:
                return self.hand_module.process(frame)
            except Exception as e:
                print(f"HandVolumeTask 处理异常: {e}")
        # 无模块时的降级显示
        h, w = frame.shape[:2]
        cv2.putText(frame, "HandModule not loaded", (w//4, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        info = {"status": "Hand model not loaded", "volume": 0.0}
        return frame, info
