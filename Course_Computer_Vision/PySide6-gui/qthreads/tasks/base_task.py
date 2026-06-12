# qthreads/tasks/base_task.py
"""BaseCVTask — 算法卡带抽象基类（PPT 第二部分：模组化工件架构）"""

import abc
import cv2
import numpy as np


class BaseCVTask(abc.ABC):
    """所有视觉算法任务必须继承此基类，实现 process() 方法。

    设计理念（PPT Slide 12-23）：
      每个 task 像游戏卡带一样即插即用，
      BaseWorker 不关心算法细节，只调用 process()。
    """

    def __init__(self):
        self._params = {}

    def set_param(self, key: str, value):
        """动态注入超参数（PPT 描述的灵活性关键）"""
        self._params[key] = value

    def get_param(self, key: str, default=None):
        return self._params.get(key, default)

    def load_model(self):
        """子类可重写：预加载模型资源"""
        pass

    def release_model(self):
        """子类可重写：释放模型资源"""
        pass

    @abc.abstractmethod
    def process(self, frame: np.ndarray):
        """处理一帧图像。

        Args:
            frame: OpenCV BGR 图像 (numpy.ndarray)

        Returns:
            (processed_frame, info_dict)
              - processed_frame: BGR 格式输出图像
              - info_dict: 业务数据 dict（检测结果、统计数据等）
        """
        raise NotImplementedError
