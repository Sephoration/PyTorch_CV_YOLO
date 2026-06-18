import cv2
from .base_task import BaseCVTask

class SimpleVideoTask(BaseCVTask):
    def __init__(self):
        super().__init__()

    def process(self, frame):
        """
        输入: OpenCV 的 BGR 矩阵图像 (来自 base_worker)
        输出: (处理后的图像, 传给 UI 的业务状态数据)
        """
        # 核心演算法：将彩色图像转换为灰度图 (实现简单处理)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 关键点：因为前端只认 BGR/RGB 3通道图像，单通道灰度图必须转换回3通道形式才能打包
        processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # 返回处理后的矩阵，并打包一个业务状态字典反馈给页脚
        info_dict = {"status": "复古灰度滤镜正在运行中..."}

        return processed_frame, info_dict

