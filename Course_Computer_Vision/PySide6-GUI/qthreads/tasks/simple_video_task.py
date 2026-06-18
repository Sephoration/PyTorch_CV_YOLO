# qthreads/tasks/simple_video_task.py
import cv2
from .base_task import BaseCVTask

class SimpleVideoTask(BaseCVTask):
    """
    教学小单元专用：简单视讯处理卡带。
    负责将传入的彩色画面转换为复古灰度图，用于验证双视窗分发流。
    """

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

    def close(self):
        """
        显式释放当前任务占用的资源。
        """
        # 1. 打印日志，让学生在终端看到资源释放的时刻
        print("【资源释放】SimpleVideoTask 已销毁，灰度滤镜资源已归还。")

        # 2. 如果未来在这个类里定义了大型缓存变量，可以在这里清空
        # self.buffer_cache = None

        # 3. 规范写法：调用父类的 close (如果基类中有定义)
        # super().close()