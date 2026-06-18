# qthreads/tasks/base_task.py
class BaseCVTask:
    """
    所有视觉应用插件的抽象基类。
    未来不管是人脸、车牌还是各种 AI 模型，只要继承它，就能直接塞进工作站。
    """
    def __init__(self):
        # 预留通用的初始化参数
        self.confidence_threshold = 0.5

    def update_params(self, value):
        """用于接收主界面 Slider 等组件传过来的超参数"""
        self.confidence_threshold = value

    def process(self, frame):
        """
        核心演算法接口（子类必须重写它）。
        输入: OpenCV 的 BGR 矩阵图像
        输出: (处理后的BGR图像, 想要传给UI显示的数据字典或字符串)
        """
        # 基类默认什么都不做，原样返回（代表纯视频流）
        return frame, {}