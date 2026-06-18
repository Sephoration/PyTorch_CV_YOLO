# qthreads/tasks/hand_volumn_task.py
import cv2
import math
import numpy as np

# 导入抽象基类（确保您的项目中已有此文件）
from qthreads.tasks.base_task import BaseCVTask
# 导入我们刚刚封装好的最新手部核心驱动
from utils.HandModule import HandDetector

# 引入 Windows 平台系统音量控制器接口 (pycaw)
try:
    from comtypes import CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False


class HandVolumeTask(BaseCVTask):
    """
    MediaPipe 手势音量控制任务插件 (高性能异步版)
    """

    def __init__(self):
        super().__init__()
        self.detector = None
        # 音量控制硬件相关变量
        self.device = None
        self.volume_ctrl = None
        self.min_vol = -65
        self.max_vol = 0
        self.com_initialized = False

    def init_audio_system(self):
        """
        初始化 Windows 系统音量控制 (在子线程内安全执行)
        """
        if PYCAW_AVAILABLE and self.volume_ctrl is None:
            try:
                # 在独立的多线程环境中调用 Windows 硬件 API，必须先初始化 COM 库
                CoInitialize()
                self.com_initialized = True

                self.device = AudioUtilities.GetSpeakers()
                self.volume_ctrl = self.device.EndpointVolume
                vol_range = self.volume_ctrl.GetVolumeRange()
                self.min_vol = vol_range[0]
                self.max_vol = vol_range[1]
            except Exception as e:
                print(f"pycaw 音量模块初始化失败: {e}")

    def process(self, frame):
        """
        核心视频帧深度检测管道
        """
        # ====================================================================
        # 🎯 延迟加载 (Lazy Initialization):
        # 确保在使用前，截取到主界面 UI 最新同步过来的 MediaPipe 全局参数
        # ====================================================================
        if self.detector is None:
            self.detector = HandDetector(
                mode=getattr(self, 'running_mode', 'VIDEO'),
                num_hands=getattr(self, 'max_targets', 2),
                detectionCon=getattr(self, 'detection_con', 0.5),
                presenceCon=getattr(self, 'presence_con', 0.5),
                trackingCon=getattr(self, 'tracking_con', 0.5)
            )

        # 确保音频控制库在当前线程内可用
        self.init_audio_system()

        # 1. 调用 HandDetector 提取手势骨架
        frame = self.detector.findHands(frame, draw=True)
        lmList = self.detector.findPosition(frame, draw=False)

        vol_per = 0
        status_text = "等待手势输入..."

        # 2. 如果检测到手掌骨骼点
        if len(lmList) != 0:
            # 提取拇指指尖 (ID 4) 和 食指指尖 (ID 8) 的坐标
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 在图像上绘制可视化连接线与交互提示点
            cv2.circle(frame, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 计算两指距离
            length = math.hypot(x2 - x1, y2 - y1)

            # 当两指距离极近时，改变中心点颜色以提供触控反馈
            if length < 30:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # 将手指的物理距离 [20, 200] 映射到系统音量分贝值以及界面百分比
            vol_target = np.interp(length, [20, 200], [self.min_vol, self.max_vol])
            vol_bar_y = np.interp(length, [20, 200], [400, 150])
            vol_per = int(np.interp(length, [20, 200], [0, 100]))

            status_text = f"音量控制中: {vol_per}%"

            # 跨多线程安全、非阻塞地更新系统主音量
            if PYCAW_AVAILABLE and self.volume_ctrl:
                try:
                    self.volume_ctrl.SetMasterVolumeLevel(vol_target, None)
                except Exception as e:
                    status_text = "系统音量调节异常"

            # 顶部标题
            cv2.putText(frame, 'VOL', (28, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # 音量条外框
            cv2.rectangle(frame, (30, 150), (80, 400), (255, 255, 255), 2)
            # 音量条动态填充
            cv2.rectangle(frame, (30, int(vol_bar_y)), (80, 400), (255, 0, 255), cv2.FILLED)
            # 百分比显示
            cv2.putText(frame, f'{int(vol_per)}%', (22, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # 完美返回：处理后的画面，以及传给 UI 底栏的文字数据
        return frame, {"volume": vol_per, "status": status_text}

    def close(self):
        """
        安全释放生命周期资源：关闭底层模型并卸载 COM 音频驱动连接
        """
        if self.detector:
            self.detector.close()

        if PYCAW_AVAILABLE and self.com_initialized:
            try:
                CoUninitialize()
            except:
                pass