# qthreads/tasks/hand_volume_task.py
import cv2
import math
import numpy as np

from qthreads.tasks.base_task import BaseCVTask
from utils.HandModule import HandDetector

try:
    from comtypes import CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False


class HandVolumeTask(BaseCVTask):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.volume_ctrl = None
        self.device = None
        self.min_vol = -65
        self.max_vol = 0
        self.com_initialized = False

    def init_audio_system(self):
        """ 初始化 Windows 系统音量控制 (在子线程内安全执行) """
        if PYCAW_AVAILABLE and self.volume_ctrl is None:
            try:
                CoInitialize()
                self.com_initialized = True
                self.device = AudioUtilities.GetSpeakers()
                self.volume_ctrl = self.device.EndpointVolume
                vol_range = self.volume_ctrl.GetVolumeRange()
                self.min_vol = vol_range[0]
                self.max_vol = vol_range[1]
            except Exception as e:
                print(f"【音频提示】pycaw 音量模块初始化失败: {e}")

    def process(self, frame):
        # 核心视频帧深度检测管道
        # 拦截切换硬件源时的过渡空帧
        if frame is None:
            return frame, {"volume": 0, "status": "摄像头预热中或无画面输入..."}

        try:
            # ====================================================================
            # 🎯 延迟加载 (Lazy Initialization):
            # 确保在使用前，截取到主界面 UI 最新同步过来的 MediaPipe 全局参数
            # ====================================================================
            if self.detector is None:
                self.detector = HandDetector(
                    mode=getattr(self, 'running_mode', 'VIDEO'),
                    num_hands=getattr(self, 'max_targets', 1),
                    detectionCon=getattr(self, 'detection_con', 0.5),
                    presenceCon=getattr(self, 'presence_con', 0.5),
                    trackingCon=getattr(self, 'tracking_con', 0.5)
                )

            # 1. 提取手势骨架
            frame = self.detector.findHands(frame, draw=True)
            lmList = self.detector.findPosition(frame, draw=False)

            vol_per = 0
            status_text = "等待手势输入..."

            # 2. 如果检测到手掌骨骼点
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(frame, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                length = math.hypot(x2 - x1, y2 - y1)
                if length < 30:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                vol_per = int(np.interp(length, [20, 200], [0, 100]))
                status_text = f"调试模式(音频暂离): 模拟音量 {vol_per}%"

                # =============================================================
                # 💡 【核心排查隔离带】：暂时断开真实硬件音频控制，观察画面是否恢复
                # =============================================================
                self.init_audio_system()
                if PYCAW_AVAILABLE and self.volume_ctrl:
                    try:
                        vol_target = np.interp(length, [20, 200], [self.min_vol, self.max_vol])
                        self.volume_ctrl.SetMasterVolumeLevel(vol_target, None)
                        status_text = f"真实音量控制中: {vol_per}%"
                    except Exception as e:
                        status_text = "系统音量调节异常"

                # 绘制 UI 矩形条
                cv2.putText(frame, 'VOL', (28, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(frame, (30, 150), (80, 400), (255, 255, 255), 2)
                vol_bar_y = np.interp(length, [20, 200], [400, 150])
                cv2.rectangle(frame, (30, int(vol_bar_y)), (80, 400), (255, 0, 255), cv2.FILLED)
                cv2.putText(frame, f'{int(vol_per)}%', (22, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            return frame, {"volume": vol_per, "status": status_text}

        except Exception as err:
            # 🎯 防线 3：捕获内部所有未知突发崩溃，拒绝静默退出
            print(f"【严重错误】HandVolumeTask 内部运行时崩溃: {err}")
            return frame, {"volume": 0, "status": f"算法核心报错: {err}"}

    def close(self):
        if self.detector:
            self.detector.close()
        if PYCAW_AVAILABLE and self.com_initialized:
            try:
                CoUninitialize()
            except:
                pass