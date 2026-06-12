# qthreads/tasks/hand_volume_task.py
"""HandVolumeTask — 手势音量控制任务（PPT P4 第二部分完整实现）

PPT P4 修改要点:
  修改点 1: 剔除原脚本的 while True 循环与硬编码设备初始化，剥离 I/O 控制权
  修改点 2: 迁移音频设备初始化至 init_audio_system()，在独立线程内安全调用 CoInitialize
  修改点 3: 延迟加载 (Lazy Initialization) — HandDetector 在 process() 首帧才实例化，
           通过 getattr() 一站式中转前台的超参数，提供安全默认值防止 AttributeError
  修改点 4: 前置工序确保音频系统在当前线程完成 COM 认证
  修改点 5: 指距 → 系统音量映射 + 动态音量条可视化叠加渲染
  修改点 6: 构造返回值协定 (frame, info_dict) 双元组，拆解数据/控制管道
  修改点 7: close() 中安全释放模型资源并反初始化 COM
"""

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
    """手势音量控制任务卡带。

    架构约定:
      - 不持有任何 I/O 循环 (while True / cap.read() 已剔除)
      - 仅实现 process(self, frame) → (frame, info_dict)
      - 由 BaseWorker 的 run() 统一管理帧循环与信号发射
    """

    def __init__(self):
        super().__init__()

        # 算法模块实例 — 延迟加载，首次 process() 时再创建
        self.detector = None

        # Windows 音频系统引用
        self.device = None
        self.volume_ctrl = None
        self.min_vol = -65
        self.max_vol = 0

        # COM 线程隔离标志
        self.com_initialized = False

        # 帧计数器（用于内部调试/状态节奏控制）
        self.frame_count = 0

    # ==================================================================
    # 修改点 2: Windows COM 环境初始化 — 必须在 process() 所在线程调用
    # ==================================================================
    def init_audio_system(self):
        """初始化 Windows 系统音量控制接口（在线程内安全执行）。

        说明 (PPT Slide 23):
          在 Windows 操作系统中，任何线程若试图直接去操控系统音频硬件设备，
          必须在当前线程内部先调用 CoInitialize() 完成底层 COM 运行环境初始化。
          若在构造线程 (__init__) 中预初始化 COM，而后台工作线程频繁跨线程调用
          SetMasterVolumeLevel()，Windows 系统会判定为非法跨线程越权访问，直接崩溃。
        """
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
                print(f"pycaw 音频模块初始化失败: {e}")

    # ==================================================================
    # 修改点 3: 延迟加载 — 算法模块在首帧处理时才实例化
    # ==================================================================
    def _ensure_detector(self):
        """延迟加载 HandDetector，桥接前台 UI 下发的超参数。

        使用 getattr() 而非直接点号访问的原因 (PPT Slide 28):
          多线程热插拔框架下，无法保证前台 UI 已经将超参数同步写到 self 上。
          使用 self.detection_con 时，若属性尚不存在，Python 解释器会抛出
          AttributeError 导致后台线程瞬间崩溃。
          getattr(self, 'detection_con', 0.5) 提供了安全默认值作为兜底，
          相当于一个安全气垫——即使属性不存在也平稳返回 0.5 继续执行。
        """
        if self.detector is None:
            self.detector = HandDetector(
                mode=getattr(self, 'running_mode', 'VIDEO'),
                num_hands=getattr(self, 'max_targets', 1),
                detectionCon=getattr(self, 'detection_con', 0.5),
                presenceCon=getattr(self, 'presence_con', 0.5),
                trackingCon=getattr(self, 'tracking_con', 0.5)
            )

    # ==================================================================
    # 修改点 4 & 5: 核心处理流水线
    # ==================================================================
    def process(self, frame: np.ndarray):
        """处理单帧图像 — 手部检测 + 音量控制。

        Returns:
            (processed_frame, info_dict)
        """
        self.frame_count += 1

        # ---- 前置工序: 确保音频系统在当前线程内完成 COM 认证 ----
        self.init_audio_system()

        # ---- 延迟加载: 首次调用时动态实例化 HandDetector ----
        self._ensure_detector()

        # ---- 1. 手部检测 ----
        frame = self.detector.findHands(frame, draw=True)
        lmList = self.detector.findPosition(frame, draw=False)

        vol_per = 0
        status_text = "等待手掌进入..."

        # ---- 2. 检测到手部关键点 ----
        if len(lmList) != 0:
            # 拇指指尖 (ID 4) 与 食指指尖 (ID 8) 坐标
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 绘制交互视觉元素
            cv2.circle(frame, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 计算指尖欧氏距离
            length = math.hypot(x2 - x1, y2 - y1)

            # 距离极小时给予视觉反馈
            if length < 30:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # 修改点 5: 指距映射 → 系统音量值 + 可视化百分比
            # Hand Range: 20 ~ 200  →  Volume Range: min_vol ~ max_vol
            vol_target = np.interp(length, [20, 200], [self.min_vol, self.max_vol])
            vol_bar_y = np.interp(length, [20, 200], [400, 150])
            vol_per = int(np.interp(length, [20, 200], [0, 100]))

            status_text = f"手势音量控制: {vol_per}%"

            # 线程安全地向 Windows 系统提交音量变更
            if PYCAW_AVAILABLE and self.volume_ctrl:
                try:
                    self.volume_ctrl.SetMasterVolumeLevel(vol_target, None)
                except Exception as e:
                    status_text = "系统音量调节异常"

            # ---- 音量条可视化叠加渲染 ----
            # 标签
            cv2.putText(frame, 'VOL', (28, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # 外框
            cv2.rectangle(frame, (30, 150), (80, 400), (255, 255, 255), 2)
            # 动态填充条
            cv2.rectangle(frame, (30, int(vol_bar_y)), (80, 400),
                          (255, 0, 255), cv2.FILLED)
            # 百分比文字
            cv2.putText(frame, f'{int(vol_per)}%', (22, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # ---- 3. 修改点 6: 打包返回值协定 (数据/控制管道解耦) ----
        info_dict = {"volume": vol_per, "status": status_text}
        return frame, info_dict

    # ==================================================================
    # 修改点 7: 安全释放资源 + COM 反初始化
    # ==================================================================
    def release_model(self):
        """释放 HandDetector 模型资源与 COM 音频接口。

        此方法由 BaseWorker 在 switch_task / stop 时调用，
        确保资源在后台线程内部安全释放，而非跨线程操作。
        """
        if self.detector:
            self.detector.close()
            self.detector = None

        if PYCAW_AVAILABLE and self.com_initialized:
            try:
                CoUninitialize()
            except Exception:
                pass
            self.com_initialized = False
