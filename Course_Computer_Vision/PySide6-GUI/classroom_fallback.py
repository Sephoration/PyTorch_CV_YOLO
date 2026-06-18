# classroom_fallback.py
import cv2
import math
import numpy as np
import time

# 导入你修改好的核心驱动
from utils.HandModule import HandDetector

# 尝试载入音频控制库
try:
    from comtypes import CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities

    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False
    print("【提示】未检测到 pycaw 库，将自动进入虚拟音量演示模式。")


def main():
    print("====================================================")
    print("       计算机视觉课堂演示 - 手势音量控制保底系统      ")
    print("====================================================")

    # 1. 初始化系统音频接口
    volume_ctrl = None
    min_vol, max_vol = -65, 0
    if PYCAW_AVAILABLE:
        try:
            CoInitialize()
            device = AudioUtilities.GetSpeakers()
            volume_ctrl = device.EndpointVolume
            vol_range = volume_ctrl.GetVolumeRange()
            min_vol, max_vol = vol_range[0], vol_range[1]
            print("✨ Windows 硬件音频控制系统对接成功！")
        except Exception as e:
            print(f"⚠️ 音频硬件驱动初始化失败: {e}，切换为模拟模式。")
            volume_ctrl = None

    # 2. 初始化物理相机 (Windows 下强拉 DSHOW 后端，秒开且绝对不卡死)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 错误：未检测到任何可用的物理摄像头，请检查硬件连接！")
        return

    # 3. 实例化你的 MediaPipe 核心检测器
    detector = HandDetector(mode="VIDEO", num_hands=1, detectionCon=0.5, trackingCon=0.5)
    print("🚀 MediaPipe 手部智能骨架模型加载就绪，准备开播...")

    p_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ 警告：无法读取当前帧画面")
            continue

        # 核心算法处理
        frame = detector.findHands(frame, draw=True)
        lmList = detector.findPosition(frame, draw=False)

        vol_per = 0
        if len(lmList) != 0:
            # 获取大拇指 (ID 4) 和食指 (ID 8) 坐标
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 绘制骨架连线和特征点
            cv2.circle(frame, (x1, y1), 6, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 6, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 计算两指物理间距
            length = math.hypot(x2 - x1, y2 - y1)

            # 捏合触觉反馈
            if length < 30:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # 线性映射音量数据
            vol_per = int(np.interp(length, [20, 180], [0, 100]))

            # 动态控制真实硬件系统音量
            if volume_ctrl:
                try:
                    vol_target = np.interp(length, [20, 180], [min_vol, max_vol])
                    volume_ctrl.SetMasterVolumeLevel(vol_target, None)
                except Exception:
                    pass

            # 绘制上课演示专用 UI 矩形条
            cv2.rectangle(frame, (30, 150), (60, 380), (255, 255, 255), 2)
            vol_bar_y = np.interp(length, [20, 180], [380, 150])
            cv2.rectangle(frame, (30, int(vol_bar_y)), (60, 380), (0, 255, 255), cv2.FILLED)
            cv2.putText(frame, f'{vol_per}%', (25, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 计算并打印实时 FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) != 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # 渲染窗口
        cv2.imshow("Classroom Live Demo (Press ESC to Exit)", frame)

        # 监听退出按键 (ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    if PYCAW_AVAILABLE:
        try:
            CoUninitialize()
        except:
            pass
    print("演示结束，设备安全释放。")


if __name__ == "__main__":
    main()