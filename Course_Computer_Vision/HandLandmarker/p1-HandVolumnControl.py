import time
import math
import cv2
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ====== 1) 初始化音量控制 ======
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

# 获取系统音量范围
volume_range = volume.GetVolumeRange()
minVol = volume_range[0]
maxVol = volume_range[1]

print(f"[INFO] 音量范围: {minVol:.2f} ~ {maxVol:.2f} dB")

# ====== 2) 摄像头设置 ======
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# ====== 3) 建立手部检测器 ======
detector = htm.HandDetector(num_hands=1, detectionCon=0.7)

# ====== 4) FPS 相关变量 ======
pTime = 0

# ====== 主循环 ======
try:
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("[WARNING] 无法读取摄像头画面")
            continue

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame, draw=True, flip=False)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [20, 200], [minVol, maxVol])
            volBar = np.interp(length, [20, 200], [400, 150])
            volPer = np.interp(length, [20, 200], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            if length < 30:
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
            else:
                cv2.circle(frame, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, f'Distance: {int(length)}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, 'VOL', (28, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(frame, (30, 150), (80, 400), (255, 255, 255), 2)

        if 'volBar' in dir():
            volBar = max(150, min(400, volBar))
            cv2.rectangle(frame, (30, int(volBar)), (80, 400), (255, 0, 255), cv2.FILLED)

        if 'volPer' in dir():
            volPer = max(0, min(100, volPer))
            cv2.putText(frame, f'{int(volPer)}%', (22, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Volume Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("[INFO] 退出程序")
            break

finally:
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 资源已释放")