import time
import numpy as np
import cv2
import math
import HandModule as htm
from pycaw.pycaw import AudioUtilities

###############################
wCam, hCam = 640, 480
###############################

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

pTime = 0
if __name__ == '__main__':
    detector = htm.HandDetector()

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            continue

        # ------ 音量条 ------
        # 顶部标题
        cv2.putText(frame, 'VOL', (28, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # 音量条外框
        cv2.rectangle(frame, (30, 150), (80, 400), (255, 255, 255), 2)

        # 音量条填充
        volBar = np.interp(volume.GetMasterVolumeLevel(), [-65, 0], [400, 150])
        cv2.rectangle(frame, (30, int(volBar)), (80, 400), (255, 0, 255), cv2.FILLED)
        # 百分比显示
        volPer = np.interp(volume.GetMasterVolumeLevel(), [-65, 0], [0, 100])
        cv2.putText(frame, f'{int(volPer)}%', (22, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)

        if (len(lmList) != 0):
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
            # print(length)

            # Hand Range: 20-200
            # Volume Range: -65 - 0
            vol = np.interp(length, [20, 200], [minVol, maxVol])
            volBar = np.interp(length, [20, 200], [400, 150])
            volPer = np.interp(length, [20, 200], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            # 音量条填充
            cv2.rectangle(frame, (30, int(volBar)), (80, 400), (255, 0, 255), cv2.FILLED)

            # 百分比显示
            cv2.putText(frame, f'{int(volPer)}%', (22, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # ------ FPS ------
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('HandTracking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()