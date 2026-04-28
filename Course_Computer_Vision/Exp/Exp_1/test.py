import cv2
import time
import os
import math
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ========== 1. 图片加载 ==========
folderPath = os.path.join(os.path.dirname(__file__), "FingerImages")
myList = os.listdir(folderPath)
overlayList = []

for imgPath in sorted(myList):
    img = cv2.imread(os.path.join(folderPath, imgPath))
    if img is not None:
        overlayList.append(img)

# ========== 2. 初始化音量控制 ==========
try:
    devices = AudioUtilities.GetSpeakers()
    volume = devices.EndpointVolume
    volume_range = volume.GetVolumeRange()
    minVol = volume_range[0]
    maxVol = volume_range[1]
    volume_available = True
    print(f"[INFO] 音量控制初始化成功，范围: {minVol:.2f} ~ {maxVol:.2f} dB")
except Exception as e:
    print(f"[WARNING] 音量控制初始化失败: {e}")
    volume_available = False

# ========== 3. 摄像头初始化 ==========
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ========== 4. 手部检测器 ==========
detector = htm.HandDetector(detectionCon=0.75)

# ========== 5. 数字锁定相关变量 ==========
prev_number = -1
start_time = 0
locked_number = -1
display_number = 0
lock_duration = 2.0

# ========== 6. 图片显示变量 ==========
# 当前显示的大图片编号（锁定后保持）
current_big_image = 0

# ========== 7. 音量控制顺序锁相关变量 ==========
volume_unlock_sequence = []
required_sequence = [2, 1, 5]
volume_enabled = False
hand_present = True  # 手是否在画面中

# ========== 主循环 ==========
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    
    # 检查手是否在画面中
    hand_present = len(lmList) != 0
    
    # 如果手移出画面，关闭音量控制并重置序列
    if not hand_present and volume_enabled:
        volume_enabled = False
        volume_unlock_sequence = []
        print("[INFO] 手移出画面，音量控制已关闭")
    
    # 获取手部标签
    handLabel = None
    if detector.results and detector.results.handedness:
        handLabel = detector.results.handedness[0][0].category_name

    totalFingers = 0

    # ========== 8. 手势数字识别 ==========
    if hand_present:
        fingers = []
        tipIds = [4, 8, 12, 16, 20]

        # 拇指判断
        if handLabel == "Left":
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 其余四指判断
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

    # ========== 9. 数字锁定逻辑 ==========
    current_number = totalFingers
    current_time = time.time()

    if current_number == prev_number and current_number > 0:
        if current_number != locked_number:
            if start_time == 0:
                start_time = current_time
            elif current_time - start_time >= lock_duration:
                # 锁定新数字
                locked_number = current_number
                display_number = locked_number
                # 更新大图片（锁定后一直保持）
                current_big_image = display_number
                
                # ========== 10. 音量控制顺序解锁逻辑 ==========
                if volume_available:
                    # 检查当前锁定的数字是否是顺序中的下一个
                    next_required = required_sequence[len(volume_unlock_sequence)]
                    if locked_number == next_required:
                        volume_unlock_sequence.append(locked_number)
                        print(f"[INFO] 音量解锁进度: {volume_unlock_sequence}")
                        
                        # 检查是否完成全部顺序
                        if len(volume_unlock_sequence) == len(required_sequence):
                            volume_enabled = True
                            print("[INFO] 音量控制已解锁！(顺序: 2→1→5)")
                    else:
                        # 如果顺序错误，重置序列
                        if locked_number != next_required:
                            volume_unlock_sequence = []
                            print("[INFO] 顺序错误，音量解锁序列已重置")
    else:
        start_time = 0
        if current_number != locked_number:
            locked_number = -1
            # 注意：大图片不清除，继续保持

    prev_number = current_number

    # ========== 11. 进度条绘制 ==========
    if start_time > 0 and locked_number == -1 and current_number == prev_number and current_number > 0:
        elapsed = current_time - start_time
        progress = min(1.0, elapsed / lock_duration)
        
        bar_width = int(progress * 1280)
        cv2.rectangle(frame, (0, 680), (bar_width, 720), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (0, 680), (1280, 720), (255, 255, 255), 2)
        cv2.putText(frame, f'Hold: {int(progress * 100)}%', (20, 670),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ========== 12. 显示大图片（锁定后一直保持）==========
    if current_big_image > 0 and current_big_image <= len(overlayList):
        img = overlayList[current_big_image - 1]
        h, w, _ = img.shape
        # 放大图片，显示在右侧
        scale = 2.5
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        h_frame, w_frame, _ = frame.shape
        y_offset = (h_frame - new_h) // 2
        x_offset = w_frame - new_w - 50
        
        if y_offset >= 0 and x_offset >= 0:
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    # ========== 13. 显示角落小图片（当前识别的手势）==========
    if current_number > 0 and current_number <= len(overlayList):
        img_small = overlayList[current_number - 1]
        h_small, w_small, _ = img_small.shape
        if h_small <= frame.shape[0] and w_small <= frame.shape[1]:
            frame[0:h_small, 0:w_small] = img_small

    # ========== 14. 音量调节（拇指+食指 pinch）==========
    if volume_available and volume_enabled and hand_present and len(lmList) != 0:
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
        
        cv2.putText(frame, 'VOLUME', (28, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(frame, (30, 150), (80, 400), (255, 255, 255), 2)
        volBar = max(150, min(400, volBar))
        cv2.rectangle(frame, (30, int(volBar)), (80, 400), (255, 0, 255), cv2.FILLED)
        cv2.putText(frame, f'{int(volPer)}%', (22, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # ========== 15. 显示状态文字 ==========
    cv2.putText(frame, f'Current Fingers: {current_number}', (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if locked_number != -1:
        cv2.putText(frame, f'Locked Number: {display_number}', (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, 'Status: LOCKED', (50, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    else:
        if start_time > 0 and current_number == prev_number and current_number > 0:
            remaining = lock_duration - (current_time - start_time)
            cv2.putText(frame, f'Hold for {remaining:.1f} seconds to lock', (50, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, 'Status: DETECTING', (50, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'Keep gesture stable to lock', (50, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, 'Status: DETECTING', (50, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # ========== 16. 显示音量控制状态 ==========
    if volume_available:
        if volume_enabled:
            cv2.putText(frame, 'VOLUME CONTROL: ACTIVE', (850, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Use pinch (thumb+index) to adjust volume', (850, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, 'Remove hand from frame to disable', (850, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv2.putText(frame, 'VOLUME CONTROL: LOCKED', (850, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            progress_text = f'Unlock sequence: {volume_unlock_sequence} / {required_sequence}'
            cv2.putText(frame, progress_text, (850, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, 'Lock gestures in order: 2 -> 1 -> 5', (850, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # ========== 17. 显示提示信息 ==========
    cv2.putText(frame, 'Press Q or ESC to quit', (50, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if not hand_present:
        cv2.putText(frame, 'NO HAND DETECTED', (500, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ========== 18. 显示画面 ==========
    cv2.imshow('Hand Tracking - Digital Lock & Volume Control', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# ========== 19. 释放资源 ==========
cap.release()
cv2.destroyAllWindows()
detector.close()