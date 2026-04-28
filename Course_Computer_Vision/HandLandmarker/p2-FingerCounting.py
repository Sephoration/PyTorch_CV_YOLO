import cv2
import time
import HandTrackingModule as htm
import os

# ========== 1. 图片加载 ==========
# 使用绝对路径，确保无论从哪里执行脚本都能找到图片目录
folderPath = os.path.join(os.path.dirname(__file__), "images","fingerCounting")
myList = os.listdir(folderPath)
overlayList = []

for imgPath in sorted(myList):  # 排序确保顺序正确
    img = cv2.imread(os.path.join(folderPath, imgPath))
    if img is not None:
        overlayList.append(img)

# ========== 2. 摄像头初始化 ==========
cap = cv2.VideoCapture(0)
cap.set(3, 640)   # 宽度
cap.set(4, 480)   # 高度

# ========== 3. 手部检测器 ==========
detector = htm.HandDetector(detectionCon=0.75)

# ========== 4. 数字锁定相关变量 ==========
prev_number = -1        # 上一帧识别的手指数
start_time = 0          # 当前数字开始出现的时间
locked_number = 0       # 已被锁定显示的数字
display_number = 0      # 当前真正要显示的数字
lock_duration = 2.0     # 锁定所需秒数（可改为2秒）

# 窗口设置
cv2.namedWindow('HandTracking', cv2.WINDOW_NORMAL)
cv2.moveWindow('HandTracking', 100, 100)

while True:
    success, frame = cap.read()
    if not success:
        continue

    # ========== 5. 手部检测与关键点提取 ==========
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    totalFingers = 0  # 默认0根手指

    if len(lmList) != 0:
        fingers = []
        tipIds = [4, 8, 12, 16, 20]

        # ----- 5.1 获取左右手信息 -----
        handLabel = None
        if detector.results and detector.results.handedness:
            handLabel = detector.results.handedness[0][0].category_name

        # ----- 5.2 拇指判断（分左右手）-----
        if handLabel == "Left":
            # 左手：拇指张开时，指尖x < 前一个关节x
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif handLabel == "Right":
            # 右手：拇指张开时，指尖x > 前一个关节x
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # 兜底逻辑（默认按右手）
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # ----- 5.3 其余四指判断（看y坐标）-----
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

    # ========== 6. 数字锁定逻辑 ==========
    current_number = totalFingers
    current_time = time.time()

    if current_number == prev_number:
        # 数字没变：检查是否达到锁定时间
        if current_number != locked_number:
            if start_time == 0:
                start_time = current_time
            elif current_time - start_time >= lock_duration:
                # 连续保持2秒，锁定这个数字
                locked_number = current_number
                display_number = locked_number
    else:
        # 数字发生变化：重新计时
        start_time = 0
        locked_number = -1  # 解锁
        # 注意：display_number 暂时不变，直到新数字被锁定
        # 如果你希望变化瞬间就显示新数字（不稳定版），可以把下面注释去掉
        # display_number = current_number

    # 更新上一帧的数字
    prev_number = current_number

    # 如果还没有任何数字被锁定，且当前有识别结果，可以显示当前数字（可选）
    # 这里选择：未锁定时也显示当前数字，但加上提示文字
    show_number = display_number if locked_number != -1 else current_number

    # ========== 7. 图片显示 ==========
    # 在画面左上角显示当前要展示的数字图片
    if show_number > 0 and show_number <= len(overlayList):
        h, w, _ = overlayList[show_number - 1].shape
        frame[0:h, 0:w] = overlayList[show_number - 1]

    # ========== 8. 在画面显示文字信息 ==========
    # 显示当前识别到的原始手指数
    cv2.putText(frame, f"Detected: {current_number}", (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示锁定状态
    if locked_number != -1:
        cv2.putText(frame, f"LOCKED: {display_number}", (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 显示倒计时或已锁定提示
        if current_number == locked_number:
            cv2.putText(frame, "STABLE", (50, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        # 显示还需要保持多久才能锁定
        if start_time > 0 and current_number == prev_number:
            remaining = lock_duration - (current_time - start_time)
            if remaining > 0:
                cv2.putText(frame, f"Hold {remaining:.1f}s to lock", (50, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Not locking", (50, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ========== 9. 显示画面 ==========
    cv2.imshow('HandTracking', frame)

    # 按 ESC 或 q 退出
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# ========== 10. 释放资源 ==========
cap.release()
cv2.destroyAllWindows()