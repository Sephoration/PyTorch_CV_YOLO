# p3-HandSlideControl.py
# 手势动作控制 PPT 换页
# 两阶段判断：
# 1. 手势状态识别：五指伸直 + 掌心朝上/朝下
# 2. 动作触发控制：手掌明显向上或向下移动

import cv2
import math
import time
import pyautogui
from collections import deque
import HandTrackingModule as htm

# PyAutoGUI 安全设置
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def calc_angle(a, b, c):
    """
    计算三点夹角（角度制）
    a, b, c: 三个点的坐标 (x, y)
    b 为角的顶点
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])

    if norm_ba == 0 or norm_bc == 0:
        return 0

    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle


def is_finger_straight(ImDict, mcp_id, pip_id, dip_id, tip_id, threshold=160):
    """
    判断单根手指是否伸直（基于两个关节角度）
    mcp: 掌指关节（根部）
    pip: 近端指间关节（中间）
    dip: 远端指间关节（指尖附近）
    tip: 指尖
    角度越接近180度越直，大于 threshold 视为伸直
    """
    # 确保所有关键点都存在
    if mcp_id not in ImDict or pip_id not in ImDict or dip_id not in ImDict or tip_id not in ImDict:
        return False
    
    mcp = ImDict[mcp_id]
    pip = ImDict[pip_id]
    dip = ImDict[dip_id]
    tip = ImDict[tip_id]
    
    angle_pip = calc_angle(mcp, pip, dip)   # PIP关节角度
    angle_dip = calc_angle(pip, dip, tip)   # DIP关节角度
    
    return angle_pip > threshold and angle_dip > threshold


def is_thumb_open(ImDict):
    """
    判断拇指是否张开（角度 + 距离比例）
    拇指需要单独处理，因为其活动方式不同
    """
    # 确保关键点存在
    if 2 not in ImDict or 3 not in ImDict or 4 not in ImDict or 5 not in ImDict or 0 not in ImDict:
        return False
    
    # 条件1：拇指角度（关键点2,3,4）
    angle_thumb = calc_angle(ImDict[2], ImDict[3], ImDict[4])
    
    # 条件2：拇指尖到食指根部的距离（判断是否真的张开）
    thumb_tip = ImDict[4]
    index_mcp = ImDict[5]      # 食指根部
    wrist = ImDict[0]          # 手腕
    
    thumb_dist = math.hypot(thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1])
    palm_size = math.hypot(wrist[0] - index_mcp[0], wrist[1] - index_mcp[1])
    
    if palm_size == 0:
        return False
    
    # 角度大于145度 且 拇指距离大于手掌尺寸的0.45倍
    return angle_thumb > 145 and thumb_dist > palm_size * 0.45


def get_finger_state(ImDict):
    """
    获取五指状态字典
    返回: {"thumb": bool, "index": bool, "middle": bool, "ring": bool, "pinky": bool}
    """
    thumb_open = is_thumb_open(ImDict)
    
    # 四指判断（使用正确的关键点ID）
    # 食指: mcp=5, pip=6, dip=7, tip=8
    index_open = is_finger_straight(ImDict, 5, 6, 7, 8, threshold=160)
    # 中指: mcp=9, pip=10, dip=11, tip=12
    middle_open = is_finger_straight(ImDict, 9, 10, 11, 12, threshold=160)
    # 无名指: mcp=13, pip=14, dip=15, tip=16
    ring_open = is_finger_straight(ImDict, 13, 14, 15, 16, threshold=160)
    # 小指: mcp=17, pip=18, dip=19, tip=20
    pinky_open = is_finger_straight(ImDict, 17, 18, 19, 20, threshold=160)
    
    return {
        "thumb": thumb_open,
        "index": index_open,
        "middle": middle_open,
        "ring": ring_open,
        "pinky": pinky_open
    }


def is_all_fingers_straight(finger_state):
    """判断是否五指全部伸直"""
    return all(finger_state.values())


def get_palm_orientation_for_right_hand(ImDict, threshold=25):
    """
    判断右手掌心方向（假设手掌朝向镜头，未明显侧转）
    基于拇指尖(4)与小指尖(20)的x坐标差值
    dx < -threshold: 拇指在左，小指在右 -> 掌心朝下
    dx > threshold: 拇指在右，小指在左 -> 掌心朝上
    """
    if 4 not in ImDict or 20 not in ImDict:
        return "UNKNOWN", 0

    thumb_tip = ImDict[4]
    pinky_tip = ImDict[20]

    dx = thumb_tip[0] - pinky_tip[0]

    if dx < -threshold:
        return "PALM_DOWN", dx
    elif dx > threshold:
        return "PALM_UP", dx
    else:
        return "UNKNOWN", dx


def get_hand_center(ImDict):
    """计算手掌中心（手腕0 + 中指根9 的中点）"""
    if 0 not in ImDict or 9 not in ImDict:
        return None
    wrist = ImDict[0]
    middle_mcp = ImDict[9]  # 中指根部
    cx = (wrist[0] + middle_mcp[0]) // 2
    cy = (wrist[1] + middle_mcp[1]) // 2
    return (cx, cy)


def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化手部检测器
    detector = htm.HandDetector()

    # 动作判定参数
    TRACK_POINTS = 5          # 记录最近几帧手中心位置
    MOVE_Y_THRESHOLD = 35     # 上下移动判定阈值（像素）
    COOLDOWN_TIME = 0.8       # 两次翻页之间的冷却时间（秒）

    # 轨迹记录（使用deque自动维护长度）
    center_y_history = deque(maxlen=TRACK_POINTS)
    last_trigger_time = 0
    last_move_direction = None  # 记录上一次触发的方向

    print("程序已启动，按下 ESC 或 q 退出")
    print("控制说明：")
    print("  1. 五指伸直 + 掌心朝上或朝下 -> 进入控制状态")
    print("  2. 手掌向上移动 -> 上一页")
    print("  3. 手掌向下移动 -> 下一页")
    print("  4. 请确保PPT处于放映状态\n")

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("无法读取摄像头画面")
            break

        # 镜像翻转，体验更自然
        frame = cv2.flip(frame, 1)

        # 手部检测
        frame = detector.findHands(frame, draw=True, flip=False)  # flip=False 因为已经手动翻转了
        ImDict = detector.findPositionDict(frame, handNo=0, draw=False)

        # 默认显示文字
        palm_text = "Palm: None"
        finger_text = "Finger State: None"
        all_fingers_text = "All Fingers Straight: None"
        move_text = "Move: None"

        # 初始化手势状态
        palm_orientation = "UNKNOWN"
        gesture_ready = False
        finger_state = {}

        if len(ImDict) != 0:
            # ======================
            # 第一阶段：手势状态识别
            # ======================

            # 1. 判断掌心方向
            palm_orientation, dx = get_palm_orientation_for_right_hand(ImDict, threshold=25)
            palm_text = f"Palm: {palm_orientation} (dx={int(dx)})"

            # 2. 判断五指状态（包括拇指）
            finger_state = get_finger_state(ImDict)
            fingers_ok = is_all_fingers_straight(finger_state)
            
            # 显示五指状态（按照课件格式：T:1 I:1 M:1 R:1 P:1）
            finger_text = (f"Fingers: T:{int(finger_state['thumb'])} "
                          f"I:{int(finger_state['index'])} "
                          f"M:{int(finger_state['middle'])} "
                          f"R:{int(finger_state['ring'])} "
                          f"P:{int(finger_state['pinky'])}")
            all_fingers_text = f"All Fingers Straight: {fingers_ok}"

            # 手势就绪：五指伸直 + 掌心朝上或朝下
            gesture_ready = fingers_ok and palm_orientation in ["PALM_DOWN", "PALM_UP"]
            gesture_ready_text = f"Gesture Ready: {'Yes' if gesture_ready else 'No'}"

            # ======================
            # 第二阶段：动作触发控制（仅在手势就绪时）
            # ======================
            if gesture_ready:
                # 获取手掌中心坐标
                hand_center = get_hand_center(ImDict)
                if hand_center is not None:
                    cx, cy = hand_center
                    center_y_history.append(cy)

                    # 显示当前手掌中心
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

                    # 轨迹收集足够后判断移动方向
                    if len(center_y_history) == TRACK_POINTS:
                        # 计算最早与最新的y坐标差值
                        y_start = center_y_history[0]
                        y_end = center_y_history[-1]
                        y_diff = y_end - y_start  # 向下移动为正，向上为负

                        # 判断移动方向
                        current_time = time.time()
                        time_since_last = current_time - last_trigger_time

                        if y_diff > MOVE_Y_THRESHOLD and time_since_last > COOLDOWN_TIME:
                            # 向下移动 -> 下一页
                            pyautogui.press('right')
                            last_trigger_time = current_time
                            last_move_direction = "DOWN"
                            move_text = f"Move: DOWN -> Next Page (diff={int(y_diff)})"
                            print(f"[触发] 下一页 (y_diff={int(y_diff)})")

                        elif y_diff < -MOVE_Y_THRESHOLD and time_since_last > COOLDOWN_TIME:
                            # 向上移动 -> 上一页
                            pyautogui.press('left')
                            last_trigger_time = current_time
                            last_move_direction = "UP"
                            move_text = f"Move: UP -> Previous Page (diff={int(y_diff)})"
                            print(f"[触发] 上一页 (y_diff={int(y_diff)})")
                        else:
                            # 移动不明显或冷却中
                            if time_since_last <= COOLDOWN_TIME:
                                move_text = f"Move: Cooldown ({COOLDOWN_TIME - time_since_last:.1f}s left)"
                            else:
                                move_text = f"Move: None (y_diff={int(y_diff)})"
                    else:
                        move_text = f"Tracking... ({len(center_y_history)}/{TRACK_POINTS})"
            else:
                # 手势未就绪，清空轨迹记录
                center_y_history.clear()
                last_move_direction = None
                move_text = "Move: None (gesture not ready)"
        else:
            # 未检测到手
            center_y_history.clear()
            last_move_direction = None

        # 在画面上显示状态信息（按照课件布局）
        cv2.putText(frame, palm_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, finger_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, all_fingers_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 显示手势就绪状态
        gesture_ready_status = "Gesture Ready: Yes" if (len(ImDict) != 0 and 'gesture_ready' in dir() and gesture_ready) else "Gesture Ready: No"
        cv2.putText(frame, gesture_ready_status, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, move_text, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # 添加操作提示
        h, w = frame.shape[:2]
        cv2.putText(frame, "Gesture Ready: Straight Fingers + Palm Up/Down", (10, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Then: Move Up = Prev, Move Down = Next", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Press ESC/q to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 显示画面
        cv2.imshow("Hand Slide Control - Palm + Vertical Motion", frame)

        # 退出条件
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == '__main__':
    main()