# 2-2-JumpRope-State-V1.py
# 跳绳计数器 - 第一版：单帧状态判断 (GROUND, JUMP)

import os
import cv2
from PoseModule import PoseDetector

# ========== 1. 视频路径设置 ==========
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
VIDEO_PATH = os.path.join(VIDEO_DIR, "JumpRope-5.mp4")  # 可改为 JumpRope-1.mp4

# 跳绳判断阈值（髋部中心 y 值小于此值视为 JUMP）
# 根据实际视频调整，JumpRope-5.mp4 建议 380
JUMP_Y_THRESHOLD = 380

# 播放延迟（毫秒）- 放慢速度方便观察
PLAY_DELAY = 30  # 30ms/帧，约33fps，可改为50/80进一步放慢

# 关键点编号
LEFT_HIP = 23
RIGHT_HIP = 24


def get_hip_center(lm_dict):
    """
    计算左右髋部中心点坐标
    回传:
        cx  : 髋部中心 x 值
        cy  : 髋部中心 y 值
        ok  : 是否成功取得左右髋部
    """
    if LEFT_HIP not in lm_dict or RIGHT_HIP not in lm_dict:
        return None, None, False

    left_x, left_y = lm_dict[LEFT_HIP]
    right_x, right_y = lm_dict[RIGHT_HIP]

    cx = int((left_x + right_x) / 2)
    cy = int((left_y + right_y) / 2)

    return cx, cy, True


if __name__ == "__main__":
    # ========== 2. 检查视频文件 ==========
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] 视频文件不存在：{VIDEO_PATH}")
        print("请将跳绳视频放入 videos/ 文件夹后重试")
        exit(1)

    # ========== 3. 打开视频 ==========
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    # ========== 4. 建立 PoseDetector 检测器 ==========
    # 先用 IMAGE 模式测试单帧效果，确认阈值后再改 VIDEO 模式看连续效果
    # detector = PoseDetector(mode="VIDEO", model_complexity="full")
    detector = PoseDetector(mode="IMAGE", model_complexity="full")

    print("[INFO] 开始处理跳绳视频，按 'q' 或 ESC 键退出")
    print(f"[INFO] 判断阈值 JUMP_Y_THRESHOLD = {JUMP_Y_THRESHOLD}")

    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[INFO] 视频播放完毕")
                break

            frame_count += 1

            # ========== 5. 执行姿态检测 ==========
            frame = detector.findPose(frame, draw=True)

            h, w, c = frame.shape

            # ========== 6. 获取关键点字典 ==========
            lm_dict = detector.findPositionDict(frame, poseNo=0, draw=False)

            # 初始化状态变量
            state = "UNKNOWN"
            cx, cy = None, None
            ok = False

            if len(lm_dict) != 0:
                # ========== 7. 计算髋部中心点 ==========
                cx, cy, ok = get_hip_center(lm_dict)

                if ok:
                    # ========== 8. 根据阈值判断 GROUND / JUMP ==========
                    if cy < JUMP_Y_THRESHOLD:
                        state = "JUMP"
                    else:
                        state = "GROUND"

                    # ========== 9. 绘制髋部中心点 ==========
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "hip center",
                        (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )

                    # ========== 10. 在左上角显示 hip_center_y ==========
                    cv2.putText(
                        frame,
                        f"hip_center_y: {cy}",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "NO POSE (hip not found)",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
            else:
                cv2.putText(
                    frame,
                    "NO POSE DETECTED",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # ========== 11. 显示状态信息 ==========
            # 显示当前状态
            state_color = (0, 255, 0) if state == "GROUND" else (0, 165, 255) if state == "JUMP" else (0, 0, 255)
            cv2.putText(
                frame,
                f"STATE: {state}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                state_color,
                3
            )

            # 显示阈值
            cv2.putText(
                frame,
                f"JUMP_Y_THRESHOLD: {JUMP_Y_THRESHOLD}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            # 显示帧数
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            # ========== 12. 画出判定线（水平线） ==========
            cv2.line(
                frame,
                (0, JUMP_Y_THRESHOLD),
                (w, JUMP_Y_THRESHOLD),
                (255, 0, 0),
                2
            )
            cv2.putText(
                frame,
                f"JUMP LINE: y={JUMP_Y_THRESHOLD}",
                (20, JUMP_Y_THRESHOLD - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # ========== 13. 显示画面 ==========
            cv2.imshow("Rope Skipping Counter V1 - GROUND/JUMP", frame)

            # 按键退出
            key = cv2.waitKey(PLAY_DELAY) & 0xFF
            if key == 27 or key == ord('q'):
                print("[INFO] 退出程序")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[INFO] 资源已释放")