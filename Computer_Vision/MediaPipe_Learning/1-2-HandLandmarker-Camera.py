import os
import time
import cv2
import urllib.request
import mediapipe as mp

# ====== 路径设置 ======
# 移除图片路径，改用摄像头输入

# ====== 1) 下载/指定 HandLandmarker 模型 ======
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def ensure_model(model_path: str, model_url: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] 模型不存在，开始下载：{model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] 下载完成：{model_path}")


if __name__ == '__main__':
    ensure_model(MODEL_PATH, MODEL_URL)

    # ====== 2) 建立 HandLandmarker(VIDEO 模式)======
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker

    # 使用新版 HandLandmarker 的连接定义
    HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,  # 改为 VIDEO 模式
        num_hands=2,  # 摄像头模式下检测最多2只手
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)

    # ====== 3) 初始化摄像头 ======
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备或把 0 改成 1/2 试试")

    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] 摄像头已打开，按 'q' 或 ESC 键退出")

    # ====== 4) 实时处理摄像头画面 ======
    try:
        while True:
            # 读取摄像头帧
            success, frame = cap.read()
            if not success or frame is None:
                print("[WARNING] 无法读取摄像头画面")
                continue

            # 镜像翻转画面（更符合直觉）
            frame = cv2.flip(frame, 1)

            # OpenCV 是 BGR，MediaPipe 需要 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转成 MediaPipe Image 格式
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # VIDEO 模式必须提供时间戳（毫秒）
            timestamp_ms = int(time.time() * 1000)

            # 执行手部检测
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ====== 5) 绘制检测结果 ======
            if results.hand_landmarks:
                # 显示检测到的手的数量
                hand_count = len(results.hand_landmarks)
                cv2.putText(frame, f"Hands Detected: {hand_count}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                h, w, _ = frame.shape
                for hand_landmarks in results.hand_landmarks:
                    # 绘制手部关键点连线
                    for connection in HAND_CONNECTIONS:
                        start_idx = connection.start
                        end_idx = connection.end

                        x0 = int(hand_landmarks[start_idx].x * w)
                        y0 = int(hand_landmarks[start_idx].y * h)
                        x1 = int(hand_landmarks[end_idx].x * w)
                        y1 = int(hand_landmarks[end_idx].y * h)

                        cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)

                    # 绘制关键点
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            else:
                # 未检测到手
                cv2.putText(frame, "No Hand Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ====== 6) 显示结果 ======
            cv2.imshow("Real-time Hand Detection", frame)

            # 按键退出（q 或 ESC）
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[INFO] 退出程序")
                break

    finally:
        # ====== 7) 释放资源 ======
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()