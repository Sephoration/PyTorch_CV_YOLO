import os
import time
import cv2
import urllib.request
import mediapipe as mp

# ====== 路径设置 ======
# 使用绝对路径，确保无论从哪里执行脚本都能找到图片
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images","HandLandMarks", "Hands-4-1.jpg")

# ====== 1) 下載/指定 HandLandmarker 模型 ======
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

def ensure_model(model_path: str, model_url: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] 模型不存在，開始下載：{model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] 下載完成：{model_path}")

if __name__ == '__main__':
    ensure_model(MODEL_PATH, MODEL_URL)

    # ====== 2) 建立 HandLandmarker(IMAGE 模式)======
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,  # 改成 IMAGE 模式
        num_hands=4,  # 图中有几只手
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)


    # ====== 3) 读取图片 ======
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{IMAGE_PATH}")

    # OpenCV 是 BGR，MediaPipe 需要 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转成 MediaPipe Image 格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    # ====== 4) 执行手部检测 ======
    results = landmarker.detect(mp_image)
    print(results)


    # ====== 5) 输出检测结果 ======
    if results.hand_landmarks:
        print(f"检测到 {len(results.hand_landmarks)} 只手")

        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            print(f"\n第 {hand_idx + 1} 只手的关键点：")
            for i, lm in enumerate(hand_landmarks):
                print(f"关键点 {i}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")
    else:
        print("没有检测到手")

    # ====== 2) 建立 HandLandmarker（IMAGE 模式）======
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker

    # 使用新版 HandLandmarker 的连接定义
    HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    # ====== 6) 将关键点画到图上 ======
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, _ = image.shape

            # 手部关键点连线
            for connection in HAND_CONNECTIONS:
                start_idx = connection.start
                end_idx = connection.end

                x0 = int(hand_landmarks[start_idx].x * w)
                y0 = int(hand_landmarks[start_idx].y * h)
                x1 = int(hand_landmarks[end_idx].x * w)
                y1 = int(hand_landmarks[end_idx].y * h)

                cv2.line(image, (x0, y0), (x1, y1), (255, 0, 255), 2)

            # 画关键点
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)


    # ====== 7) 显示结果 ======
    cv2.imshow("Hand Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
