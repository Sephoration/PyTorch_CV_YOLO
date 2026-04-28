import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====== 路径设置 ======
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images","PoseEstimation", "Pose-1.jpg")  # 可替换为其他图片
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_full.task")

# ====== 下载模型（如本地不存在） ======
def ensure_model(model_path: str, model_url: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] 模型不存在，开始下载：{model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] 下载完成：{model_path}")

# ====== 主程序 ======
if __name__ == "__main__":
    # 1. 确保模型存在
    ensure_model(MODEL_PATH, MODEL_URL)

    # 2. 读取图片
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{IMAGE_PATH}")

    # 3. 初始化 PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

    landmarker = PoseLandmarker.create_from_options(options)

    # 4. 转换图片格式（BGR → RGB → mp.Image）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # 5. 执行姿态检测
    results = landmarker.detect(mp_image)

    # 6. 检测结果处理与可视化
    if results.pose_landmarks:
        # 获取默认骨架连接关系
        POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        h, w, _ = image.shape

        for pose_landmarks in results.pose_landmarks:
            # 画骨架连线
            for connection in POSE_CONNECTIONS:
                start_idx = connection.start
                end_idx = connection.end

                lm_start = pose_landmarks[start_idx]
                lm_end = pose_landmarks[end_idx]

                if lm_start.visibility < 0.5 or lm_end.visibility < 0.5:
                    continue

                x0 = int(lm_start.x * w)
                y0 = int(lm_start.y * h)
                x1 = int(lm_end.x * w)
                y1 = int(lm_end.y * h)

                cv2.line(image, (x0, y0), (x1, y1), (255, 0, 255), 2)

            # 画关键点
            for i, lm in enumerate(pose_landmarks):
                if lm.visibility < 0.5:
                    continue
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

        # 显示结果
        cv2.imshow("Pose Estimation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("没有检测到人体")