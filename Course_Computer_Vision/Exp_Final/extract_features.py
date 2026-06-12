import os
import csv
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hand_gesture_data.csv")

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_unicode(path):
    with open(path, "rb") as f:
        data = bytearray(f.read())
    return cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def create_csv_header():
    header = ["class_name", "file_name"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def normalize_landmarks(landmarks):
    """归一化：以手腕(0)为中心，以手大小为尺度，使特征不随位置/距离变化"""
    wrist = landmarks[0]
    mcp = landmarks[9]
    scale = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2)
    if scale < 1e-6:
        scale = 1.0
    row = []
    for lm in landmarks:
        row.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale])
    return row


def landmarks_to_row(landmarks):
    if len(landmarks) != 21:
        return None
    return normalize_landmarks(landmarks)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker = mp.tasks.vision.HandLandmarker

    model_path = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型文件：{model_path}")
        print("请从 HandLandmarker/models/ 复制 hand_landmarker.task 到本目录 models/ 下")
        exit(1)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )
    landmarker = HandLandmarker.create_from_options(options)

    total_count = 0
    success_count = 0
    failed_count = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(create_csv_header())

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(DATASET_DIR, class_name)
            if not os.path.isdir(class_dir):
                print(f"[警告] 找不到类别文件夹：{class_dir}")
                continue

            images = [fn for fn in os.listdir(class_dir)
                      if fn.lower().endswith(IMAGE_EXTENSIONS)]
            print(f"处理类别 {class_name}：{len(images)} 张图片")

            for file_name in images:
                total_count += 1
                image_path = os.path.join(class_dir, file_name)

                image = imread_unicode(image_path)
                if image is None:
                    failed_count += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=image_rgb
                )
                results = landmarker.detect(mp_image)

                if not results.hand_landmarks:
                    failed_count += 1
                    continue

                hand_landmarks = results.hand_landmarks[0]
                landmark_row = landmarks_to_row(hand_landmarks)
                if landmark_row is None:
                    failed_count += 1
                    continue

                row = [class_name, file_name] + landmark_row
                writer.writerow(row)
                success_count += 1

        print(f"\n完成！总计 {total_count} 张")
        print(f"成功：{success_count}  |  失败：{failed_count}")

    landmarker.close()
