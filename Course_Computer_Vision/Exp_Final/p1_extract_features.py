import os
import csv
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "csv")
OUTPUT_CSV_RAW = os.path.join(OUTPUT_DIR, "hand_gesture_data.csv")
OUTPUT_CSV_HAND = os.path.join(OUTPUT_DIR, "hand_gesture_data_hand.csv")
OUTPUT_CSV_HAND_V2 = os.path.join(OUTPUT_DIR, "hand_gesture_data_hand_v2.csv")
OUTPUT_CSV_DIST = os.path.join(OUTPUT_DIR, "hand_gesture_data_dist.csv")

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
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

def create_csv_header_2d():
    header = ["class_name", "file_name"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    return header

def create_csv_header_dist():
    header = ["class_name", "file_name"]
    for i in range(210):
        header.append(f"d{i}")
    return header


def normalize_landmarks_raw(landmarks):
    """原始归一化：相对手腕，按手尺度缩放——相机坐标系"""
    wrist = landmarks[0]
    mcp = landmarks[9]
    scale = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2)
    if scale < 1e-6:
        scale = 1.0
    row = []
    for lm in landmarks:
        row.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale])
    return row


def normalize_landmarks_hand(landmarks):
    """旋转不变归一化：2D 旋转对齐，让中指根始终指向正上方，按手尺度缩放"""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # 2D 方向：手腕 → 中指根
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    scale = np.sqrt(dx*dx + dy*dy)
    if scale < 1e-6:
        scale = 1.0

    # 旋转角：让中指根始终指向"正上方"
    angle = np.arctan2(dy, dx)
    rot = -np.pi / 2 - angle
    cos_a, sin_a = np.cos(rot), np.sin(rot)

    row = []
    for lm in landmarks:
        rx = lm.x - wrist.x
        ry = lm.y - wrist.y
        x_rot = cos_a * rx - sin_a * ry
        y_rot = sin_a * rx + cos_a * ry
        row.extend([x_rot / scale, y_rot / scale])
    return row


def normalize_landmarks_hand_v2(landmarks):
    """2D 手掌坐标系归一化（去掉 Z 轴，避免手心/手背翻转导致符号反转）
     用 4 个掌骨点构造 2D 正交基：
     Y = wrist(0) → middle_mcp(9)
     X = index_mcp(5) → pinky_mcp(17)（Gram-Schmidt 正交化）
     """
    wrist    = np.array([landmarks[0].x,  landmarks[0].y])
    index_mcp  = np.array([landmarks[5].x,  landmarks[5].y])
    middle_mcp = np.array([landmarks[9].x,  landmarks[9].y])
    pinky_mcp  = np.array([landmarks[17].x, landmarks[17].y])

    y_axis = middle_mcp - wrist
    scale = np.linalg.norm(y_axis)
    if scale < 1e-6:
        scale = 1.0
    y_axis = y_axis / scale

    x_raw = pinky_mcp - index_mcp
    x_axis = x_raw - np.dot(x_raw, y_axis) * y_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm > 1e-6:
        x_axis = x_axis / x_norm
    else:
        # 回退：选择一个与 y_axis 不平行的向量
        fallback = np.array([1.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 1.0])
        x_axis = y_axis - np.dot(y_axis, fallback) * fallback
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            x_axis = np.array([1.0, 0.0])

    R = np.column_stack([x_axis, y_axis])

    row = []
    for lm in landmarks:
        p = np.array([lm.x, lm.y])
        p_local = R.T @ (p - wrist)
        row.extend([p_local[0] / scale, p_local[1] / scale])
    return row


def normalize_landmarks_distances(landmarks):
    """用所有点对之间的欧氏距离作为特征，完全几何不变（平移/旋转/翻转不变）"""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    features = []
    for i in range(21):
        for j in range(i + 1, 21):
            features.append(np.linalg.norm(pts[i] - pts[j]))
    # 按手腕到中指根距离归一化，实现尺度不变
    scale = features[9] if len(features) > 9 else 1.0  # 点 0-9 是第 9 个距离
    if scale < 1e-6:
        scale = 1.0
    return [f / scale for f in features]


def landmarks_to_rows(landmarks):
    """对一个手势的 landmarks，返回 (raw_row, hand_row, hand_v2_row, dist_row) 四个版本"""
    if len(landmarks) != 21:
        return None, None, None, None
    return (normalize_landmarks_raw(landmarks),
            normalize_landmarks_hand(landmarks),
            normalize_landmarks_hand_v2(landmarks),
            normalize_landmarks_distances(landmarks))


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
    mirror_count = 0

    f_raw = open(OUTPUT_CSV_RAW, "w", newline="")
    f_hand = open(OUTPUT_CSV_HAND, "w", newline="")
    f_hand_v2 = open(OUTPUT_CSV_HAND_V2, "w", newline="")
    f_dist = open(OUTPUT_CSV_DIST, "w", newline="")
    writer_raw = csv.writer(f_raw)
    writer_hand = csv.writer(f_hand)
    writer_hand_v2 = csv.writer(f_hand_v2)
    writer_dist = csv.writer(f_dist)
    writer_raw.writerow(create_csv_header())
    writer_hand.writerow(create_csv_header_2d())
    writer_hand_v2.writerow(create_csv_header_2d())
    writer_dist.writerow(create_csv_header_dist())

    try:
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
                raw_row, hand_row, hand_v2_row, dist_row = landmarks_to_rows(hand_landmarks)
                if raw_row is None:
                    failed_count += 1
                    continue

                writer_raw.writerow([class_name, file_name] + raw_row)
                writer_hand.writerow([class_name, file_name] + hand_row)
                writer_hand_v2.writerow([class_name, file_name] + hand_v2_row)
                writer_dist.writerow([class_name, file_name] + dist_row)
                success_count += 1

                # ---- 镜像翻转增强：对同一张图做水平翻转再提取特征 ----
                mirror_file = f"{os.path.splitext(file_name)[0]}_mirror.png"
                image_flip = cv2.flip(image, 1)  # 水平翻转
                image_flip_rgb = cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB)
                mp_image_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_flip_rgb)
                results_flip = landmarker.detect(mp_image_flip)
                if results_flip.hand_landmarks:
                    hand_lms_flip = results_flip.hand_landmarks[0]
                    raw_flip, hand_flip, hand_v2_flip, dist_flip = landmarks_to_rows(hand_lms_flip)
                    if raw_flip:
                        writer_raw.writerow([class_name, mirror_file] + raw_flip)
                        writer_hand.writerow([class_name, mirror_file] + hand_flip)
                        writer_hand_v2.writerow([class_name, mirror_file] + hand_v2_flip)
                        # Dist 模式不写镜像（距离特征已几何不变，不需要）
                        mirror_count += 1
    finally:
        f_raw.close()
        f_hand.close()
        f_hand_v2.close()
        f_dist.close()

    print(f"\n完成！总计 {total_count} 张原始图片")
    print(f"成功提取原图：{success_count}  |  失败：{failed_count}")
    print(f"镜像增强新增：{mirror_count}  |  总样本：{success_count + mirror_count}")
    print(f"原始特征 CSV：{OUTPUT_CSV_RAW}")
    print(f"手掌坐标系(2D) CSV：{OUTPUT_CSV_HAND}")
    print(f"手掌坐标系(2D-正交基) CSV：{OUTPUT_CSV_HAND_V2}")
    print(f"距离特征 CSV：{OUTPUT_CSV_DIST}")

    landmarker.close()
