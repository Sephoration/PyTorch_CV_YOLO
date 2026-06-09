import os
import csv
import cv2
from PoseModule import PoseDetector

##################### 设定全局常数 #####################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "YogaPose")
CSV_DIR = os.path.join(BASE_DIR, "csv")
TRAIN_CSV_PATH = os.path.join(CSV_DIR, "yoga_pose_train.csv")
TEST_CSV_PATH = os.path.join(CSV_DIR, "yoga_pose_test.csv")

SPLIT_NAMES = ["train", "test"]
CLASS_NAMES = ["tree", "warrior2", "plank", "downdog", "goddess"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
##################### 设定全局常数 #####################


# ==============================
# 建立 CSV 表头
# 格式参考 MediaPipe pose classification：
# class_name, file_name, x0, y0, z0, v0, ...
# ==============================
def create_csv_header():
    header = ["class_name", "file_name"]

    for i in range(33):
        header.extend([
            f"x{i}", f"y{i}", f"z{i}", f"v{i}"
        ])

    return header


# ==============================
# 将 landmarks 取出 33 个关键点
# 每个关键点保存 x, y, z, visibility
# ==============================
def landmarks_to_row(landmarks):
    if len(landmarks) != 33:
        return None

    row = []

    for lm in landmarks:
        row.extend([
            lm.x,
            lm.y,
            lm.z,
            lm.visibility
        ])

    return row


# ==============================
# 取出某个 split 下所有图片
# split_name: train 或 test
# ==============================
def get_image_paths(split_name):
    image_items = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_DIR, split_name, class_name)

        if not os.path.isdir(class_dir):
            print(f"[警告] 找不到类别文件夹：{class_dir}")
            continue

        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(class_dir, file_name)
                image_items.append((class_name, file_name, image_path))

    return image_items


if __name__ == '__main__':
    # 初始化 PoseDetector（图片模式）
    detector = PoseDetector(mode="IMAGE", model_complexity="full")

    for split_name in SPLIT_NAMES:
        print("=" * 60)
        print(f"开始处理：{split_name}")

        os.makedirs(CSV_DIR, exist_ok=True)
        if split_name == "train":
            output_csv_path = TRAIN_CSV_PATH
        else:
            output_csv_path = TEST_CSV_PATH
        print(f"输出 CSV：{output_csv_path}")

        # 读取图片路径并初始化统计变量
        image_items = get_image_paths(split_name)

        total_count = 0
        success_count = 0
        failed_count = 0

        # 打开 CSV 文件并写入表头
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(create_csv_header())

            # 遍历数据集图片并提取 Pose Landmark
            for class_name, file_name, image_path in image_items:
                total_count += 1

                image = cv2.imread(image_path)
                if image is None:
                    failed_count += 1
                    print(f"[读取失败] {image_path}")
                    continue

                # Pose 检测
                image = detector.findPose(image, draw=False)

                # 取出原始 landmarks
                landmarks = detector.getLandmarks(poseNo=0)

                if len(landmarks) == 0:
                    failed_count += 1
                    print(f"[未检测到人体] {image_path}")
                    continue

                # 将关键点转换为 CSV 数据列
                landmark_row = landmarks_to_row(landmarks)
                if landmark_row is None:
                    failed_count += 1
                    print(f"[关键点数量异常] {image_path}")
                    continue

                # 合并类别名、图片名并写入 CSV
                row = [class_name, file_name] + landmark_row
                writer.writerow(row)

                success_count += 1
                if success_count % 50 == 0:
                    print(f"[进度] 成功转换 {success_count} 张图片")

        # 输出转换统计结果
        print("-" * 60)
        print(f"{split_name} 转换结果")
        print(f"图片总数：{total_count}")
        print(f"成功转换：{success_count}")
        print(f"失败数量：{failed_count}")
        print("=" * 60)

    detector.close()
