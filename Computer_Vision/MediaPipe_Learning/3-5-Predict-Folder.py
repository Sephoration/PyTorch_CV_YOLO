import os
import pickle
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from PoseModule import PoseDetector

####################### 设定全局常数 #######################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yoga_pose_knn.pkl")

# 默认测试文件夹（可修改为任意图片文件夹路径）
DEFAULT_TEST_DIR = os.path.join(BASE_DIR, "datasets", "YogaPose", "test")

# 显示参数
MAX_DISPLAY_SIZE = 800      # 图片缩放后的最大宽/高
DISPLAY_DELAY = 1000        # 每张图片显示时间（毫秒）= 1 秒

# 支持的图片格式
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
####################### 设定全局常数 #######################


def create_feature_names():
    """创建 132 个特征列名"""
    names = []
    for i in range(33):
        names.append(f"x{i}")
        names.append(f"y{i}")
        names.append(f"z{i}")
        names.append(f"v{i}")
    return names


def landmarks_to_features(landmarks):
    """将 33 个 Pose Landmark 转换为 132 维特征向量"""
    if len(landmarks) != 33:
        return None

    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    return features


def adaptive_resize(img, max_size=MAX_DISPLAY_SIZE):
    """
    自适应缩放图片，保持宽高比

    Args:
        img: 输入图像
        max_size: 最大宽度或高度

    Returns:
        缩放后的图像
    """
    h, w = img.shape[:2]

    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    return img


def collect_images(folder_path):
    """
    递归收集文件夹下所有支持的图片文件

    Args:
        folder_path: 文件夹路径

    Returns:
        图片路径列表
    """
    image_files = []
    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(folder_path).rglob(f"*{ext}"))
        image_files.extend(Path(folder_path).rglob(f"*{ext.upper()}"))

    # 去重并转为字符串列表
    image_files = sorted(set(str(p) for p in image_files))
    return image_files


if __name__ == '__main__':
    print("=" * 55)
    print("Pose Classification - 批量文件夹预测")
    print("=" * 55)

    # ===== 1) 加载模型 =====
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件：{MODEL_PATH}")
        print("请先执行 3-3-Train-KNN.py 训练模型。")
        exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"[信息] 模型已加载：{MODEL_PATH}")
    print(f"[信息] 支持类别：{model.classes_}")
    print("-" * 55)

    # ===== 2) 初始化 PoseDetector（IMAGE 模式） =====
    detector = PoseDetector(mode="IMAGE", model_complexity="full")

    # ===== 3) 收集图片 =====
    test_dir = DEFAULT_TEST_DIR
    if not os.path.exists(test_dir):
        print(f"[警告] 默认测试文件夹不存在：{test_dir}")
        print("请修改 DEFAULT_TEST_DIR 为有效的图片文件夹路径。")
        exit(1)

    image_files = collect_images(test_dir)

    if len(image_files) == 0:
        print(f"[错误] 文件夹中没有找到图片: {test_dir}")
        exit(1)

    print(f"[信息] 测试文件夹: {test_dir}")
    print(f"[信息] 共找到 {len(image_files)} 张图片")

    # 随机打乱顺序
    random.shuffle(image_files)
    print("[信息] 图片顺序已随机打乱")
    print("-" * 55)
    print("[信息] 每张图片显示 1 秒，按 'q' 键提前退出")
    print("=" * 55)

    # ===== 4) 逐张预测并显示 =====
    total = len(image_files)

    for idx, img_path in enumerate(image_files):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"[警告] 无法读取图片 ({idx+1}/{total}): {img_path}")
            continue

        # 进行 Pose 检测
        img = detector.findPose(img, draw=True)
        landmarks = detector.getLandmarks(poseNo=0)

        # 类别名称（从路径中提取，用于对比真实值）
        parent_folder = Path(img_path).parent.name

        # 预测
        if len(landmarks) == 33:
            features = landmarks_to_features(landmarks)
            if features is not None:
                feature_names = create_feature_names()
                X = pd.DataFrame([features], columns=feature_names)

                prediction = model.predict(X)[0]

                # 获取置信度
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    confidence = max(proba)
                    conf_text = f"Confidence: {confidence:.2%}"
                else:
                    conf_text = ""

                # 在图片上显示预测结果
                color = (0, 255, 0) if prediction == parent_folder else (0, 0, 255)
                cv2.putText(
                    img, f"Pred: {prediction}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3, cv2.LINE_AA
                )
                if conf_text:
                    cv2.putText(
                        img, conf_text,
                        (30, 125), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA
                    )
                cv2.putText(
                    img, f"Actual: {parent_folder}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    img, "Pred: ? (feat err)",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA
                )
        else:
            cv2.putText(
                img, "No Pose Detected",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA
            )

        # 右上角显示进度
        progress_text = f"{idx + 1} / {total}"
        h, w = img.shape[:2]
        cv2.putText(
            img, progress_text,
            (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (200, 200, 200), 2, cv2.LINE_AA
        )

        # 自适应缩放并显示
        display_img = adaptive_resize(img, MAX_DISPLAY_SIZE)
        cv2.imshow("Pose Classification - Batch Predict", display_img)

        # 每张图片显示指定时间，按 'q' 可提前退出
        key = cv2.waitKey(DISPLAY_DELAY) & 0xFF
        if key == ord('q') or key == 27:
            print("[信息] 用户提前退出")
            break

    print("=" * 55)
    print(f"批量预测完成，共处理 {idx + 1} 张图片")
    print("=" * 55)

    cv2.destroyAllWindows()
    detector.close()
