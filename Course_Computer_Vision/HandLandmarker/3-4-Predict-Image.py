import os
import pickle
import cv2
import numpy as np
import pandas as pd
from PoseModule import PoseDetector

####################### 设定全局常数 #######################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yoga_pose_knn.pkl")

# 默认测试图片（可修改为任意图片路径）
DEFAULT_IMAGE_PATH = os.path.join(BASE_DIR, "images", "PoseEstimation", "Pose-1.jpg")
####################### 设定全局常数 #######################


def create_feature_names():
    """
    创建 132 个特征列名：x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32
    对应 33 个 Pose Landmark，每个 landmark 有 4 个值 (x, y, z, visibility)
    """
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


def predict_image(image_path, model, detector):
    """
    对单张图片进行姿势分类预测

    Args:
        image_path: 图片路径
        model: 已加载的 KNN 模型
        detector: PoseDetector 实例

    Returns:
        (prediction, confidence, annotated_img) 或 (None, None, img)
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return None, None, None

    print(f"[信息] 读取图片: {image_path}")
    print(f"[信息] 图片尺寸: {img.shape[1]} x {img.shape[0]}")

    # 进行 Pose 检测（IMAGE 模式），同时绘制骨架
    img = detector.findPose(img, draw=True)
    landmarks = detector.getLandmarks(poseNo=0)

    if len(landmarks) != 33:
        print("[警告] 未检测到完整人体（33 个关键点）")
        cv2.putText(
            img, "No Pose Detected",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 0, 255), 3, cv2.LINE_AA
        )
        return None, None, img

    # 转换为特征向量
    features = landmarks_to_features(landmarks)
    if features is None:
        print("[错误] 特征转换失败")
        return None, None, img

    # 构建单行 DataFrame（必须与训练时的列名一致）
    feature_names = create_feature_names()
    X = pd.DataFrame([features], columns=feature_names)

    # 预测
    prediction = model.predict(X)[0]

    # 获取置信度（predict_proba 返回所有类别的概率）
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = max(proba)
    else:
        confidence = None

    # 在图片上显示预测结果
    result_text = f"Prediction: {prediction}"
    cv2.putText(
        img, result_text,
        (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (0, 255, 0), 3, cv2.LINE_AA
    )

    if confidence is not None:
        conf_text = f"Confidence: {confidence:.2%}"
        cv2.putText(
            img, conf_text,
            (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 2, cv2.LINE_AA
        )

    return prediction, confidence, img


if __name__ == '__main__':
    # ===== 1) 加载模型 =====
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件：{MODEL_PATH}")
        print("请先执行 3-3-Train-KNN.py 训练模型。")
        exit(1)

    print("=" * 55)
    print("Pose Classification - 单张图片预测")
    print("=" * 55)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"[信息] 模型已加载：{MODEL_PATH}")
    print(f"[信息] 支持类别：{model.classes_}")
    print(f"[信息] 特征维度：{model.n_features_in_}")
    print("-" * 55)

    # ===== 2) 初始化 PoseDetector（IMAGE 模式） =====
    detector = PoseDetector(mode="IMAGE", model_complexity="full")

    # ===== 3) 预测 =====
    prediction, confidence, annotated_img = predict_image(
        DEFAULT_IMAGE_PATH, model, detector
    )

    # ===== 4) 输出结果 =====
    print("-" * 55)
    if prediction is not None:
        print(f"[结果] 预测类别: {prediction}")
        if confidence is not None:
            print(f"[结果] 置信度: {confidence:.2%}")
    else:
        print("[结果] 未检测到人体，无法预测")
    print("=" * 55)

    # ===== 5) 显示图片 =====
    if annotated_img is not None:
        cv2.imshow("Pose Classification - Image", annotated_img)
        print("[信息] 按任意键关闭窗口")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    detector.close()
