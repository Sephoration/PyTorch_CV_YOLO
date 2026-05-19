import os
import pickle
import cv2
import numpy as np
from PoseModule import PoseDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yoga_pose_knn.pkl")


def landmarks_to_features(landmarks):
    """将 33 个 Pose Landmark 转换为 132 维特征向量"""
    if len(landmarks) != 33:
        return None

    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    return features


if __name__ == '__main__':
    # 加载训练好的 KNN 模型
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件：{MODEL_PATH}")
        print("请先执行 3-3-Train-KNN.py 训练模型。")
        exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"[信息] 模型已加载：{MODEL_PATH}")
    print(f"[信息] 支持类别：{model.classes_}")

    # 初始化摄像头与 PoseDetector（视频模式）
    cap = cv2.VideoCapture(0)
    detector = PoseDetector(mode="VIDEO", model_complexity="full")

    print("[信息] 按 'q' 键退出实时识别")

    while True:
        success, img = cap.read()
        if not success:
            print("[错误] 无法读取摄像头画面")
            break

        # 进行 Pose 检测并绘制骨架
        img = detector.findPose(img, draw=True)
        landmarks = detector.getLandmarks(poseNo=0)

        # 预测并显示结果
        if len(landmarks) == 33:
            features = landmarks_to_features(landmarks)
            if features is not None:
                # KNN predict 需要 2D 数组 (1, 132)
                X = np.array(features).reshape(1, -1)
                prediction = model.predict(X)[0]

                # 在画面上显示预测类别
                cv2.putText(
                    img,
                    f"Pose: {prediction}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )
        else:
            # 未检测到完整人体
            cv2.putText(
                img,
                "No Pose Detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        cv2.imshow("Pose Classification Realtime", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
