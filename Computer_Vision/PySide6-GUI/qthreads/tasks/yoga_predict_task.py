import cv2
import pickle
import pandas as pd
from qthreads.tasks.base_task import BaseCVTask
from utils.PoseModule import PoseDetector


class YogaPredictTask(BaseCVTask):
    def __init__(self, model_path="./models/yoga_pose_knn.pkl"):
        super().__init__()
        self.model_path = model_path
        self.detector = None
        self.model = None
        self.feature_names = self._create_feature_names()

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"YogaPredictTask 模型加载失败: {e}")

    def _create_feature_names(self):
        return [f"{ax}{i}" for i in range(33) for ax in ['x', 'y', 'z', 'v']]

    def _landmarks_to_features(self, landmarks):
        if len(landmarks) != 33: return None
        return [val for lm in landmarks for val in [lm.x, lm.y, lm.z, lm.visibility]]

    def process(self, frame):
        # 1. 延迟加载 Detector，并强制设为 VIDEO 模式以支持 timestamp
        if self.detector is None:
            self.detector = PoseDetector(
                mode="VIDEO",
                model_complexity="full"
            )

        # 2. 推理
        frame = self.detector.findPose(frame, draw=True)
        landmarks = self.detector.getLandmarks(poseNo=0)

        predicted_class = "等待检测..."
        confidence = None

        if landmarks:
            features = self._landmarks_to_features(landmarks)
            if features:
                X = pd.DataFrame([features], columns=self.feature_names)
                predicted_class = self.model.predict(X)[0]
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)[0]
                    confidence = probs.max()

        text = f"Yoga: {predicted_class} ({confidence:.2f})" if confidence else f"Yoga: {predicted_class}"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        return frame, {"status": text}

    def close(self):
        if self.detector:
            self.detector.close()