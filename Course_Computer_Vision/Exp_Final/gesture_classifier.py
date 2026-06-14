# gesture_classifier.py
"""KNN/SVM 手势分类器 — 与 extract_features.py 归一化算法一致"""
import os, pickle, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNN_PATH = os.path.join(BASE_DIR, "models", "hand_gesture_knn.pkl")
SVM_PATH = os.path.join(BASE_DIR, "models", "hand_gesture_svm.pkl")


class GestureClassifier:
    def __init__(self):
        self.knn = self.svm = None
        self.current = "knn"
        if os.path.exists(KNN_PATH):
            with open(KNN_PATH, "rb") as f:
                self.knn = pickle.load(f)
        if os.path.exists(SVM_PATH):
            with open(SVM_PATH, "rb") as f:
                self.svm = pickle.load(f)

    @staticmethod
    def normalize(landmarks):
        """归一化：以手腕(0)为原点，以手部尺度为基准"""
        wrist = landmarks[0]
        mcp = landmarks[9]
        sx, sy, sz = mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z
        scale = np.sqrt(sx*sx + sy*sy + sz*sz)
        if scale < 1e-6:
            scale = 1.0
        feats = []
        for lm in landmarks:
            feats.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
        return np.array(feats).reshape(1, -1)

    def predict(self, landmarks):
        model = getattr(self, self.current, None)
        if model is None:
            return None, None
        feats = self.normalize(landmarks)
        pred = model.predict(feats)[0]
        proba = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
        return int(pred), proba

    def switch(self, name):
        if getattr(self, name, None) is not None:
            self.current = name

    @property
    def available(self):
        return [m for m in ["knn", "svm"] if getattr(self, m)]
