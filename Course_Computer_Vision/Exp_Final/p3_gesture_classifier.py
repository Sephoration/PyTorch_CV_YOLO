# gesture_classifier.py
"""KNN/SVM 手势分类器 — 支持原始坐标和手掌坐标系两种特征"""
import os, pickle, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型定义：key → (文件名, 特征模式)
MODEL_DEFS = {
    "knn":          ("hand_gesture_knn.pkl",          "raw"),
    "svm":          ("hand_gesture_svm.pkl",          "raw"),
    "knn_hand":     ("hand_gesture_knn_hand.pkl",     "hand"),
    "svm_hand":     ("hand_gesture_svm_hand.pkl",     "hand"),
    "knn_hand_v2":  ("hand_gesture_knn_hand_v2.pkl",  "hand_v2"),
    "svm_hand_v2":  ("hand_gesture_svm_hand_v2.pkl",  "hand_v2"),
}


class GestureClassifier:
    def __init__(self):
        self.models = {}
        self.current = "knn"
        for key, (filename, _) in MODEL_DEFS.items():
            path = os.path.join(BASE_DIR, "models", filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[key] = pickle.load(f)

    # ==================================================================
    # 原始归一化（相机坐标系）
    # ==================================================================
    @staticmethod
    def _normalize_raw(landmarks):
        wrist = landmarks[0]
        mcp = landmarks[9]
        sx, sy, sz = mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z
        scale = np.sqrt(sx*sx + sy*sy + sz*sz)
        if scale < 1e-6:
            scale = 1.0
        feats = []
        for lm in landmarks:
            feats.extend([(lm.x - wrist.x) / scale,
                          (lm.y - wrist.y) / scale,
                          (lm.z - wrist.z) / scale])
        return np.array(feats).reshape(1, -1)

    # ==================================================================
    # 手掌坐标系归一化（2D 旋转对齐，旋转不变）
    # ==================================================================
    @staticmethod
    def _normalize_hand(landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # 2D 方向：手腕 → 中指根
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        scale = np.sqrt(dx*dx + dy*dy)
        if scale < 1e-6:
            scale = 1.0

        # 旋转角：让中指根始终指向"正上方"（-pi/2）
        angle = np.arctan2(dy, dx)
        rot = -np.pi / 2 - angle
        cos_a, sin_a = np.cos(rot), np.sin(rot)

        feats = []
        for lm in landmarks:
            rx = lm.x - wrist.x
            ry = lm.y - wrist.y
            rz = lm.z - wrist.z
            # 2D 旋转（z 不动）
            x_rot = cos_a * rx - sin_a * ry
            y_rot = sin_a * rx + cos_a * ry
            feats.extend([x_rot / scale, y_rot / scale, rz / scale])
        return np.array(feats).reshape(1, -1)

    # ==================================================================
    # 3D 手掌坐标系归一化（完整三维正交基，消除手掌翻转影响）
    # ==================================================================
    @staticmethod
    def _normalize_hand_v2(landmarks):
        """用 4 个掌骨点构造 3D 正交基，投影 21 点到手掌局部坐标系
         Y   = wrist(0) → middle_mcp(9)    手掌长度方向
         X   = index_mcp(5) → pinky_mcp(17) 手掌宽度方向（Gram-Schmidt 正交化）
         Z   = X × Y                        手掌法向量
         """
        wrist    = np.array([landmarks[0].x,  landmarks[0].y,  landmarks[0].z])
        index_mcp  = np.array([landmarks[5].x,  landmarks[5].y,  landmarks[5].z])
        middle_mcp = np.array([landmarks[9].x,  landmarks[9].y,  landmarks[9].z])
        pinky_mcp  = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

        # Y 轴：手腕 → 中指根（手掌长度）
        y_axis = middle_mcp - wrist
        scale = np.linalg.norm(y_axis)
        if scale < 1e-6:
            scale = 1.0
        y_axis = y_axis / scale

        # X 轴：食指根 → 小指根（手掌宽度），Gram-Schmidt 正交化
        x_raw = pinky_mcp - index_mcp
        x_axis = x_raw - np.dot(x_raw, y_axis) * y_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            # 退路：食指根和小指根在画面重叠时，取任意垂直于 Y 的向量
            fallback = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(y_axis, fallback)
            x_axis = x_axis / np.linalg.norm(x_axis)

        # Z 轴：叉积（手掌法向量，垂直掌面指向外）
        z_axis = np.cross(x_axis, y_axis)

        # 旋转矩阵（列向量是基向量）
        R = np.column_stack([x_axis, y_axis, z_axis])

        feats = []
        for lm in landmarks:
            p = np.array([lm.x, lm.y, lm.z])
            p_local = R.T @ (p - wrist)  # 投影到手掌坐标系
            feats.extend([p_local[0] / scale, p_local[1] / scale, p_local[2] / scale])
        return np.array(feats).reshape(1, -1)

    # ==================================================================
    def normalize(self, landmarks):
        """根据当前模型模式选择归一化方法"""
        mode = MODEL_DEFS[self.current][1]
        if mode == "hand":
            return self._normalize_hand(landmarks)
        if mode == "hand_v2":
            return self._normalize_hand_v2(landmarks)
        return self._normalize_raw(landmarks)

    def predict(self, landmarks):
        model = self.models.get(self.current)
        if model is None:
            return None, None
        feats = self.normalize(landmarks)
        pred = model.predict(feats)[0]
        proba = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
        return int(pred), proba

    def switch(self, name):
        if name in self.models:
            self.current = name

    @property
    def available(self):
        return list(self.models.keys())
