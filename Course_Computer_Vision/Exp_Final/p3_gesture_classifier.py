# gesture_classifier.py
"""通用手势分类器 — 动态扫描 models/ 目录加载 .pkl 模型"""
import os, pickle, threading, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def _infer_feature_mode(filename):
    if "_hand_v2" in filename.lower():
        return "hand_v2"
    if "_hand" in filename.lower():
        return "hand"
    return "raw"


def _display_name(key):
    s = key.lower()
    for prefix in ("hand_gesture_", "gesture_", "model_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.upper()


class GestureClassifier:
    def __init__(self):
        self.models = {}
        self.meta = {}  # key → {filename, feature_mode, display}
        self.current = None
        self._lock = threading.Lock()
        self._scan_models()
        if self.available:
            self.current = self.available[0]
            self._load(self.current)  # 加载默认模型

    def _normalize(self, landmarks, mode):
        if mode == "hand":
            return self._normalize_hand(landmarks)
        if mode == "hand_v2":
            return self._normalize_hand_v2(landmarks)
        return self._normalize_raw(landmarks)

    def preload_all(self):
        """后台预加载所有未加载的模型，切换时零等待"""
        for key in self.meta:
            self._load(key)

    def _scan_models(self):
        if not os.path.exists(MODEL_DIR):
            return
        for fn in sorted(os.listdir(MODEL_DIR)):
            if not fn.endswith(".pkl"):
                continue
            key = fn[:-4].lower()
            self.meta[key] = {
                "filename": fn,
                "feature_mode": _infer_feature_mode(fn),
                "display": _display_name(key),
            }

    def _load(self, key):
        """延迟加载单个模型，仅首次调用时读取文件"""
        if key in self.models:
            return self.models[key]
        info = self.meta.get(key)
        if not info:
            return None
        path = os.path.join(MODEL_DIR, info["filename"])
        with open(path, "rb") as f:
            model = pickle.load(f)
        if hasattr(model, "feature_names_in_"):
            try:
                del model.feature_names_in_
            except AttributeError:
                pass
        self.models[key] = model
        return model

    # ==================================================================
    # 归一化（不变）
    # ==================================================================
    @staticmethod
    def _normalize_raw(landmarks):
        wrist = landmarks[0]
        mcp = landmarks[9]
        sx, sy, sz = mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z
        scale = np.sqrt(sx*sx + sy*sy + sz*sz)
        if scale < 1e-6:
            scale = 1.0
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return ((pts - np.array([wrist.x, wrist.y, wrist.z])) / scale).reshape(1, -1)

    @staticmethod
    def _normalize_hand(landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        scale = np.sqrt(dx*dx + dy*dy)
        if scale < 1e-6:
            scale = 1.0
        angle = np.arctan2(dy, dx)
        rot = -np.pi / 2 - angle
        cos_a, sin_a = np.cos(rot), np.sin(rot)
        pts = np.array([[lm.x, lm.y] for lm in landmarks])
        rel = pts - np.array([wrist.x, wrist.y])
        x_rot = cos_a * rel[:, 0] - sin_a * rel[:, 1]
        y_rot = sin_a * rel[:, 0] + cos_a * rel[:, 1]
        out = np.column_stack([x_rot / scale, y_rot / scale])
        return out.reshape(1, -1)

    @staticmethod
    def _normalize_hand_v2(landmarks):
        wrist    = np.array([landmarks[0].x,  landmarks[0].y,  landmarks[0].z])
        index_mcp  = np.array([landmarks[5].x,  landmarks[5].y,  landmarks[5].z])
        middle_mcp = np.array([landmarks[9].x,  landmarks[9].y,  landmarks[9].z])
        pinky_mcp  = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
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
            fallback = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(y_axis, fallback)
            x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return ((pts - wrist) @ R.T / scale).reshape(1, -1)

    def normalize(self, landmarks):
        mode = self.meta[self.current]["feature_mode"]
        return self._normalize(landmarks, mode)

    def predict(self, landmarks):
        with self._lock:
            cur = self.current
            if cur is None:
                return None, None
            model = self._load(cur)
            if model is None:
                return None, None
            mode = self.meta[cur]["feature_mode"]
        feats = self._normalize(landmarks, mode)
        pred = model.predict(feats)[0]
        proba = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
        return int(pred), proba

    def switch(self, name):
        with self._lock:
            if name in self.meta:
                self.current = name
                self._load(name)  # 预加载，避免 Worker 首次预测卡顿

    @property
    def available(self):
        return list(self.meta.keys())

    @property
    def display_names(self):
        return [self.meta[k]["display"] for k in self.available]

    def key_from_display(self, display):
        for k, v in self.meta.items():
            if v["display"] == display.upper():
                return k
        return None
