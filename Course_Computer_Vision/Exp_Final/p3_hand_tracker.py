# hand_tracker.py
"""MediaPipe 手部关键点检测器（21点提取，不做分类）"""
import os, time, cv2, numpy as np, mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

DETECT_W, DETECT_H = 320, 240  # 检测用小分辨率，大幅提速


class HandTracker:
    def __init__(self, detection_con=0.75):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self.connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.results = None

    def detect(self, display_frame):
        """检测手部。内部缩放到 320×240 做推理，返回归一化坐标的 results。
        传入 display_frame 仅用于获取宽高比例信息。
        """
        # 缩放到小分辨率 → MediaPipe 推理快 10 倍
        small = cv2.resize(display_frame, (DETECT_W, DETECT_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if hasattr(self, '_last_ts') and ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        self.results = self.detector.detect_for_video(mp_img, ts)
        return self.results

    def draw(self, frame):
        """在 frame 上绘制骨架和关键点（基于归一化坐标，适配任意分辨率）"""
        if not self.results or not self.results.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand_lms in self.results.hand_landmarks:
            for conn in self.connections:
                x0 = int(hand_lms[conn.start].x * w)
                y0 = int(hand_lms[conn.start].y * h)
                x1 = int(hand_lms[conn.end].x * w)
                y1 = int(hand_lms[conn.end].y * h)
                cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        return frame

    def close(self):
        self.detector.close()
