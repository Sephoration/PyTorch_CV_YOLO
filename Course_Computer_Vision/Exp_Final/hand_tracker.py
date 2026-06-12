import os
import time
import pickle
import cv2
import numpy as np
import mediapipe as mp


class HandTracker:
    def __init__(self, model_path=None, num_hands=2):
        self.num_hands = num_hands

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "hand_gesture_knn.pkl")
        self.model_path = model_path

        self.model = None
        self.class_names = []
        self._load_model()

        task_path = os.path.join(base_dir, "models", "hand_landmarker.task")
        if not os.path.exists(task_path):
            print("[INFO] 下载 HandLandmarker 模型中...")
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/"
                   "hand_landmarker.task")
            urllib.request.urlretrieve(url, task_path)
            print("[INFO] 下载完成")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.HandLandmarker = mp.tasks.vision.HandLandmarker

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = self.HandLandmarker.create_from_options(options)

        self.results = None
        self.predictions = []
        self.raw_landmarks = []

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[警告] 模型文件不存在：{self.model_path}")
            print("请先运行 train_knn.py 训练模型")
            self.model = None
            return
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        self.class_names = list(self.model.classes_)
        print(f"[信息] 模型已加载，支持类别：{self.class_names}")

    def landmarks_to_features(self, landmarks):
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        return features

    def update(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int(time.time() * 1000)
        self.results = self.landmarker.detect_for_video(mp_img, timestamp_ms)

        self.predictions = []
        self.raw_landmarks = []

        if not self.results or not self.results.hand_landmarks:
            return self.predictions

        for hand_landmarks in self.results.hand_landmarks:
            self.raw_landmarks.append(hand_landmarks)

            if self.model is not None:
                features = self.landmarks_to_features(hand_landmarks)
                X = np.array(features).reshape(1, -1)
                pred = self.model.predict(X)[0]
                self.predictions.append(pred)
            else:
                self.predictions.append(None)

        return self.predictions

    def draw_landmarks(self, img):
        if not self.results or not self.results.hand_landmarks:
            return img

        HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        h, w, _ = img.shape

        for i, hand_landmarks in enumerate(self.results.hand_landmarks):
            for connection in HAND_CONNECTIONS:
                x0 = int(hand_landmarks[connection.start].x * w)
                y0 = int(hand_landmarks[connection.start].y * h)
                x1 = int(hand_landmarks[connection.end].x * w)
                y1 = int(hand_landmarks[connection.end].y * h)
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            if i < len(self.predictions) and self.predictions[i] is not None:
                wrist = hand_landmarks[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h) - 40
                label = f"Hand{i}: {self.predictions[i]}"
                cv2.putText(img, label, (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2, cv2.LINE_AA)

        return img

    def get_landmark_pos(self, img, hand_idx=0, landmark_idx=8):
        if not self.results or not self.results.hand_landmarks:
            return None
        if hand_idx >= len(self.results.hand_landmarks):
            return None
        h, w, _ = img.shape
        lm = self.results.hand_landmarks[hand_idx][landmark_idx]
        return (int(lm.x * w), int(lm.y * h))

    def get_finger_distance(self, img, hand_idx=0, tip1=4, tip2=8):
        pos1 = self.get_landmark_pos(img, hand_idx, tip1)
        pos2 = self.get_landmark_pos(img, hand_idx, tip2)
        if pos1 is None or pos2 is None:
            return None
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def close(self):
        if self.landmarker:
            self.landmarker.close()
