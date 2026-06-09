# utils/HandModule.py
"""HandModule — 手部侦测追踪模块（供 HandVolumeTask 使用）"""

import cv2
import numpy as np


class HandModule:
    """手部检测与追踪模块。

    基于 MediaPipe Hands 实现手部关键点检测，
    当前提供骨架占位，后续课程扩展完整手势识别功能。
    """

    def __init__(self):
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self._loaded = True
        except ImportError:
            print("【HandModule】mediapipe 未安装，手部模块将以降级模式运行")

    def process(self, frame: np.ndarray):
        """检测手部并返回标注后的画面和关键点数据。

        Returns:
            (annotated_frame, info_dict)
        """
        info = {"hand_count": 0, "landmarks": [], "volume": 0.0}

        if not self._loaded or self.hands is None:
            cv2.putText(frame, "MediaPipe not installed", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame, {**info, "status": "mediapipe unavailable"}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            info["hand_count"] = len(result.multi_hand_landmarks)
            for hand_lms in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # 计算拇指 (4) 和食指 (8) 指尖距离 → 音量映射
                h, w = frame.shape[:2]
                thumb = hand_lms.landmark[4]
                index = hand_lms.landmark[8]
                dx = (thumb.x - index.x) * w
                dy = (thumb.y - index.y) * h
                distance = np.hypot(dx, dy)
                # 映射到 0.0 ~ 1.0
                vol = min(distance / 200.0, 1.0)
                info["volume"] = vol
                info["landmarks"].append({
                    "thumb": (int(thumb.x * w), int(thumb.y * h)),
                    "index": (int(index.x * w), int(index.y * h)),
                    "distance": distance,
                })
                # 画连线
                cv2.line(frame, info["landmarks"][-1]["thumb"],
                         info["landmarks"][-1]["index"], (0, 255, 0), 3)
                cv2.putText(frame, f"Vol: {vol:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        info["status"] = f"hands: {info['hand_count']}"
        return frame, info

    def release(self):
        if self.hands:
            self.hands.close()
        self._loaded = False
