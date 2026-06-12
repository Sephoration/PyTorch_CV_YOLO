# utils/HandModule.py
"""HandDetector — 手部侦测追踪模块（基于新版 MediaPipe Tasks API）

PPT P4 修改要点:
  修改点 1: 加入 mode 参数，支持外部动态指定运行模式 ("VIDEO" / "IMAGE")
  修改点 2: 针对不同运行模式调用不同的 MediaPipe 底层推理 API
  修改点 3: MODEL_PATH 修复为跨目录引用 (../models)，防止跨文件夹导入时找不到模型
  说明:    内部使用 img_rgb 副本做推理，不污染原始 BGR 色域通道
"""

import os
import time
import cv2
import mediapipe as mp

# 修改点 3: 指定上级目录下的 models 文件夹，防止跨目录导入时找不到模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")


class HandDetector:
    # 修改点 1: 加入 mode 参数，允许外部动态注入 "VIDEO" 或 "IMAGE" 运行模式
    def __init__(self, mode="VIDEO", num_hands=2, detectionCon=0.5, presenceCon=0.5,
                 trackingCon=0.5):
        self.mode = mode
        self.num_hands = num_hands
        self.results = None

        # MediaPipe Tasks 新版 API 引用
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode
        self.HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        # 修改点 1 (续): 动态判断运行模式
        if self.mode == "IMAGE":
            mp_running_mode = self.RunningMode.IMAGE
        else:
            mp_running_mode = self.RunningMode.VIDEO

        # 建立 HandLandmarker 实例
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=presenceCon,
            min_tracking_confidence=trackingCon
        )
        self.detector = self.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        """输入 BGR 图像，输出画好关键点与连线的图像。

        说明 (PPT Slide 12):
          MediaPipe 底层要求 RGB 输入，但与工站后台线程 BaseWorker 统一约
          定回传图像格式为 OpenCV 原生的 BGR。因此内部使用 img_rgb 副本做
          推理，不污染原始 BGR 色域通道。
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # 修改点 2: 针对不同运行模式调用不同的 MediaPipe 底层推理 API
        if self.mode == "VIDEO":
            # VIDEO 模式要求严格单调递增的毫秒级时间戳
            timestamp_ms = int(time.time() * 1000)
            self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            # IMAGE 模式仅处理静态单帧，不能传入时间戳
            self.results = self.detector.detect(mp_image)

        if draw and self.results and self.results.hand_landmarks:
            h, w, c = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                # 画连线
                for connection in self.HAND_CONNECTIONS:
                    start_idx = connection.start
                    end_idx = connection.end
                    x0 = int(hand_landmarks[start_idx].x * w)
                    y0 = int(hand_landmarks[start_idx].y * h)
                    x1 = int(hand_landmarks[end_idx].x * w)
                    y1 = int(hand_landmarks[end_idx].y * h)
                    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # 画关键点
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return img

    def findPosition(self, img, handNo=0, draw=False):
        """回传指定手的 21 个关键点坐标: [[id, cx, cy], ...]"""
        lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                h, w, c = img.shape
                myHand = self.results.hand_landmarks[handNo]
                for idx, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([idx, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

    def findPositionDict(self, img, handNo=0, draw=False):
        """回传指定手的 21 个关键点坐标字典: {id: (cx, cy), ...}"""
        lmDict = {}
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                h, w, c = img.shape
                myHand = self.results.hand_landmarks[handNo]
                for idx, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmDict[idx] = (cx, cy)
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmDict

    def close(self):
        """释放底层 MediaPipe 模型资源"""
        if self.detector:
            self.detector.close()
