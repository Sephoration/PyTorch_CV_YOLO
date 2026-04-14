import os
import time
import urllib.request
import cv2
import mediapipe as mp

# ====== 1) 下载/指定 HandLandmarker 模型 ======
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

def ensure_model(model_path: str, model_url: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] 模型不存在，开始下载：{model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] 下载完成：{model_path}")

class HandDetector:
    def __init__(self, num_hands=2, detectionCon=0.5, presenceCon=0.5, trackingCon=0.5):
        self.num_hands = num_hands
        self.results = None

        # ====== 2) 建立 HandLandmarker（IMAGE 模式）======
        self.BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker

        # 使用新版 HandLandmarker 的连接定义
        self.HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=self.VisionRunningMode.VIDEO,  # 改成 VIDEO 模式
            num_hands=num_hands,  # 图中只有一只手
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=presenceCon,
            min_tracking_confidence=trackingCon,
        )
        self.detector=self.HandLandmarker.create_from_options(options)
        # 修复：将 detector 赋值给 landmarker 属性，保持后续调用一致
        self.landmarker = self.detector

    def findHands(self, img, draw=True, flip=True):
        # 水平翻转图像，使画面与现实一致
        if flip:
            img = cv2.flip(img, 1)

        # 1) BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2) numpy -> mp.Image
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # 3) VIDEO 模式必须给 timestamp(毫秒)
        timestamp_ms = int(time.time() * 1000)

        # 4) 手部关键点检测
        self.results = self.landmarker.detect_for_video(mp_img, timestamp_ms)

        # 5) 绘制手部关键点 + 关键点
        # 修复：判断条件改为检查 hand_landmarks 是否存在
        if draw and self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                h, w, _ = img.shape

                # 手部关键点连线
                for connection in self.HAND_CONNECTIONS:
                    start_idx = connection.start
                    end_idx = connection.end

                    x0 = int(hand_landmarks[start_idx].x * w)
                    y0 = int(hand_landmarks[start_idx].y * h)
                    x1 = int(hand_landmarks[end_idx].x * w)
                    y1 = int(hand_landmarks[end_idx].y * h)

                    cv2.line(img, (x0, y0), (x1, y1), (255, 0, 255), 2)

                # 画关键点
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        return img

    def findPosition(self, img, handNo=0, draw=False):
        # 回传指定手的 21 个关键点座标:[[id, cx, cy], ...]
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

    def close(self):
        """关闭 HandLandmarker 资源"""
        if self.landmarker:
            self.landmarker.close()


if __name__ == '__main__':
    # 修复1：参数顺序错误，应该是 MODEL_PATH 在前，MODEL_URL 在后
    ensure_model(MODEL_PATH, MODEL_URL)
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            continue

        frame=detector.findHands(frame, draw=True)

        # 修复2：取消注释，显示窗口
        cv2.imshow('HandImage', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()