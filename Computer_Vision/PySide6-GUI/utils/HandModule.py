import os
import time
import urllib.request
import cv2
import mediapipe as mp

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models", "hand_landmarker.task")


def ensure_model(model_path: str, model_url: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] 模型不存在，开始下载：{model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] 下载完成：{model_path}")


class HandDetector:
    def __init__(self, mode="VIDEO", num_hands=2, detectionCon=0.5, presenceCon=0.5, trackingCon=0.5):
        # 🎯 修复 1：在类初始化时强制检查模型，确保跨文件调用时绝对安全
        ensure_model(MODEL_PATH, MODEL_URL)

        self.mode = mode
        self.num_hands = num_hands
        self.results = None

        # 🎯 修复 2：为 VIDEO 模式准备一个可靠的内部时钟起点
        self.start_time = time.perf_counter()

        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode
        self.HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        if self.mode == "IMAGE":
            mp_running_mode = self.RunningMode.IMAGE
        else:
            mp_running_mode = self.RunningMode.VIDEO

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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        if self.mode == "VIDEO":
            # 🎯 修复 2：使用 perf_counter 计算相对毫秒差，保证时间戳绝对单调递增
            current_time = time.perf_counter()
            timestamp_ms = int((current_time - self.start_time) * 1000)

            # 安全防线：如果运算过快导致毫秒级时间戳重复，人为+1
            if hasattr(self, 'last_timestamp_ms') and timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms

            self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            self.results = self.detector.detect(mp_image)

        if draw and self.results and self.results.hand_landmarks:
            h, w, c = img.shape

            for hand_landmarks in self.results.hand_landmarks:
                for connection in self.HAND_CONNECTIONS:
                    start_idx = connection.start
                    end_idx = connection.end

                    x0 = int(hand_landmarks[start_idx].x * w)
                    y0 = int(hand_landmarks[start_idx].y * h)
                    x1 = int(hand_landmarks[end_idx].x * w)
                    y1 = int(hand_landmarks[end_idx].y * h)

                    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return img

    def findPosition(self, img, handNo=0, draw=False):
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
        self.detector.close()


if __name__ == "__main__":
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(mode="VIDEO")

    while True:
        success, frame = cap.read()
        if not success:
            print("无法读取摄像头画面")
            break

        frame = detector.findHands(frame, draw=True)
        lmList = detector.findPosition(frame, handNo=0, draw=False)

        if len(lmList) != 0: print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()