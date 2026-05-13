import os
import urllib.request
import cv2
import mediapipe as mp

POSE_MODEL_CONFIGS = {
    "lite": {
        "filename": "pose_landmarker_lite.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
               "float16/latest/pose_landmarker_lite.task"
    },
    "full": {
        "filename": "pose_landmarker_full.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/"
               "float16/latest/pose_landmarker_full.task"
    },
    "heavy": {
        "filename": "pose_landmarker_heavy.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/"
               "float16/latest/pose_landmarker_heavy.task"
    }
}

def ensure_model(model_path: str, model_url: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] 模型不存在，开始下载：{model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] 下载完成：{model_path}")

class PoseDetector:
    def __init__(self, mode="VIDEO", model_complexity="full", model_path=None,
                 num_poses=1, detectionCon=0.5, presenceCon=0.5,
                 trackingCon=0.5, output_segmentation_masks=False):

        self.mode = mode.upper()
        self.model_complexity = model_complexity.lower()
        self.num_poses = num_poses
        self.detectionCon = detectionCon
        self.presenceCon = presenceCon
        self.trackingCon = trackingCon
        self.output_segmentation_masks = output_segmentation_masks

        self.results = None
        self.timestamp_ms = 0

        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        self.POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

        if model_path is not None:
            self.model_path = model_path
            self.model_url = None
        else:
            if self.model_complexity not in POSE_MODEL_CONFIGS:
                raise ValueError("model_complexity 只支持: 'lite', 'full', 'heavy'")

            model_info = POSE_MODEL_CONFIGS[self.model_complexity]
            self.model_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                model_info["filename"]
            )
            self.model_url = model_info["url"]

            ensure_model(self.model_path, self.model_url)

        if self.mode == "IMAGE":
            running_mode = self.RunningMode.IMAGE
        else:
            running_mode = self.RunningMode.VIDEO

        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=running_mode,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.detectionCon,
            min_pose_presence_confidence=self.presenceCon,
            min_tracking_confidence=self.trackingCon,
            output_segmentation_masks=self.output_segmentation_masks
        )

        self.detector = self.PoseLandmarker.create_from_options(options)

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        if self.mode == "IMAGE":
            self.results = self.detector.detect(mp_image)
        elif self.mode == "VIDEO":
            self.timestamp_ms += 1
            self.results = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        else:
            raise ValueError("mode 只支持: 'IMAGE', 'VIDEO'")

        if draw:
            self.drawPose(img)

        return img

    def drawPose(self, img, poseNo=0, draw_points=True, draw_lines=True):
        if self.results is None or not self.results.pose_landmarks:
            return img

        if poseNo >= len(self.results.pose_landmarks):
            return img

        h, w, c = img.shape
        pose_landmarks = self.results.pose_landmarks[poseNo]

        if draw_lines:
            for connection in self.POSE_CONNECTIONS:
                start_idx = connection.start
                end_idx = connection.end

                lm_start = pose_landmarks[start_idx]
                lm_end = pose_landmarks[end_idx]

                if lm_start.visibility < 0.5 or lm_end.visibility < 0.5:
                    continue

                x0 = int(lm_start.x * w)
                y0 = int(lm_start.y * h)
                x1 = int(lm_end.x * w)
                y1 = int(lm_end.y * h)

                cv2.line(img, (x0, y0), (x1, y1), (255, 0, 255), 2)

        if draw_points:
            for i, lm in enumerate(pose_landmarks):
                if lm.visibility < 0.5:
                    continue

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        return img

    def findPosition(self, img, poseNo=0, draw=False):
        lmList = []

        if self.results is None or not self.results.pose_landmarks:
            return lmList

        if poseNo >= len(self.results.pose_landmarks):
            return lmList

        h, w, c = img.shape
        myPose = self.results.pose_landmarks[poseNo]

        for idx, lm in enumerate(myPose):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            lmList.append([idx, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList

    def findPositionDict(self, img, poseNo=0, draw=False):
        lmDict = {}

        if self.results is None or not self.results.pose_landmarks:
            return lmDict

        if poseNo >= len(self.results.pose_landmarks):
            return lmDict

        h, w, c = img.shape
        myPose = self.results.pose_landmarks[poseNo]

        for idx, lm in enumerate(myPose):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            lmDict[idx] = (cx, cy)

            if draw:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmDict

    def getLandmarks(self, poseNo=0):
        if self.results is None or not self.results.pose_landmarks:
            return []

        if poseNo >= len(self.results.pose_landmarks):
            return []

        return self.results.pose_landmarks[poseNo]

    def close(self):
        if self.detector:
            self.detector.close()


if __name__ == '__main__':
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images", "PoseEstimation", "Pose-4.jpg")

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{IMAGE_PATH}")

    detector = PoseDetector(mode="IMAGE", model_complexity="heavy")
    image = detector.findPose(image, draw=True)

    if detector.results and detector.results.pose_landmarks:
        print(f"检测到 {len(detector.results.pose_landmarks)} 个人体")
    else:
        print("没有检测到人体")

    lmList = detector.findPosition(image, poseNo=0, draw=False)
    if len(lmList) != 0:
        print("前5个关键点：")
        print(lmList[:5])

    lmDict = detector.findPositionDict(image, poseNo=0, draw=False)
    if len(lmDict) != 0:
        print("前5个关键点字典：")
        first_keys = list(lmDict.keys())[:5]
        print({k: lmDict[k] for k in first_keys})

    cv2.imshow("Pose Image Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.close()
