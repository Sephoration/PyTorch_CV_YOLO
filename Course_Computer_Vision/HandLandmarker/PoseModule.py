import os
import cv2
import urllib.request
import mediapipe as mp

# ====== PoseLandmarker 模型配置 ======
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
        # ====== 1) 保存参数 ======
        self.mode = mode.upper()
        self.model_complexity = model_complexity.lower()
        self.num_poses = num_poses
        self.detectionCon = detectionCon
        self.presenceCon = presenceCon
        self.trackingCon = trackingCon
        self.output_segmentation_masks = output_segmentation_masks

        # ====== 2) 保存检测结果 ======
        self.results = None
        self.timestamp_ms = 0

        # ====== 3) MediaPipe 相关类别名 ======
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        # ====== 4) Pose 骨架连接定义 ======
        self.POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

        # ====== 5) 决定模型路径与模型 URL ======
        if model_path is not None:
            # 使用者手动指定模型路径时，直接使用该路径
            self.model_path = model_path
            self.model_url = None
        else:
            if self.model_complexity not in POSE_MODEL_CONFIGS:
                raise ValueError(
                    "model_complexity 只支持: 'lite', 'full', 'heavy'"
                )

            model_info = POSE_MODEL_CONFIGS[self.model_complexity]
            self.model_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                model_info["filename"]
            )
            self.model_url = model_info["url"]

            # 自动下载所需模型
            ensure_model(self.model_path, self.model_url)

        # ====== 6) 决定运行模式 ======
        if self.mode == "IMAGE":
            running_mode = self.RunningMode.IMAGE
        else:
            running_mode = self.RunningMode.VIDEO

        # ====== 7) 建立 PoseLandmarker 参数 ======
        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=running_mode,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.detectionCon,
            min_pose_presence_confidence=self.presenceCon,
            min_tracking_confidence=self.trackingCon,
            output_segmentation_masks=self.output_segmentation_masks
        )

        # ====== 8) 建立 PoseLandmarker 检测器 ======
        self.detector = self.PoseLandmarker.create_from_options(options)

    def findPose(self, img, draw=True):
        """
        输入:
            img  : OpenCV 读取的 BGR 图像
            draw : 是否将骨架与关键点画到图像上

        输出:
            img  : 处理后的图像
        """

        # ====== 1) BGR 转 RGB ======
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ====== 2) 转换成 MediaPipe Image ======
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # ====== 3) 根据模式执行姿态检测 ======
        if self.mode == "IMAGE":
            self.results = self.detector.detect(mp_image)

        elif self.mode == "VIDEO":
            self.timestamp_ms += 1
            self.results = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        else:
            raise ValueError("mode 只支持: 'IMAGE', 'VIDEO'")

        # ====== 4) 需要时绘制骨架 ======
        if draw:
            self.drawPose(img)

        return img

    def drawPose(self, img, poseNo=0, draw_points=True, draw_lines=True):
        """
        输入:
            img         : OpenCV 读取的 BGR 图像
            poseNo      : 指定要绘制的第几个人体
            draw_points : 是否绘制关键点
            draw_lines  : 是否绘制骨架连线

        输出:
            img         : 绘制后的图像
        """

        # ====== 1) 没有检测结果，直接返回 ======
        if self.results is None or not self.results.pose_landmarks:
            return img

        # ====== 2) 检查 poseNo 是否超出范围 ======
        if poseNo >= len(self.results.pose_landmarks):
            return img

        h, w, c = img.shape
        pose_landmarks = self.results.pose_landmarks[poseNo]

        # ====== 3) 绘制骨架连线 ======
        if draw_lines:
            for connection in self.POSE_CONNECTIONS:
                start_idx = connection.start
                end_idx = connection.end

                lm_start = pose_landmarks[start_idx]
                lm_end = pose_landmarks[end_idx]

                # 可见度太低就不画线
                if lm_start.visibility < 0.5 or lm_end.visibility < 0.5:
                    continue

                x0 = int(lm_start.x * w)
                y0 = int(lm_start.y * h)
                x1 = int(lm_end.x * w)
                y1 = int(lm_end.y * h)

                cv2.line(img, (x0, y0), (x1, y1), (255, 0, 255), 2)

        # ====== 4) 绘制关键点 ======
        if draw_points:
            for i, lm in enumerate(pose_landmarks):
                if lm.visibility < 0.5:
                    continue

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        return img

    def findPosition(self, img, poseNo=0, draw=False):
        """
        输入:
            img    : OpenCV 读取的 BGR 图像
            poseNo : 指定要读取的第几个人体
            draw   : 是否在图像上额外绘制关键点

        输出:
            lmList : 指定人体的关键点像素坐标列表
                     格式: [[id, cx, cy], ...]
        """

        lmList = []      #先建立空列表

        # ====== 1) 检查是否有检测结果 ======
        if self.results is None or not self.results.pose_landmarks:
            return lmList

        # ====== 2) 检查 poseNo 是否超出范围 ======
        if poseNo >= len(self.results.pose_landmarks):
            return lmList

        h, w, c = img.shape
        myPose = self.results.pose_landmarks[poseNo]

        # ====== 3) 遍历该人体的所有关键点 ======
        for idx, lm in enumerate(myPose):
            # 可见度太低就跳过
            if lm.visibility < 0.5:
                continue

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmList.append([idx, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return lmList

    def findPositionDict(self, img, poseNo=0, draw=False):
        """
        输入:
            img    : OpenCV 读取的 BGR 图像
            poseNo : 指定要读取的第几个人体
            draw   : 是否在图像上额外绘制关键点

        输出:
            lmDict : 指定人体的关键点像素坐标字典
                     格式: {id: (cx, cy), ...}
        """

        lmDict = {}

        # ====== 1) 检查是否有检测结果 ======
        if self.results is None or not self.results.pose_landmarks:
            return lmDict

        # ====== 2) 检查 poseNo 是否超出范围 ======
        if poseNo >= len(self.results.pose_landmarks):
            return lmDict

        h, w, c = img.shape
        myPose = self.results.pose_landmarks[poseNo]

        # ====== 3) 遍历该人体的所有关键点 ======
        for idx, lm in enumerate(myPose):
            # 可见度太低就跳过
            if lm.visibility < 0.5:
                continue

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmDict[idx] = (cx, cy)

            if draw:
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return lmDict

    def getLandmarks(self, poseNo=0):
        """
        输入:
            poseNo : 指定要读取的第几个人体

        输出:
            landmarks : 指定人体的原始 landmark 对象列表
        """

        # ====== 1) 检查是否有检测结果 ======
        if self.results is None or not self.results.pose_landmarks:
            return []

        # ====== 2) 检查 poseNo 是否超出范围 ======
        if poseNo >= len(self.results.pose_landmarks):
            return []

        # ====== 3) 回传指定人体的原始 landmarks ======
        return self.results.pose_landmarks[poseNo]

    def close(self):
        self.detector.close()

if __name__ == '__main__':
    IMAGE_PATH = "images/Pose-4.jpg"

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{IMAGE_PATH}")

    detector = PoseDetector(mode="IMAGE", model_complexity="heavy")
    image = detector.findPose(image, draw=True)

    # 查看是否有检测结果
    if detector.results and detector.results.pose_landmarks:
        print(f"检测到 {len(detector.results.pose_landmarks)} 个人体")
    else:
        print("没有检测到人体")

    lmList = detector.findPosition(image, poseNo=0, draw=False)

    if len(lmList) != 0:
        print("前5个关键点：")
        print(lmList[:5])
    else:
        print("没有读取到关键点")

    lmDict = detector.findPositionDict(image, poseNo=0, draw=False)
    if len(lmDict) != 0:
        print("前5个关键点字典：")
        first_keys = list(lmDict.keys())[:5]
        print({k: lmDict[k] for k in first_keys})


    landmarks = detector.getLandmarks(poseNo=0)
    if len(landmarks) != 0:
        print("第 0 个关键点的原始信息：")
        print(f"x = {landmarks[0].x:.4f}")
        print(f"y = {landmarks[0].y:.4f}")
        print(f"z = {landmarks[0].z:.4f}")
        print(f"visibility = {landmarks[0].visibility:.4f}")
    else:
        print("没有读取到原始 landmarks")


    cv2.imshow("Pose Image Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.close()