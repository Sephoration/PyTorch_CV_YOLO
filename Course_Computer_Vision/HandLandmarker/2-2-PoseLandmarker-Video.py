import os
import cv2
from PoseModule import PoseDetector

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
VIDEO_PATH = os.path.join(VIDEO_DIR, "ManRun-1.mp4")

if __name__ == '__main__':
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] 视频文件不存在：{VIDEO_PATH}")
        print("请将视频文件放入 videos/ 文件夹后重试")
        exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    detector = PoseDetector(mode="VIDEO", model_complexity="full")

    print("[INFO] 开始处理视频，按 'q' 或 ESC 键退出")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[INFO] 视频播放完毕")
                break

            frame = detector.findPose(frame, draw=True)

            cv2.imshow("Pose Landmarker Video", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[INFO] 退出程序")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
