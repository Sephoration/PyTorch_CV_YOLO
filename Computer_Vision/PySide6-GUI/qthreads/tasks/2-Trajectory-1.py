import cv2
import numpy as np
from yoloTracker import YOLOTracker


####################### 全局设定 #############################
# VIDEO_PATH = 'videos/people-1-short.mp4'
# VIDEO_PATH = 'videos/Traffic-1.mp4'
VIDEO_PATH = 'videos/1162199305-1-192.mp4'
RESULT_PATH = "videos_output/2-Trajectory-1.mp4"
cv2.namedWindow('YOLO Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Tracking', 960, 540)
cv2.moveWindow('YOLO Tracking', 300, 200)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # 设定输出影片格式
####################### 全局设定 #############################

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections=[]

    def add(self, xyxy, tracker_id):
        self.detections.append((xyxy, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail


if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)   # 开启影片

    # 读取第一帧，确保 frame 有值
    success, frame = cap.read()
    if not success:
        print("Error: 无法读取影片，请检查影片路径或影片格式")
        cap.release()
        exit()
    # 获取影像高度、宽度
    frame_height, frame_width = frame.shape[:2]
    # 重新放回第一帧，确保循环内可以处理完整影片
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # 建立输出影片对象
    out = cv2.VideoWriter(RESULT_PATH, fourcc, 30, (frame_width, frame_height))

    # Dictionary to store the trail points of each object
    object_trails = {}
    lost_frames_counter = {}   # 记录每个目标未侦测到的连续帧数

    tracker = YOLOTracker()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 利用 YOLO Tracker 来读取 frame 中的目标框
        myDetections = Detections()
        output_image_frame, list_boxes = tracker.track(frame)
        for item_bbox in list_boxes:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            myDetections.add((x1, y1, x2, y2), track_id)

        # Add the current object‘s position to the trail
        current_ids = []
        for xyxy, track_id in myDetections.detections:
            x1, y1, x2, y2 = xyxy
            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            current_ids.append(track_id)  # 收集目前有侦测到的 ID
            if track_id in object_trails:
                object_trails[track_id].append((center.x, center.y))
            else:
                object_trails[track_id] = [(center.x, center.y)]

            # 重置此 ID 的失踪帧数
            lost_frames_counter[track_id] = 0

        # Draw the trail for each object
        trail_colors = [(255, 0, 255)] * len(object_trails)  # Magenta color（洋红色）  for all trails
        draw_trail(output_image_frame, list(object_trails.values()), trail_colors)

        # Remove trails of objects that are not detected in the current frame
        # for track_id in list(object_trails.keys()):
        #     if track_id not in [item[1] for item in myDetections.detections]:
        #         object_trails.pop(track_id)
        # 修改 trail 删除逻辑：加上等待时间
        remove_ids = []
        for track_id in object_trails:
            if track_id not in current_ids:
                # 记录未侦测帧数，若超过阈值才准备删除
                lost_frames_counter[track_id] = lost_frames_counter.get(track_id, 0) + 1
                if lost_frames_counter[track_id] > 20:  # 你可以调整这个阈值(这里是 20 帧)
                    remove_ids.append(track_id)

                if len(object_trails[track_id]) > 0:
                    object_trails[track_id].pop(0)

        for tid in remove_ids:
            object_trails.pop(tid)
            lost_frames_counter.pop(tid)  # 清除对应的计数器

        # 存储结果
        # out.write(frame)
        # 显示结果
        cv2.imshow('YOLO Tracking', frame)

        # 按 ESC 或 'q' 退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()