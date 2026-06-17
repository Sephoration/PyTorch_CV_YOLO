import cv2
import numpy as np

from yoloTracker import YOLOTracker

########################## 全局设定 ################################
VIDEO_PATH = 'videos/people-1-short.mp4'
RESULT_PATH = "videos_output/4-Zone-1.mp4"
# 指定敏感区域的多边形顶点坐标
polygonPoints = [[710, 200], [1110, 200], [810, 400], [410, 400]]
color_light_yellow = (0, 155, 255)   # Light yellow color
########################## 全局设定 ################################


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def drawAndFillPolygon(image, polygonPoints, fillColor):
    # Convert the polygon points to NumPy array
    polygonPoints = np.array([polygonPoints], dtype=np.int32)

    # Create a mask with the same size as the image
    mask = np.zeros_like(image)

    # Fill the polygon with the specified color on the mask
    mask = cv2.fillPoly(mask, [polygonPoints], color=fillColor)

    # Draw the polygon contour on the original image
    cv2.polylines(image, [polygonPoints], isClosed=True, color=(255, 0, 255), thickness=5)

    # Overlay the mask and the original image
    overlaidImage = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return overlaidImage


if __name__ == '__main__':
    # 开启影片
    cap = cv2.VideoCapture(VIDEO_PATH)
    # 读取第一帧，确保 frame 有值
    success, frame = cap.read()
    if not success:
        print("Error: 无法读取影片，请检查影片路径或影片格式")
        cap.release()
        exit()

    frame_height, frame_width = frame.shape[:2]  # 获取影像高度、宽度
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取影片帧率
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 设定输出影片格式
    out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重新放回第一帧，确保循环内可以处理完整影片

    tracker = YOLOTracker()

    while cap.isOpened():
        success, frame = cap.read()
        if success:

            # 利用 YOLO Tracker 来读取 frame 中的目标框
            frame, list_boxes = tracker.track(frame)



            # Draw the boundary monitoring area 绘制敏感区域
            frame = drawAndFillPolygon(frame, polygonPoints, color_light_yellow)

            # 存储结果
            # out.write(frame)
            # 显示结果
            cv2.imshow("Demo", frame)
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()