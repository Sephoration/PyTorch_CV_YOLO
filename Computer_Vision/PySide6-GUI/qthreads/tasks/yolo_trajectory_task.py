import os
import sys

# ====================================================================
# 👑 动态将项目根目录挂载到系统环境变量
# ====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
from qthreads.tasks.base_task import BaseCVTask
from utils.yoloTracker import YOLOTracker


# ====================================================================
# 🎓 教学保留区
# ====================================================================
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, tracker_id):
        self.detections.append((xyxy, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j - 1][0]), int(trail_points[i][j - 1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)


# ====================================================================
# 🚀 核心任务外挂
# ====================================================================
class YoloTrajectoryTask(BaseCVTask):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.object_trails = {}
        self.lost_frames_counter = {}

        self.show_trail = True
        self.trail_length = 50
        self.lost_threshold = 20

    def update_special_params(self, param_dict):
        # --- 接收 Tab 6 传来的特效与业务逻辑参数 ---
        if "show_trail" in param_dict:
            self.show_trail = param_dict["show_trail"]
        if "trail_length" in param_dict:
            self.trail_length = param_dict["trail_length"]
        if "lost_threshold" in param_dict:
            self.lost_threshold = param_dict["lost_threshold"]

        # --- 接收 Tab 5 传来的 YOLO 引擎超参数 ---
        if "detection_con" in param_dict:
            self.detection_con = param_dict["detection_con"]
        if "iou" in param_dict:
            self.iou = param_dict["iou"]
        if "obj_list" in param_dict:
            self.obj_list = param_dict["obj_list"]

        if self.tracker is not None:
            self.tracker.update_params(
                conf=getattr(self, 'detection_con', None),
                iou=getattr(self, 'iou', None),
                obj_list=getattr(self, 'obj_list', None)
            )

    def process(self, frame):
        if self.tracker is None:
            # 🎯 修复点：接收 UI 传来的模型档名，并组装成绝对路径
            model_name = getattr(self, 'model_name', 'yolo26s.pt')
            model_path = os.path.join(project_root, "models", model_name)

            # 实例化时将完整的 model_path 传给底层引擎！
            self.tracker = YOLOTracker(
                model_path=model_path,
                conf=getattr(self, 'detection_con', 0.25),
                iou=getattr(self, 'iou', 0.70),
                obj_list=getattr(self, 'obj_list', None)
            )

        myDetections = Detections()
        output_image_frame, list_boxes = self.tracker.track(frame)

        for item_bbox in list_boxes:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            myDetections.add((x1, y1, x2, y2), track_id)

        current_ids = []
        for xyxy, track_id in myDetections.detections:
            x1, y1, x2, y2 = xyxy
            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            current_ids.append(track_id)

            if track_id in self.object_trails:
                self.object_trails[track_id].append((center.x, center.y))
            else:
                self.object_trails[track_id] = [(center.x, center.y)]

            self.lost_frames_counter[track_id] = 0

        if self.show_trail:
            trail_colors = [(255, 0, 255)] * len(self.object_trails)
            draw_trail(output_image_frame, list(self.object_trails.values()), trail_colors,
                       trail_length=self.trail_length)

        remove_ids = []
        for track_id in self.object_trails:
            if track_id not in current_ids:
                self.lost_frames_counter[track_id] = self.lost_frames_counter.get(track_id, 0) + 1

                if self.lost_frames_counter[track_id] > self.lost_threshold:
                    remove_ids.append(track_id)

                if len(self.object_trails[track_id]) > 0:
                    self.object_trails[track_id].pop(0)

        for tid in remove_ids:
            self.object_trails.pop(tid)
            self.lost_frames_counter.pop(tid)

        status_text = f"正在追踪目标数: {len(current_ids)} | 轨迹渲染: {'启用 (ON)' if self.show_trail else '关闭 (OFF)'}"
        return output_image_frame, {"status": status_text}

    def close(self):
        if self.tracker is not None:
            self.tracker.close()


if __name__ == '__main__':
    print("====== 开始独立测试 YOLO Trajectory Task ======")
    test_img_path = os.path.join(project_root, "images/bus.jpg")
    img = cv2.imread(test_img_path)

    if img is not None:
        task = YoloTrajectoryTask()
        out_img, info = task.process(img)
        print(info)
        cv2.imshow("Test Task", out_img)
        cv2.waitKey(0)
        task.close()
    else:
        print("测试图片读取失败。")