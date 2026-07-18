# qthreads/tasks/yolo_trajectory_task.py
import os
import sys

# ====================================================================
# 👑 【专业级导入黑科技】动态将项目根目录挂载到系统环境变量
# 作用：保证这段代码既能被 main.py 正常调用，又能被单独按右键执行测试，绝不报 ImportError！
# ====================================================================
# 1. 获取当前档案所在的绝对路径 (即 tasks/ 目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 往上推两层，找到专案的根目录 (即 2026-PySide6-GUI/)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# 3. 如果根目录不在 Python 的视野里，就强制加入最高优先级 (index 0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 挂载完成后，我们就可以放心大胆地使用 "绝对路径" 来 Import 任何包了！
import cv2
import numpy as np
from qthreads.tasks.base_task import BaseCVTask

# 🎯 注意这里的档名大小写：必须完全贴合 utils/yoloTracker.py
from utils.yoloTracker import YOLOTracker


# ====================================================================
# 🎓 教学保留区：完全沿用原本的辅助类别与函数，保持教学一贯性
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
            trail_points[i].pop(0)  # Remove the oldest point from the trail


# ====================================================================
# 🚀 核心任务外挂：衔接 UI 与底层 YOLOTracker 的桥梁
# ====================================================================
class YoloTrajectoryTask(BaseCVTask):
    """
    YOLO 目标追踪与轨迹绘制任务 (完美相容 BaseWorker 多线程底座)
    """

    def __init__(self):
        super().__init__()
        self.tracker = None

        # Dictionary to store the trail points of each object
        self.object_trails = {}
        # 记录每个目标未侦测到的连续帧数
        self.lost_frames_counter = {}

        # 💡 【为下一步 GUI 预留的超参数】
        # 这些参数之后可以让学生在 UI 界面上用滑动条 (Slider) 实时动态调整！
        self.trail_length = 50  # 轨迹尾巴的保留长度
        self.lost_threshold = 20  # 目标丢失几帧后彻底删除其轨迹

    def process(self, frame):
        """
        核心视频帧深度检测管道 (取代了原本的 while cap.isOpened() 循环内部逻辑)
        """
        # 🎯 延迟加载 (Lazy Initialization)：确保读取到 UI 传来的最新参数
        if self.tracker is None:
            self.tracker = YOLOTracker(conf=getattr(self, 'detection_con', 0.25))

        # 利用 YOLO Tracker 来读取 frame 中的目标框
        myDetections = Detections()
        output_image_frame, list_boxes = self.tracker.track(frame)

        for item_bbox in list_boxes:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            myDetections.add((x1, y1, x2, y2), track_id)

        # Add the current object's position to the trail
        current_ids = []
        for xyxy, track_id in myDetections.detections:
            x1, y1, x2, y2 = xyxy
            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            current_ids.append(track_id)  # 收集目前有侦测到的 ID

            if track_id in self.object_trails:
                self.object_trails[track_id].append((center.x, center.y))
            else:
                self.object_trails[track_id] = [(center.x, center.y)]

            # 重置此 ID 的失踪帧数
            self.lost_frames_counter[track_id] = 0

        # Draw the trail for each object (颜色使用洋红色)
        trail_colors = [(255, 0, 255)] * len(self.object_trails)

        # 💡 使用 self.trail_length 动态决定长度，方便 UI 控制
        draw_trail(output_image_frame, list(self.object_trails.values()), trail_colors, trail_length=self.trail_length)

        # 修改 trail 删除逻辑：加上等待时间
        remove_ids = []
        for track_id in self.object_trails:
            if track_id not in current_ids:
                # 记录未侦测帧数，若超过阈值才准备删除 (使用 self.lost_threshold 支持 UI 动态控制)
                self.lost_frames_counter[track_id] = self.lost_frames_counter.get(track_id, 0) + 1

                if self.lost_frames_counter[track_id] > self.lost_threshold:
                    remove_ids.append(track_id)

                if len(self.object_trails[track_id]) > 0:
                    self.object_trails[track_id].pop(0)

        for tid in remove_ids:
            self.object_trails.pop(tid)
            self.lost_frames_counter.pop(tid)  # 清除对应的计数器

        # 整理状态数据，回传给 UI 的 Footer 状态栏显示
        status_text = f"正在追踪目标数: {len(current_ids)} | 缓存轨迹数: {len(self.object_trails)}"
        return output_image_frame, {"status": status_text}

    def close(self):
        """
        安全释放生命周期资源
        """
        if self.tracker is not None:
            self.tracker.close()