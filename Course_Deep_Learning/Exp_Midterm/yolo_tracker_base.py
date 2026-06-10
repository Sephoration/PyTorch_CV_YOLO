# yolo_tracker_base.py
"""
YOLO Tracker 基础类
Part A: 方向向量、角度计算、角度序列、CSV导出
Part B: 监控区间、三窗口判断、横向位移确认
Part C: 净角度变化检查、指数平滑轨迹、异常角度过滤
"""

import cv2
import math
import torch
import numpy as np
import os
from ultralytics import YOLO


# ========== 项目路径 ==========
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'videos')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'videos_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MODEL_PATHS = [
    os.path.join(MODELS_DIR, 'yolo26s.pt'),
    os.path.join(MODELS_DIR, 'yolo26n.pt'),
    os.path.join(MODELS_DIR, 'yolov8n.pt'),
    os.path.join(MODELS_DIR, 'yolov9t.pt'),
]

# ========== 全局配置 ==========
OBJ_LIST = ['person', 'car', 'bus', 'truck']
COLORS = {
    'person': (0, 255, 0),
    'car': (255, 0, 0),
    'bus': (0, 0, 255),
    'truck': (0, 255, 255)
}
TRAIL_COLOR = (255, 0, 255)


class YOLOTracker:

    def __init__(self, model_path=None, device=None):
        if model_path is None:
            model_path = self._find_model()
        if model_path is None:
            raise FileNotFoundError(f"未找到模型文件！请在 {MODELS_DIR} 目录下放置模型文件")

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        self.model = YOLO(model_path)
        self.img_size = 640
        self.conf = 0.35
        self.iou = 0.70

        # ---- 轨迹记录 ----
        self.trails = {}
        self.lost_counter = {}
        self.trail_length = 80
        self.lost_threshold = 20

        # ---- Part A: 方向分析 ----
        self.angle_history = {}
        self.angle_diffs = {}
        self.analysis_records = []
        self.frame_id = 0

        # ---- Part B: 监控区间 ----
        self.valid_y_min = 120
        self.valid_y_max = 600
        self.angle_step = 5
        self.perspective_enabled = True  # 透视补偿：远距离自动放宽阈值

        # ---- Part B: 三窗口角度判断 ----
        self.short_window_size = 10
        self.short_acc_threshold = 20
        self.short_consistent_ratio = 0.50
        self.long_window_size = 35
        self.long_acc_threshold = 35
        self.long_consistent_ratio = 0.45
        self.long2_window_size = 60
        self.long2_acc_threshold = 20
        self.long2_consistent_ratio = 0.40

        # ---- Part B: 横向位移确认 ----
        self.trajectory_window_size = 25
        self.min_lateral_shift = 15
        self.min_lateral_ratio = 0.15
        self.min_x_consistent_ratio = 0.55

        # ---- Part C: 净角度检查（过滤振荡）----
        self.short_net_threshold = 15
        self.long_net_threshold = 25
        self.long2_net_threshold = 20

        # ---- Part C: 指数平滑 ----
        self.smooth_alpha = 0.3
        self.trails_smooth = {}

        # ---- Part C: 异常角度过滤 ----
        self.abnormal_angle_diff_threshold = 8.0

        # ---- 速度 / 轴方向 ----
        self.min_speed = 2.0
        self.lateral_axis = 'x'

        # ---- 结果存储 ----
        self.lane_change_results = {}

    # ==================== 模型查找 ====================
    def _find_model(self):
        for model_path in DEFAULT_MODEL_PATHS:
            if os.path.exists(model_path):
                return model_path
        if os.path.exists(MODELS_DIR):
            pt_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
            if pt_files:
                return os.path.join(MODELS_DIR, pt_files[0])
        return None

    def reset(self):
        self.trails.clear()
        self.trails_smooth.clear()
        self.lost_counter.clear()
        self.angle_history.clear()
        self.angle_diffs.clear()
        self.analysis_records.clear()
        self.lane_change_results.clear()
        self.frame_id = 0

    # ==================== 监控区间 ====================
    def in_valid_y_zone(self, y):
        return self.valid_y_min <= y <= self.valid_y_max

    def get_perspective_scale(self, cy):
        """透视补偿：远处(cy小)→返回<1放宽阈值，近处(cy大)→返回1.0严格"""
        if not self.perspective_enabled:
            return 1.0
        ratio = (cy - self.valid_y_min) / max(self.valid_y_max - self.valid_y_min, 1)
        ratio = max(0.0, min(1.0, ratio))
        return 0.7 + 0.3 * ratio  # 远处放宽到70%，近处100%

    # ==================== Part C: 指数平滑 ====================
    def exponential_smooth_point(self, prev_smooth_point, current_point, alpha):
        if prev_smooth_point is None:
            return current_point
        smooth_x = alpha * current_point[0] + (1 - alpha) * prev_smooth_point[0]
        smooth_y = alpha * current_point[1] + (1 - alpha) * prev_smooth_point[1]
        return (smooth_x, smooth_y)

    def update_smooth_trail(self, track_id, center):
        if track_id not in self.trails_smooth:
            self.trails_smooth[track_id] = [center]
            return center
        prev = self.trails_smooth[track_id][-1]
        sc = self.exponential_smooth_point(prev, center, self.smooth_alpha)
        self.trails_smooth[track_id].append(sc)
        if len(self.trails_smooth[track_id]) > self.trail_length:
            self.trails_smooth[track_id] = self.trails_smooth[track_id][-self.trail_length:]
        return sc

    # ==================== Part B: 角度计算 ====================
    def get_angle_diff_deg(self, angle1, angle2):
        diff = angle2 - angle1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff

    def judge_lane_change_by_window(self, angle_diff_list, window_size, acc_threshold,
                                     consistent_ratio, scale=1.0):
        if len(angle_diff_list) < window_size:
            return False, None
        diff_window = angle_diff_list[-window_size:]
        positive_diffs = [d for d in diff_window if d > 0]
        negative_diffs = [d for d in diff_window if d < 0]
        positive_sum = sum(positive_diffs)
        negative_sum = sum(abs(d) for d in negative_diffs)

        scaled_th = acc_threshold * scale  # 远处scale<1→阈值降低→更易触发
        if positive_sum >= scaled_th and len(positive_diffs) / len(diff_window) >= consistent_ratio:
            return True, 'right'
        if negative_sum >= scaled_th and len(negative_diffs) / len(diff_window) >= consistent_ratio:
            return True, 'left'
        return False, None

    def judge_lane_change_by_triple_window(self, angle_diff_list, scale=1.0):
        short_ok, _ = self.judge_lane_change_by_window(
            angle_diff_list, self.short_window_size,
            self.short_acc_threshold, self.short_consistent_ratio, scale)
        if short_ok:
            return True, 'short'
        long_ok, _ = self.judge_lane_change_by_window(
            angle_diff_list, self.long_window_size,
            self.long_acc_threshold, self.long_consistent_ratio, scale)
        if long_ok:
            return True, 'long'
        long2_ok, _ = self.judge_lane_change_by_window(
            angle_diff_list, self.long2_window_size,
            self.long2_acc_threshold, self.long2_consistent_ratio, scale)
        if long2_ok:
            return True, 'long2'
        return False, None

    # ==================== Part C: 净角度检查 ====================
    def judge_net_angle_change(self, angle_diff_list, window_size, net_threshold, scale=1.0):
        if len(angle_diff_list) < window_size:
            return False
        diff_window = angle_diff_list[-window_size:]
        positive_sum = sum(d for d in diff_window if d > 0)
        negative_sum = sum(abs(d) for d in diff_window if d < 0)
        return abs(positive_sum - negative_sum) >= net_threshold * scale

    def judge_net_angle_change_by_triple_window(self, angle_diff_list, trigger_type, scale=1.0):
        if trigger_type == 'short':
            return self.judge_net_angle_change(angle_diff_list, self.short_window_size,
                                               self.short_net_threshold, scale)
        if trigger_type == 'long':
            return self.judge_net_angle_change(angle_diff_list, self.long_window_size,
                                               self.long_net_threshold, scale)
        if trigger_type == 'long2':
            return self.judge_net_angle_change(angle_diff_list, self.long2_window_size,
                                               self.long2_net_threshold, scale)
        return False

    # ==================== Part B: 横向位移确认 ====================
    def judge_lateral_shift_by_trail(self, trail, scale=1.0):
        if len(trail) < self.trajectory_window_size:
            return False
        recent = trail[-self.trajectory_window_size:]
        sx, sy = recent[0]
        ex, ey = recent[-1]
        dx, dy = ex - sx, ey - sy

        if self.lateral_axis == 'y':
            lateral_disp, forward_disp = dy, dx
            axis_diffs = [recent[i][1] - recent[i-1][1] for i in range(1, len(recent))]
        else:
            lateral_disp, forward_disp = dx, dy
            axis_diffs = [recent[i][0] - recent[i-1][0] for i in range(1, len(recent))]

        # 透视补偿：远处scale<1→阈值降低→更易通过
        scaled_pixel = self.min_lateral_shift * scale
        scaled_ratio = self.min_lateral_ratio * scale

        if abs(lateral_disp) < scaled_pixel:
            return False
        if abs(lateral_disp) / (abs(forward_disp) + 1e-6) < scaled_ratio:
            return False
        if len(axis_diffs) == 0:
            return False

        pc = sum(1 for d in axis_diffs if d > 0)
        nc = sum(1 for d in axis_diffs if d < 0)
        cr = pc / len(axis_diffs) if lateral_disp > 0 else nc / len(axis_diffs) if lateral_disp < 0 else 0
        return cr >= self.min_x_consistent_ratio

    # ==================== 辅助 ====================
    def get_motion_vector(self, p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    def get_vector_angle_deg(self, vec):
        vx, vy = vec
        if vx == 0 and vy == 0:
            return None
        return math.degrees(math.atan2(vy, vx))

    # ==================== 核心追踪 ====================
    def track(self, frame, draw_trail=False, analyze_lane_change=False):
        self.frame_id += 1

        results = self.model.track(
            frame, persist=True, device=self.device,
            imgsz=self.img_size, conf=self.conf, iou=self.iou)

        pred_boxes = []
        current_ids = []

        if results[0].boxes and results[0].boxes.id is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().item())
                lbl = self.model.names[class_id]
                if lbl not in OBJ_LIST:
                    continue

                xyxy = box.xyxy.cpu()[0].numpy()
                x1, y1, x2, y2 = xyxy
                track_id = int(box.id.cpu().item())
                current_ids.append(track_id)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                center = (cx, cy)

                if draw_trail or analyze_lane_change:
                    if track_id in self.trails:
                        self.trails[track_id].append(center)
                    else:
                        self.trails[track_id] = [center]
                    self.lost_counter[track_id] = 0
                    if len(self.trails[track_id]) > self.trail_length:
                        self.trails[track_id] = self.trails[track_id][-self.trail_length:]
                    smooth_center = self.update_smooth_trail(track_id, center)
                else:
                    smooth_center = center

                if analyze_lane_change:
                    if not self.in_valid_y_zone(cy):
                        pred_boxes.append((x1, y1, x2, y2, lbl, track_id))
                        continue

                    trail_use = self.trails_smooth.get(track_id, self.trails.get(track_id, []))

                    angle = None
                    angle_diff = None
                    angle_ok = False
                    net_angle_ok = False
                    lateral_ok = False
                    trigger_type = None

                    if len(trail_use) >= self.angle_step + 1:
                        p1 = trail_use[-1 - self.angle_step]
                        p2 = trail_use[-1]
                        vx, vy = self.get_motion_vector(p1, p2)
                        motion_mag = math.sqrt(vx**2 + vy**2)

                        if motion_mag >= self.min_speed:
                            if vx != 0 or vy != 0:
                                angle = self.get_vector_angle_deg((vx, vy))

                            if track_id not in self.angle_history:
                                self.angle_history[track_id] = []
                            if angle is not None:
                                self.angle_history[track_id].append(angle)

                            if track_id not in self.angle_diffs:
                                self.angle_diffs[track_id] = []

                            if len(self.angle_history[track_id]) > 1:
                                ad = self.get_angle_diff_deg(
                                    self.angle_history[track_id][-2], angle)
                                if abs(ad) <= self.abnormal_angle_diff_threshold:
                                    self.angle_diffs[track_id].append(ad)
                                    angle_diff = ad

                            # 透视补偿
                            p_scale = self.get_perspective_scale(cy)

                            angle_ok, trigger_type = self.judge_lane_change_by_triple_window(
                                self.angle_diffs.get(track_id, []), p_scale)

                            if angle_ok and trigger_type:
                                net_angle_ok = self.judge_net_angle_change_by_triple_window(
                                    self.angle_diffs.get(track_id, []), trigger_type, p_scale)

                            lateral_ok = self.judge_lateral_shift_by_trail(trail_use, p_scale)

                            # 最终判定：角度 + 净角度 + 横向位移
                            all_checks_ok = angle_ok and net_angle_ok and lateral_ok
                            if all_checks_ok:
                                if track_id not in self.lane_change_results:
                                    self.lane_change_results[track_id] = True

                        self.analysis_records.append({
                            'frame_id': self.frame_id,
                            'track_id': track_id,
                            'center_x': cx, 'center_y': cy,
                            'smooth_x': smooth_center[0], 'smooth_y': smooth_center[1],
                            'vx': vx, 'vy': vy,
                            'angle': angle if angle is not None else 0,
                            'angle_diff': angle_diff if angle_diff is not None else 0,
                            'angle_ok': angle_ok,
                            'net_angle_ok': net_angle_ok,
                            'lateral_ok': lateral_ok
                        })

                pred_boxes.append((x1, y1, x2, y2, lbl, track_id))

        # ---- 清理消失的目标 ----
        if draw_trail or analyze_lane_change:
            remove_ids = []
            for tid in list(self.trails.keys()):
                if tid not in current_ids:
                    self.lost_counter[tid] = self.lost_counter.get(tid, 0) + 1
                    if self.lost_counter[tid] > self.lost_threshold:
                        remove_ids.append(tid)
            for tid in remove_ids:
                for d in [self.trails, self.trails_smooth, self.lost_counter,
                          self.angle_history, self.angle_diffs, self.lane_change_results]:
                    d.pop(tid, None)

        # ---- 绘制 ----
        annotated = self._draw_boxes(frame.copy(), pred_boxes)
        if draw_trail:
            annotated = self._draw_trails(annotated)
        if analyze_lane_change:
            annotated = self._draw_lane_change_status(annotated, pred_boxes)

        return annotated, pred_boxes

    def _draw_boxes(self, im, boxes):
        for x1, y1, x2, y2, lbl, track_id in boxes:
            color = (0, 0, 255) if self.lane_change_results.get(track_id, False) else COLORS.get(lbl, (128, 128, 128))
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            tag = f"{lbl}({track_id}) LANE CHANGE" if self.lane_change_results.get(track_id, False) else f"{lbl}({track_id})"
            cv2.putText(im, tag, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return im

    def _draw_trails(self, im):
        for tid, trail in self.trails_smooth.items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                cv2.line(im, (int(trail[i-1][0]), int(trail[i-1][1])),
                         (int(trail[i][0]), int(trail[i][1])), TRAIL_COLOR, 2)
        return im

    def _draw_lane_change_status(self, im, boxes):
        for x1, y1, x2, y2, lbl, track_id in boxes:
            if self.lane_change_results.get(track_id, False):
                cv2.putText(im, "LANE CHANGE", (int(x1), int(y1) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return im

    # ==================== CSV导出 ====================
    def export_csv(self, csv_path="analysis_records.csv"):
        import csv
        if not self.analysis_records:
            print("没有分析数据可导出")
            return
        if not os.path.isabs(csv_path) and not csv_path.startswith('.'):
            csv_path = os.path.join(OUTPUT_DIR, csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ['frame_id', 'track_id', 'center_x', 'center_y',
                      'smooth_x', 'smooth_y', 'vx', 'vy', 'angle',
                      'angle_diff', 'angle_ok', 'net_angle_ok',
                      'lateral_ok']
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.analysis_records)
        print(f"分析数据已导出到: {csv_path}")
        print(f"共导出 {len(self.analysis_records)} 条记录")

    def get_angle_history(self, track_id):
        return self.angle_history.get(track_id, [])

    def get_angle_diffs(self, track_id):
        return self.angle_diffs.get(track_id, [])

    def get_lane_change_result(self, track_id):
        return self.lane_change_results.get(track_id, False)
