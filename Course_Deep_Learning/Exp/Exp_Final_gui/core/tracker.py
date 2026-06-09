import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATHS = [
    PROJECT_ROOT / "models" / "yolo26n.pt",
    PROJECT_ROOT / "models" / "yolo26s.pt",
    PROJECT_ROOT / "models" / "yolo11n.pt",
    PROJECT_ROOT / "models" / "yolo11s.pt",
]

OBJ_LIST = ['person', 'car', 'bus', 'truck']

COLORS = {
    'person': (255, 0, 0),
    'car': (0, 255, 0),
    'bus': (0, 0, 255),
    'truck': (255, 255, 0),
}


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def isInsidePolygon(pt, polygon):
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def drawAndFillPolygon(frame, polygon, color, alpha=0.3):
    overlay = frame.copy()
    pts = np.array([[p.x, p.y] for p in polygon], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)


def generate_trail_color(track_id):
    h = hashlib.md5(str(track_id).encode()).digest()
    return (int(h[0]) % 256, int(h[1]) % 256, int(h[2]) % 256)


class VehicleTracker:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or self._find_model()
        if self.model_path:
            self._load_model()

        self.trails = defaultdict(list)
        self.trail_colors = {}
        self.counted_ids = set()
        self.counts = {
            'car': {'in': 0, 'out': 0},
            'bus': {'in': 0, 'out': 0},
            'truck': {'in': 0, 'out': 0},
        }

        self.zone_polygon = []
        self.intrusion_log = []
        self.intruded_ids = set()
        self.intrusion_start = {}
        self.snapshot_count = 0

        self.count_line_y_ratio = 0.6
        self.conf_threshold = 0.25
        self.target_classes = ['car', 'bus', 'truck']

        self.show_boxes = True
        self.show_trails = True
        self.show_count_line = True
        self.show_zone = True
        self.snapshot_enabled = True

        self.fps_values = []
        self.fps = 0.0
        self.frame_count = 0
        self.frame_h = 0
        self.frame_w = 0

    def _find_model(self):
        for p in DEFAULT_MODEL_PATHS:
            if p.exists():
                return str(p)
        exp_final_models = PROJECT_ROOT.parent / "Exp_Final" / "models"
        if exp_final_models.exists():
            for f in sorted(exp_final_models.glob("*.pt")):
                return str(f)
        alt = list(PROJECT_ROOT.rglob("*.pt"))
        if alt:
            return str(alt[0])
        return None

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            print("[ERROR] ultralytics not installed")

    def set_zone_polygon(self, points):
        self.zone_polygon = [Point(p[0], p[1]) for p in points]

    def set_default_zone(self, frame_w, frame_h):
        self.zone_polygon = [
            Point(int(frame_w * 0.10), int(frame_h * 0.85)),
            Point(int(frame_w * 0.90), int(frame_h * 0.85)),
            Point(int(frame_w * 0.70), int(frame_h * 0.35)),
            Point(int(frame_w * 0.30), int(frame_h * 0.35)),
        ]

    def reset(self):
        self.trails.clear()
        self.trail_colors.clear()
        self.counted_ids.clear()
        self.intruded_ids.clear()
        self.intrusion_start.clear()
        self.intrusion_log.clear()
        self.snapshot_count = 0
        for k in self.counts:
            self.counts[k]['in'] = 0
            self.counts[k]['out'] = 0

    def _get_trail_color(self, track_id):
        if track_id not in self.trail_colors:
            self.trail_colors[track_id] = generate_trail_color(track_id)
        return self.trail_colors[track_id]

    def process(self, frame):
        if self.model is None:
            return frame, {'error': 'Model not loaded', 'fps': 0, 'frame': 0, 'counts': self.counts}

        self.frame_h, self.frame_w = frame.shape[:2]
        self.frame_count += 1

        t_start = cv2.getTickCount()

        results = self.model.track(frame, persist=True, conf=self.conf_threshold, verbose=False)

        t_end = cv2.getTickCount()
        dt = (t_end - t_start) / cv2.getTickFrequency()
        current_fps = 1.0 / dt if dt > 0 else 0
        self.fps_values.append(current_fps)
        if len(self.fps_values) > 30:
            self.fps_values.pop(0)
        self.fps = sum(self.fps_values) / len(self.fps_values)

        display = frame.copy()
        class_names = self.model.names if self.model else {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            cls_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
            xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

            for i in range(len(boxes)):
                cls_name = class_names.get(cls_ids[i], 'unknown')
                if cls_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]
                track_id = ids[i] if ids else -1

                if track_id > 0:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    self.trails[track_id].append(Point(cx, cy))
                    if len(self.trails[track_id]) > 50:
                        self.trails[track_id].pop(0)

                    count_line_y = int(self.frame_h * self.count_line_y_ratio)
                    if track_id not in self.counted_ids and len(self.trails[track_id]) >= 2:
                        prev_y = self.trails[track_id][-2].y
                        if prev_y < count_line_y <= cy:
                            self.counts[cls_name]['in'] += 1
                            self.counted_ids.add(track_id)
                        elif prev_y > count_line_y >= cy:
                            self.counts[cls_name]['out'] += 1
                            self.counted_ids.add(track_id)

                    if self.zone_polygon:
                        foot = Point(int(cx), int(y2))
                        if isInsidePolygon(foot, self.zone_polygon):
                            if track_id not in self.intruded_ids:
                                self.intruded_ids.add(track_id)
                                self.intrusion_start[track_id] = datetime.now()
                                if self.snapshot_enabled:
                                    self._capture_snapshot(frame, track_id, cls_name)
                                self.intrusion_log.append({
                                    'id': track_id,
                                    'class': cls_name,
                                    'time': datetime.now().strftime('%H:%M:%S'),
                                })

                    self._draw_box(display, x1, y1, x2, y2, cls_name, conf, track_id)

        if self.show_count_line:
            self._draw_count_line(display)
        if self.show_zone and self.zone_polygon:
            drawAndFillPolygon(display, self.zone_polygon, (0, 0, 255), 0.25)
        if self.show_trails:
            self._draw_trails(display)

        info = {
            'fps': self.fps,
            'frame': self.frame_count,
            'counts': dict(self.counts),
            'intrusions': len(self.intruded_ids),
        }

        return display, info

    def _draw_box(self, frame, x1, y1, x2, y2, cls_name, conf, track_id):
        color = COLORS.get(cls_name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} #{track_id}" if track_id > 0 else cls_name
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_count_line(self, frame):
        y = int(self.frame_h * self.count_line_y_ratio)
        cv2.line(frame, (0, y), (self.frame_w, y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNT LINE", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "OUT", (self.frame_w - 80, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "IN", (self.frame_w - 80, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_trails(self, frame):
        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue
            color = self._get_trail_color(tid)
            for i in range(1, len(trail)):
                cv2.line(frame,
                         (int(trail[i - 1].x), int(trail[i - 1].y)),
                         (int(trail[i].x), int(trail[i].y)),
                         color, 2)

    def _capture_snapshot(self, frame, track_id, cls_name):
        snaps_dir = PROJECT_ROOT / "snapshots"
        snaps_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = snaps_dir / f"{track_id}_{cls_name}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        self.snapshot_count += 1

    def get_count_summary(self):
        lines = []
        for cls_name in ['car', 'bus', 'truck']:
            c = self.counts[cls_name]
            lines.append(f"{cls_name}: IN {c['in']} | OUT {c['out']} | TOTAL {c['in'] + c['out']}")
        return "\n".join(lines)
