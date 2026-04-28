# yolo_tracker_base.py (修改后)

"""
YOLO Tracker 基础类
支持：追踪、画框、画轨迹、方向向量、角度计算、变道判断、CSV数据收集
用于 T-2、T-3、T-4 共用
"""

import cv2
import math
import torch
from ultralytics import YOLO

import os
import sys

# ========== 项目路径自动检测 ==========
def get_project_root():
    """
    获取项目根目录
    当前文件(yolo_tracker_base.py)位于项目根目录下
    """
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()

# 定义各个子文件夹路径
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'videos')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'videos_output')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认模型路径（优先使用存在的模型）
DEFAULT_MODEL_PATHS = [
    os.path.join(MODELS_DIR, 'yolo26n.pt'),
    os.path.join(MODELS_DIR, 'yolov8n.pt'),
    os.path.join(MODELS_DIR, 'yolov9t.pt'),
]

# ========== 全局配置 ==========
OBJ_LIST = ['person', 'car', 'bus', 'truck']
COLORS = {
    'person': (0, 255, 0),    # 绿色
    'car': (255, 0, 0),       # 蓝色
    'bus': (0, 0, 255),       # 红色
    'truck': (0, 255, 255)    # 黄色
}
TRAIL_COLOR = (255, 0, 255)   # 轨迹颜色：品红色
# ==============================


class YOLOTracker:
    """YOLO目标追踪基础类"""
    
    def __init__(self, model_path=None, device=None):
        """
        初始化追踪器
        Args:
            model_path: YOLO模型路径
            device: 设备，None自动选择
        """
        # 自动选择模型路径
        if model_path is None:
            model_path = self._find_model()
        
        if model_path is None:
            raise FileNotFoundError(f"未找到模型文件！请在 {MODELS_DIR} 目录下放置模型文件")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        self.model = YOLO(model_path)
        self.img_size = 640
        self.conf = 0.25
        self.iou = 0.70
        
        # T-3/T-4 轨迹记录
        self.trails = {}           # {track_id: [(x,y), ...]}
        self.lost_counter = {}     # {track_id: 消失帧数}
        self.trail_length = 50     # 轨迹最大长度
        self.lost_threshold = 20   # 消失多少帧后删除轨迹
        
        # T-4 方向分析数据
        self.angle_history = {}    # {track_id: [angle1, angle2, ...]}
        self.lane_change_status = {}  # {track_id: True/False}
        self.change_frames = {}    # {track_id: 持续变道帧数}
        
        # T-4 CSV数据存储
        self.analysis_records = []  # 存储每一帧每一辆车的分析数据
        self.frame_id = 0           # 当前帧编号
        
        # T-4 变道判断参数
        self.change_threshold = 25   # 累计角度变化超过此值触发变道
        self.change_window = 15      # 观察窗口帧数
        self.change_min_frames = 10  # 最少持续帧数才判定
    
    def _find_model(self):
        """自动查找可用的模型文件"""
        for model_path in DEFAULT_MODEL_PATHS:
            if os.path.exists(model_path):
                return model_path
        
        # 如果默认路径都不存在，尝试搜索models目录下的任何.pt文件
        if os.path.exists(MODELS_DIR):
            pt_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
            if pt_files:
                return os.path.join(MODELS_DIR, pt_files[0])
        
        return None
    
    def get_video_path(self, video_name):
        """获取视频文件的完整路径"""
        return os.path.join(VIDEOS_DIR, video_name)
    
    def get_output_path(self, output_name):
        """获取输出文件的完整路径"""
        return os.path.join(OUTPUT_DIR, output_name)
    
    def reset(self):
        """重置所有轨迹和分析数据（处理新视频时调用）"""
        self.trails.clear()
        self.lost_counter.clear()
        self.angle_history.clear()
        self.lane_change_status.clear()
        self.change_frames.clear()
        self.analysis_records.clear()
        self.frame_id = 0
    
    def track(self, frame, draw_trail=False, analyze_lane_change=False):
        """
        对一帧图像进行追踪
        Args:
            frame: 输入图像
            draw_trail: 是否画轨迹线（T-3/T-4用）
            analyze_lane_change: 是否进行变道分析（T-4用）
        Returns:
            annotated_frame: 标注后的图像
            pred_boxes: 检测结果列表 [(x1,y1,x2,y2,lbl,track_id), ...]
        """
        self.frame_id += 1
        
        # YOLO追踪
        results = self.model.track(
            frame, persist=True, device=self.device,
            imgsz=self.img_size, conf=self.conf, iou=self.iou
        )
        
        pred_boxes = []
        current_ids = []
        
        # 解析追踪结果
        if results[0].boxes and results[0].boxes.id is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().item())
                lbl = self.model.names[class_id]
                
                # 只追踪OBJ_LIST中的类别
                if lbl not in OBJ_LIST:
                    continue
                
                xyxy = box.xyxy.cpu()[0].numpy()
                x1, y1, x2, y2 = xyxy
                track_id = int(box.id.cpu().item())
                current_ids.append(track_id)
                
                # 计算中心点（T-3轨迹用）
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 更新轨迹
                if draw_trail or analyze_lane_change:
                    if track_id in self.trails:
                        self.trails[track_id].append((center_x, center_y))
                    else:
                        self.trails[track_id] = [(center_x, center_y)]
                    self.lost_counter[track_id] = 0
                
                # T-4: 分析方向向量和角度
                if analyze_lane_change and track_id in self.trails and len(self.trails[track_id]) >= 2:
                    trail = self.trails[track_id]
                    p1 = trail[-2]
                    p2 = trail[-1]
                    
                    # 计算方向向量
                    vx = p2[0] - p1[0]
                    vy = p2[1] - p1[1]
                    
                    # 计算角度
                    angle = None
                    if vx != 0 or vy != 0:
                        angle = math.degrees(math.atan2(vy, vx))
                    
                    # 记录角度历史
                    if track_id not in self.angle_history:
                        self.angle_history[track_id] = []
                    if angle is not None:
                        self.angle_history[track_id].append(angle)
                    
                    # 变道判断
                    self._check_lane_change(track_id)
                    
                    # 记录CSV数据
                    self.analysis_records.append({
                        'frame_id': self.frame_id,
                        'track_id': track_id,
                        'center_x': center_x,
                        'center_y': center_y,
                        'vx': vx,
                        'vy': vy,
                        'angle': angle if angle is not None else 0
                    })
                
                pred_boxes.append((x1, y1, x2, y2, lbl, track_id))
        
        # 处理消失的目标（T-3/T-4）
        if draw_trail or analyze_lane_change:
            remove_ids = []
            for tid in list(self.trails.keys()):
                if tid not in current_ids:
                    self.lost_counter[tid] = self.lost_counter.get(tid, 0) + 1
                    if self.lost_counter[tid] > self.lost_threshold:
                        remove_ids.append(tid)
            
            for tid in remove_ids:
                if tid in self.trails:
                    self.trails.pop(tid)
                if tid in self.lost_counter:
                    self.lost_counter.pop(tid)
                if tid in self.angle_history:
                    self.angle_history.pop(tid)
                if tid in self.lane_change_status:
                    self.lane_change_status.pop(tid)
                if tid in self.change_frames:
                    self.change_frames.pop(tid)
        
        # 绘制标注
        annotated_frame = self._draw_boxes(frame.copy(), pred_boxes)
        
        # 画轨迹线（T-3/T-4）
        if draw_trail:
            annotated_frame = self._draw_trails(annotated_frame)
        
        # 显示变道状态（T-4）
        if analyze_lane_change:
            annotated_frame = self._draw_lane_change_status(annotated_frame, pred_boxes)
        
        return annotated_frame, pred_boxes
    
    def _draw_boxes(self, im, boxes):
        """绘制边界框和标签"""
        for x1, y1, x2, y2, lbl, track_id in boxes:
            color = COLORS.get(lbl, (128, 128, 128))
            # 如果是变道中，用红色框
            if self.lane_change_status.get(track_id, False):
                color = (0, 0, 255)  # 红色
            
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 标签文本
            text = f"{lbl}({track_id})"
            if self.lane_change_status.get(track_id, False):
                text = f"{lbl}({track_id}) LANE CHANGING"
            
            cv2.putText(im, text, (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return im
    
    def _draw_trails(self, im):
        """绘制轨迹线"""
        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                cv2.line(im, pt1, pt2, TRAIL_COLOR, 2)
            # 限制轨迹长度
            if len(trail) > self.trail_length:
                self.trails[tid] = trail[-self.trail_length:]
        return im
    
    def _draw_lane_change_status(self, im, boxes):
        """显示变道状态"""
        for x1, y1, x2, y2, lbl, track_id in boxes:
            if self.lane_change_status.get(track_id, False):
                # 在框上方显示变道文字
                text = "LANE CHANGING"
                text_x = int(x1)
                text_y = int(y1) - 25
                cv2.putText(im, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return im
    
    def _check_lane_change(self, track_id):
        """
        基于角度历史判断是否变道
        判断逻辑：在最近change_window帧内，角度累计变化超过change_threshold度
        """
        if track_id not in self.angle_history:
            return
        
        angles = self.angle_history[track_id]
        if len(angles) < self.change_window:
            return
        
        # 取最近change_window帧的角度
        recent = angles[-self.change_window:]
        
        # 计算累计角度变化
        total_change = 0
        for i in range(1, len(recent)):
            diff = recent[i] - recent[i-1]
            total_change += abs(diff)
        
        # 判断是否变道
        if total_change > self.change_threshold:
            self.change_frames[track_id] = self.change_frames.get(track_id, 0) + 1
            if self.change_frames[track_id] >= self.change_min_frames:
                self.lane_change_status[track_id] = True
        else:
            self.change_frames[track_id] = 0
            self.lane_change_status[track_id] = False
    
    def export_csv(self, csv_path="analysis_records.csv"):
        """导出分析数据到CSV文件"""
        import csv
        
        if not self.analysis_records:
            print("没有分析数据可导出")
            return
        
        # 如果路径是相对路径，则保存到输出目录
        if not os.path.isabs(csv_path) and not csv_path.startswith('.'):
            csv_path = os.path.join(OUTPUT_DIR, csv_path)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['frame_id', 'track_id', 'center_x', 'center_y', 'vx', 'vy', 'angle']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.analysis_records)
        
        print(f"分析数据已导出到: {csv_path}")
    
    def get_angle_history(self, track_id):
        """获取指定track_id的角度历史"""
        return self.angle_history.get(track_id, [])
    
    def get_lane_change_status(self, track_id):
        """获取指定track_id的变道状态"""
        return self.lane_change_status.get(track_id, False)