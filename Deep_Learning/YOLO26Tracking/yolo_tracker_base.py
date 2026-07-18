# yolo_tracker_base.py (完整版 - 支持Part A/B/C全部功能)

"""
YOLO Tracker 基础类
支持：追踪、画框、画轨迹、方向向量、角度计算、变道判断、CSV数据收集
完整实现 PPT Part A、Part B、Part C 的所有功能：
- Part A: 方向向量、角度计算、角度序列、CSV导出、折线图
- Part B: 监控区间、ANGLE_STEP、角度差序列、双窗口判断、横向位移确认
- Part C: 净角度变化检查、指数平滑轨迹、异常角度过滤
"""

import cv2
import math
import torch
from ultralytics import YOLO

import os

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
    """YOLO目标追踪基础类 - 完整实现Part A/B/C功能"""
    
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
        
        # ========== T-3/T-4 基础轨迹记录 ==========
        self.trails = {}           # {track_id: [(x,y), ...]} 原始轨迹
        self.lost_counter = {}     # {track_id: 消失帧数}
        self.trail_length = 50     # 轨迹最大长度
        self.lost_threshold = 20   # 消失多少帧后删除轨迹
        
        # ========== Part A: 方向分析数据 ==========
        self.angle_history = {}    # {track_id: [angle1, angle2, ...]}
        self.lane_change_status = {}  # {track_id: "lane_change" 或 None}
        self.lane_change_locked = {}  # {track_id: True/False} 锁定后不再改变
        
        # ========== Part A: CSV数据存储 ==========
        self.analysis_records = []  # 存储每一帧每一辆车的分析数据
        self.frame_id = 0           # 当前帧编号
        
        # ========== Part B: 监控区间参数 ==========
        self.valid_y_min = 350      # 有效Y轴范围最小值（上边界）
        self.valid_y_max = 850      # 有效Y轴范围最大值（下边界）
        
        # ========== Part B: ANGLE_STEP 参数 ==========
        self.angle_step = 5         # 间隔轨迹点数（避免相邻帧抖动）
        
        # ========== Part B: 角度差序列 ==========
        self.angle_diffs = {}       # {track_id: [diff1, diff2, ...]}
        
        # ========== Part B: 双窗口角度变化参数 ==========
        # 短窗口（快速变道）
        self.short_window_size = 8
        self.short_acc_threshold = 35      # 累积角度阈值
        self.short_consistent_ratio = 0.60 # 方向一致性比例
        
        # 长窗口（慢速变道）
        self.long_window_size = 30
        self.long_acc_threshold = 55
        self.long_consistent_ratio = 0.50
        
        # ========== Part B: 横向位移确认参数 ==========
        self.trajectory_window_size = 20   # 最近20个轨迹点
        self.min_lateral_shift = 35        # 横向总位移至少35像素
        self.min_lateral_ratio = 0.35      # 横向/纵向比例至少0.35
        self.min_x_consistent_ratio = 0.65 # 横向方向一致性至少65%
        
        # ========== Part C: 净角度变化检查参数 ==========
        self.short_net_threshold = 20      # 短窗口净角度门槛
        self.long_net_threshold = 30       # 长窗口净角度门槛
        
        # ========== Part C: 指数平滑轨迹参数 ==========
        self.smooth_alpha = 0.3            # 平滑系数（0.2~0.4之间效果较好）
        self.trails_smooth = {}            # {track_id: [(x,y), ...]} 平滑轨迹
        
        # ========== Part C: 异常角度过滤参数 ==========
        self.abnormal_angle_diff_threshold = 6.0  # 单步角度差超过6度视为异常
        
        # 变道判断结果存储
        self.lane_change_results = {}      # {track_id: True/False}
    
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
        self.trails_smooth.clear()
        self.lost_counter.clear()
        self.angle_history.clear()
        self.angle_diffs.clear()
        self.lane_change_status.clear()
        self.lane_change_locked.clear()
        self.lane_change_results.clear()
        self.analysis_records.clear()
        self.frame_id = 0
    
    # ==================== Part B: 监控区间函数 ====================
    def in_valid_y_zone(self, y):
        """判断车辆中心点Y坐标是否在有效监控区间内"""
        return self.valid_y_min <= y <= self.valid_y_max
    
    def draw_valid_y_zone(self, frame, color=(0, 255, 255), thickness=2):
        """在画面上画出有效侦测区的上下边界线（用于调试）"""
        h, w = frame.shape[:2]
        # 上边界线
        cv2.line(frame, (0, int(self.valid_y_min)), (w, int(self.valid_y_min)), color, thickness)
        # 下边界线
        cv2.line(frame, (0, int(self.valid_y_max)), (w, int(self.valid_y_max)), color, thickness)
        # 显示文字说明
        cv2.putText(frame, f"VALID_Y_MIN = {self.valid_y_min}", (20, int(self.valid_y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"VALID_Y_MAX = {self.valid_y_max}", (20, int(self.valid_y_max) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
    
    # ==================== Part C: 指数平滑函数 ====================
    def exponential_smooth_point(self, prev_smooth_point, current_point, alpha):
        """
        对当前轨迹点做指数平滑
        参数：
            prev_smooth_point: 上一笔平滑后的点 (x, y)
            current_point: 当前原始点 (x, y)
            alpha: 平滑系数，范围 0~1
        返回：
            当前平滑后的点 (x, y)
        """
        if prev_smooth_point is None:
            return current_point
        
        smooth_x = alpha * current_point[0] + (1 - alpha) * prev_smooth_point[0]
        smooth_y = alpha * current_point[1] + (1 - alpha) * prev_smooth_point[1]
        
        return (smooth_x, smooth_y)
    
    def update_smooth_trail(self, track_id, center):
        """
        更新平滑轨迹
        Args:
            track_id: 目标ID
            center: 当前原始中心点 (x, y)
        Returns:
            平滑后的中心点 (x, y)
        """
        if track_id not in self.trails_smooth:
            # 第一笔平滑轨迹，直接使用当前原始点
            self.trails_smooth[track_id] = [center]
            return center
        else:
            prev_smooth_point = self.trails_smooth[track_id][-1]
            smooth_center = self.exponential_smooth_point(prev_smooth_point, center, self.smooth_alpha)
            self.trails_smooth[track_id].append(smooth_center)
            # 限制平滑轨迹长度
            if len(self.trails_smooth[track_id]) > self.trail_length:
                self.trails_smooth[track_id] = self.trails_smooth[track_id][-self.trail_length:]
            return smooth_center
    
    # ==================== Part B: 角度差计算函数 ====================
    def get_angle_diff_deg(self, angle1, angle2):
        """
        计算两个角度之间的差值，并处理 -180/180 跨界问题
        Returns: 角度差（带符号）
        """
        diff = angle2 - angle1
        
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        return diff
    
    # ==================== Part B: 单窗口角度变化判断 ====================
    def judge_lane_change_by_window(self, angle_diff_list, window_size, acc_threshold, consistent_ratio):
        """
        使用角度差序列进行窗口判断：
            同方向累积角度变化 + 方向一致性比例
        Returns: (是否触发, 触发方向) 方向: 'left' 或 'right'
        """
        # 角度数量不足，不判断
        if len(angle_diff_list) < window_size:
            return False, None
        
        # 取最近 window_size 个角度差
        diff_window = angle_diff_list[-window_size:]
        
        # 分别统计正向变化（向右）与负向变化（向左）
        # 注意：在图像坐标系中，角度为正表示向右，为负表示向左
        positive_diffs = [d for d in diff_window if d > 0]
        negative_diffs = [d for d in diff_window if d < 0]
        
        # 累积角度变化
        positive_sum = sum(positive_diffs)
        negative_sum = sum(abs(d) for d in negative_diffs)
        
        # 方向一致性比例
        positive_ratio = len(positive_diffs) / len(diff_window)
        negative_ratio = len(negative_diffs) / len(diff_window)
        
        if positive_sum >= acc_threshold and positive_ratio >= consistent_ratio:
            return True, 'right'
        
        if negative_sum >= acc_threshold and negative_ratio >= consistent_ratio:
            return True, 'left'
        
        return False, None
    
    # ==================== Part B: 双窗口角度变化判断 ====================
    def judge_lane_change_by_dual_window(self, angle_diff_list):
        """
        双窗口疑似变道判断：
            短窗口用于检测局部明显变化
            长窗口用于检测缓慢累积变化
        Returns: (是否触发, 触发类型) 类型: 'short' 或 'long'
        """
        # 短窗口判断
        short_result, _ = self.judge_lane_change_by_window(
            angle_diff_list,
            self.short_window_size,
            self.short_acc_threshold,
            self.short_consistent_ratio
        )
        
        if short_result:
            return True, 'short'
        
        # 长窗口判断
        long_result, _ = self.judge_lane_change_by_window(
            angle_diff_list,
            self.long_window_size,
            self.long_acc_threshold,
            self.long_consistent_ratio
        )
        
        if long_result:
            return True, 'long'
        
        return False, None
    
    # ==================== Part C: 净角度变化检查 ====================
    def judge_net_angle_change(self, angle_diff_list, window_size, net_threshold):
        """
        净角度变化确认：
            检查最近一段窗口内，正向累计与负向累计是否具有明显主导方向
        Returns: True（方向主导明显）/ False（来回摆动）
        """
        if len(angle_diff_list) < window_size:
            return False
        
        diff_window = angle_diff_list[-window_size:]
        
        positive_sum = sum(d for d in diff_window if d > 0)
        negative_sum = sum(abs(d) for d in diff_window if d < 0)
        
        net_diff = abs(positive_sum - negative_sum)
        
        return net_diff >= net_threshold
    
    def judge_net_angle_change_by_dual_window(self, angle_diff_list, trigger_type):
        """
        根据短窗口或长窗口触发结果，选择对应的净角度变化门槛
        """
        if trigger_type == 'short':
            return self.judge_net_angle_change(
                angle_diff_list,
                self.short_window_size,
                self.short_net_threshold
            )
        elif trigger_type == 'long':
            return self.judge_net_angle_change(
                angle_diff_list,
                self.long_window_size,
                self.long_net_threshold
            )
        return False
    
    # ==================== Part B: 横向位移确认 ====================
    def judge_lateral_shift_by_trail(self, trail):
        """
        使用轨迹点确认车辆是否具有明显横向位移
        Returns: True（有明显横向位移）/ False
        """
        # 条件一：轨迹点数量要足够
        if len(trail) < self.trajectory_window_size:
            return False
        
        # 取最近一段轨迹
        recent_trail = trail[-self.trajectory_window_size:]
        
        start_x, start_y = recent_trail[0]
        end_x, end_y = recent_trail[-1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        # 条件二：横向总位移要达到最小门槛
        if abs(dx) < self.min_lateral_shift:
            return False
        
        # 条件三：横向位移占纵向位移的比例要足够
        lateral_ratio = abs(dx) / (abs(dy) + 1e-6)
        if lateral_ratio < self.min_lateral_ratio:
            return False
        
        # 条件四：横向变化方向要有一致性
        x_diffs = []
        for i in range(1, len(recent_trail)):
            x_diffs.append(recent_trail[i][0] - recent_trail[i-1][0])
        
        if len(x_diffs) == 0:
            return False
        
        positive_count = sum(1 for d in x_diffs if d > 0)
        negative_count = sum(1 for d in x_diffs if d < 0)
        
        if dx > 0:
            x_consistent_ratio = positive_count / len(x_diffs)
        elif dx < 0:
            x_consistent_ratio = negative_count / len(x_diffs)
        else:
            x_consistent_ratio = 0
        
        return x_consistent_ratio >= self.min_x_consistent_ratio
    
    # ==================== 角度和向量计算辅助函数 ====================
    def get_motion_vector(self, p1, p2):
        """根据连续两个轨迹点，计算车辆当前的前进方向向量"""
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def get_vector_angle_deg(self, vec):
        """将方向向量转成角度（单位：度）"""
        vx, vy = vec
        if vx == 0 and vy == 0:
            return None
        return math.degrees(math.atan2(vy, vx))
    
    # ==================== 核心追踪方法 ====================
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
                center = (center_x, center_y)
                
                # 更新原始轨迹
                if draw_trail or analyze_lane_change:
                    if track_id in self.trails:
                        self.trails[track_id].append(center)
                    else:
                        self.trails[track_id] = [center]
                    self.lost_counter[track_id] = 0
                    
                    # 限制原始轨迹长度
                    if len(self.trails[track_id]) > self.trail_length:
                        self.trails[track_id] = self.trails[track_id][-self.trail_length:]
                    
                    # Part C: 更新平滑轨迹
                    smooth_center = self.update_smooth_trail(track_id, center)
                else:
                    smooth_center = center
                
                # Part B: 使用平滑轨迹进行变道分析
                if analyze_lane_change:
                    # Part B: 检查是否在监控区间内
                    if not self.in_valid_y_zone(center_y):
                        # 不在监控区间，不进行变道分析，只添加框
                        pred_boxes.append((x1, y1, x2, y2, lbl, track_id))
                        continue
                    
                    # 使用平滑轨迹获取轨迹点
                    trail_to_use = self.trails_smooth.get(track_id, self.trails.get(track_id, []))
                    
                    # Part B: 使用 ANGLE_STEP 间隔计算方向向量
                    if len(trail_to_use) >= self.angle_step + 1:
                        # 使用间隔轨迹点（前N个点和当前点）
                        p1 = trail_to_use[-1 - self.angle_step]
                        p2 = trail_to_use[-1]
                        
                        # 计算方向向量
                        vx, vy = self.get_motion_vector(p1, p2)
                        
                        # 计算角度
                        angle = None
                        if vx != 0 or vy != 0:
                            angle = self.get_vector_angle_deg((vx, vy))
                        
                        # 记录角度历史
                        if track_id not in self.angle_history:
                            self.angle_history[track_id] = []
                        if angle is not None:
                            self.angle_history[track_id].append(angle)
                        
                        # Part B: 计算角度差序列
                        angle_diff = None
                        if track_id not in self.angle_diffs:
                            self.angle_diffs[track_id] = []
                        
                        if len(self.angle_history[track_id]) > 1:
                            last_angle = self.angle_history[track_id][-2]
                            angle_diff = self.get_angle_diff_deg(last_angle, angle)
                            
                            # Part C: 异常角度过滤
                            if abs(angle_diff) <= self.abnormal_angle_diff_threshold:
                                self.angle_diffs[track_id].append(angle_diff)
                            else:
                                # 异常角度差，不加入序列
                                angle_diff = None
                        
                        # Part B: 双窗口角度变化判断
                        angle_ok, trigger_type = self.judge_lane_change_by_dual_window(
                            self.angle_diffs.get(track_id, [])
                        )
                        
                        # Part C: 净角度变化检查
                        net_angle_ok = False
                        if angle_ok and trigger_type:
                            net_angle_ok = self.judge_net_angle_change_by_dual_window(
                                self.angle_diffs.get(track_id, []),
                                trigger_type
                            )
                        else:
                            net_angle_ok = False
                        
                        # Part B: 横向位移确认
                        lateral_ok = self.judge_lateral_shift_by_trail(trail_to_use)
                        
                        # 最终判定：角度变化 + 净角度检查 + 横向位移 三者同时成立
                        if angle_ok and net_angle_ok and lateral_ok:
                            if track_id not in self.lane_change_results:
                                self.lane_change_results[track_id] = True
                        
                        # 记录CSV数据（包含 angle_diff）
                        self.analysis_records.append({
                            'frame_id': self.frame_id,
                            'track_id': track_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'smooth_x': smooth_center[0],
                            'smooth_y': smooth_center[1],
                            'vx': vx,
                            'vy': vy,
                            'angle': angle if angle is not None else 0,
                            'angle_diff': angle_diff if angle_diff is not None else 0,
                            'angle_ok': angle_ok,
                            'net_angle_ok': net_angle_ok,
                            'lateral_ok': lateral_ok
                        })
                        
                        # Part C 调试输出（观察 Track ID 28）
                        if track_id == 28:
                            print(f"track_id={track_id}, angle_ok={angle_ok}, trigger_type={trigger_type}, net_angle_ok={net_angle_ok}, lateral_ok={lateral_ok}")
                
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
                if tid in self.trails_smooth:
                    self.trails_smooth.pop(tid)
                if tid in self.lost_counter:
                    self.lost_counter.pop(tid)
                if tid in self.angle_history:
                    self.angle_history.pop(tid)
                if tid in self.angle_diffs:
                    self.angle_diffs.pop(tid)
                if tid in self.lane_change_results:
                    self.lane_change_results.pop(tid)
        
        # 绘制标注
        annotated_frame = self._draw_boxes(frame.copy(), pred_boxes)
        
        # 画轨迹线（T-3/T-4）- 使用平滑轨迹
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
            if self.lane_change_results.get(track_id, False):
                color = (0, 0, 255)  # 红色
            
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 标签文本
            if self.lane_change_results.get(track_id, False):
                text = f"{lbl}({track_id}) LANE CHANGE"
            else:
                text = f"{lbl}({track_id})"
            
            cv2.putText(im, text, (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return im
    
    def _draw_trails(self, im):
        """绘制轨迹线（使用平滑轨迹）"""
        for tid, trail in self.trails_smooth.items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                cv2.line(im, pt1, pt2, TRAIL_COLOR, 2)
        return im
    
    def _draw_lane_change_status(self, im, boxes):
        """显示变道状态"""
        for x1, y1, x2, y2, lbl, track_id in boxes:
            if self.lane_change_results.get(track_id, False):
                # 在框上方显示变道文字
                text = "LANE CHANGE"
                text_x = int(x1)
                text_y = int(y1) - 25
                cv2.putText(im, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # 显示 Normal 状态（可选）
                if self.in_valid_y_zone((y1 + y2) / 2):
                    text = "Normal"
                    text_x = int(x1)
                    text_y = int(y2) + 20
                    cv2.putText(im, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return im
    
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
            fieldnames = ['frame_id', 'track_id', 'center_x', 'center_y', 
                         'smooth_x', 'smooth_y', 'vx', 'vy', 'angle', 
                         'angle_diff', 'angle_ok', 'net_angle_ok', 'lateral_ok']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.analysis_records)
        
        print(f"分析数据已导出到: {csv_path}")
        print(f"共导出 {len(self.analysis_records)} 条记录")
    
    def get_angle_history(self, track_id):
        """获取指定track_id的角度历史"""
        return self.angle_history.get(track_id, [])
    
    def get_angle_diffs(self, track_id):
        """获取指定track_id的角度差序列"""
        return self.angle_diffs.get(track_id, [])
    
    def get_lane_change_result(self, track_id):
        """获取指定track_id的变道结果"""
        return self.lane_change_results.get(track_id, False)