# p1_SuspectedLaneChange.py

"""
T-4: 疑似变道行为检测
功能：追踪 + 轨迹 + 方向向量 + 角度 + 变道判断 + CSV导出
"""

import cv2
import os
from yolo_tracker_base import YOLOTracker, VIDEOS_DIR, OUTPUT_DIR, PROJECT_ROOT

# ========== 配置 ==========
VIDEO_NAME = "2.mp4"                      # 视频文件名
SAVE_VIDEO = True                          # 是否保存结果视频
OUTPUT_NAME = "p1_SuspectedLaneChange.mp4" # 输出视频文件名
CSV_NAME = "analysis_records.csv"          # CSV输出文件名

# ========== 俯视 / 虚线变道专用参数 ==========

MIN_TRAIL_POINTS = 15        # 俯视下轨迹短一点也能判断
ANGLE_STEP = 3               # 减小间隔，及时捕捉方向变化

# 监控区间（全画面）
VALID_Y_MIN = 0
VALID_Y_MAX = 9999

# 短窗口（快速 / 虚线变道）
SHORT_WINDOW_SIZE = 8
SHORT_ACC_THRESHOLD = 14      # 提高，减少直行车微小摆动误报
SHORT_CONSISTENT_RATIO = 0.55

# 长窗口
LONG_WINDOW_SIZE = 20
LONG_ACC_THRESHOLD = 25       # 提高，减少缓慢累积的误报
LONG_CONSISTENT_RATIO = 0.50

# 横向位移确认（俯视下调整）
TRAJECTORY_WINDOW_SIZE = 15
MIN_LATERAL_SHIFT = 8         # 降低，因为俯视画面有限
MIN_LATERAL_RATIO = 0.08      # 放宽
MIN_X_CONSISTENT_RATIO = 0.50

# 净角度变化
SHORT_NET_THRESHOLD = 10
LONG_NET_THRESHOLD = 15

# 异常角度过滤（最关键）
ABNORMAL_ANGLE_DIFF = 15.0    # 放宽，避免切除变道信号

# 速度过滤：静止/慢速车辆跳过变道分析
MIN_SPEED = 2.5               # 像素/帧，低于此值视为静止

# 横向位移轴：水平道路（左到右）用 'y'，垂直道路用 'x'
LATERAL_AXIS = 'y'

SMOOTH_ALPHA = 0.4
TRAIL_LENGTH = 40

def main():
    video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print(f"请将视频文件放入 {VIDEOS_DIR} 文件夹")
        return
    
    tracker = YOLOTracker()
    
    # ===== 传入所有参数 =====
    tracker.trail_length = MIN_TRAIL_POINTS
    tracker.angle_step = ANGLE_STEP
    
    tracker.valid_y_min = VALID_Y_MIN
    tracker.valid_y_max = VALID_Y_MAX
    
    tracker.short_window_size = SHORT_WINDOW_SIZE
    tracker.short_acc_threshold = SHORT_ACC_THRESHOLD
    tracker.short_consistent_ratio = SHORT_CONSISTENT_RATIO
    
    tracker.long_window_size = LONG_WINDOW_SIZE
    tracker.long_acc_threshold = LONG_ACC_THRESHOLD
    tracker.long_consistent_ratio = LONG_CONSISTENT_RATIO
    
    tracker.trajectory_window_size = TRAJECTORY_WINDOW_SIZE
    tracker.min_lateral_shift = MIN_LATERAL_SHIFT
    tracker.min_lateral_ratio = MIN_LATERAL_RATIO
    tracker.min_x_consistent_ratio = MIN_X_CONSISTENT_RATIO
    
    tracker.short_net_threshold = SHORT_NET_THRESHOLD
    tracker.long_net_threshold = LONG_NET_THRESHOLD
    
    tracker.smooth_alpha = SMOOTH_ALPHA
    tracker.abnormal_angle_diff_threshold = ABNORMAL_ANGLE_DIFF
    tracker.min_speed = MIN_SPEED
    tracker.lateral_axis = LATERAL_AXIS
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 窗口设置（等比例缩放）
    max_width = 1536
    scale = max_width / w if w > max_width else 1
    new_w = int(w * scale)
    new_h = int(h * scale)
    cv2.namedWindow('Suspected Lane Change', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Suspected Lane Change', new_w, new_h)
    cv2.moveWindow('Suspected Lane Change', 100, 50)
    
    # 视频写入器
    out = None
    if SAVE_VIDEO:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"结果视频将保存到: {output_path}")
    
    print("开始变道检测，按 'q' 退出...")
    print(f"视频分辨率: {w}x{h}, FPS: {fps:.1f}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 追踪 + 画轨迹 + 变道分析
        annotated_frame, _ = tracker.track(frame, draw_trail=True, analyze_lane_change=True)
        
        cv2.imshow('Suspected Lane Change', annotated_frame)
        
        if out:
            out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 导出CSV
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    tracker.export_csv(csv_path)
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("检测完成")

if __name__ == "__main__":
    main()