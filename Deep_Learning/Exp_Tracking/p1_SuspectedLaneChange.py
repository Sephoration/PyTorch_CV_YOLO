# p1_SuspectedLaneChange.py
"""
期中报告 — 疑似变道行为检测
场景：高架桥俯拍，车辆从远到近，双车道
方法：角度变化检测 + 横向位移确认
"""

import cv2
import os
from yolo_tracker_base import YOLOTracker, VIDEOS_DIR, OUTPUT_DIR

# ============================================================
# 视频配置
# ============================================================
VIDEO_NAME = "exp_1.mp4"
SAVE_VIDEO = True
OUTPUT_NAME = "p1_LC_result.mp4"
CSV_NAME = "LC_records.csv"

# ============================================================
# 轨迹参数
# ============================================================
TRAIL_LENGTH = 80       # 轨迹最大缓冲长度
ANGLE_STEP = 3          # 方向向量间隔帧数（跳帧抑制噪声）
SMOOTH_ALPHA = 0.35     # 指数平滑系数

# ============================================================
# 角度变化检测（三窗口：短/中/长，覆盖快中慢变道）
# ============================================================
SHORT_WINDOW_SIZE = 8
SHORT_ACC_THRESHOLD = 25        # 短窗口累积角度阈值
SHORT_CONSISTENT_RATIO = 0.55   # 短窗口方向一致性

LONG_WINDOW_SIZE = 20
LONG_ACC_THRESHOLD = 35         # 中窗口累积角度阈值
LONG_CONSISTENT_RATIO = 0.50    # 中窗口方向一致性

LONG2_WINDOW_SIZE = 60          # 大窗口（慢速变道）
LONG2_ACC_THRESHOLD = 20        # 大窗口累积角度阈值（低门槛靠帧数积累）
LONG2_ACC_THRESHOLD = 20        # 大窗口累积角度阈值（低门槛靠帧数积累）
LONG2_CONSISTENT_RATIO = 0.40   # 大窗口方向一致性

SHORT_NET_THRESHOLD = 15        # 短窗口净角度门槛
LONG_NET_THRESHOLD = 25         # 中窗口净角度门槛
LONG2_NET_THRESHOLD = 20        # 大窗口净角度门槛

# ============================================================
# 透视补偿（远处自动放宽阈值）
# ============================================================
PERSPECTIVE_ENABLED = True
VALID_Y_MIN = 120
VALID_Y_MAX = 600

# ============================================================
# 横向位移确认
# ============================================================
TRAJECTORY_WINDOW_SIZE = 30
MIN_LATERAL_SHIFT = 40          # 最少横向位移量（像素）
MIN_LATERAL_RATIO = 0.15        # 横向/纵向比例
MIN_X_CONSISTENT_RATIO = 0.50   # 横向方向一致性
# ============================================================
# 过滤参数
# ============================================================
ABNORMAL_ANGLE_DIFF = 20.0      # 单帧角度跳变阈值
MIN_SPEED = 2.0                 # 最低速度（像素/帧）

# ============================================================
# 道路方向（车从上往下 → 前进Y轴，横向X轴）
# ============================================================
LATERAL_AXIS = 'x'


def main():
    video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    tracker = YOLOTracker()

    # ---- 应用参数 ----
    tracker.trail_length = TRAIL_LENGTH
    tracker.angle_step = ANGLE_STEP
    tracker.smooth_alpha = SMOOTH_ALPHA

    tracker.short_window_size = SHORT_WINDOW_SIZE
    tracker.short_acc_threshold = SHORT_ACC_THRESHOLD
    tracker.short_consistent_ratio = SHORT_CONSISTENT_RATIO
    tracker.long_window_size = LONG_WINDOW_SIZE
    tracker.long_acc_threshold = LONG_ACC_THRESHOLD
    tracker.long_consistent_ratio = LONG_CONSISTENT_RATIO
    tracker.long2_window_size = LONG2_WINDOW_SIZE
    tracker.long2_acc_threshold = LONG2_ACC_THRESHOLD
    tracker.long2_consistent_ratio = LONG2_CONSISTENT_RATIO
    tracker.short_net_threshold = SHORT_NET_THRESHOLD
    tracker.long_net_threshold = LONG_NET_THRESHOLD
    tracker.long2_net_threshold = LONG2_NET_THRESHOLD
    tracker.valid_y_min = VALID_Y_MIN
    tracker.valid_y_max = VALID_Y_MAX
    tracker.perspective_enabled = PERSPECTIVE_ENABLED

    tracker.trajectory_window_size = TRAJECTORY_WINDOW_SIZE
    tracker.min_lateral_shift = MIN_LATERAL_SHIFT
    tracker.min_lateral_ratio = MIN_LATERAL_RATIO
    tracker.min_x_consistent_ratio = MIN_X_CONSISTENT_RATIO
    tracker.abnormal_angle_diff_threshold = ABNORMAL_ANGLE_DIFF
    tracker.min_speed = MIN_SPEED
    tracker.lateral_axis = LATERAL_AXIS

    # ---- 开视频 ----
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 窗口
    max_width = 1536
    scale = max_width / w if w > max_width else 1
    cv2.namedWindow('Suspected Lane Change', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Suspected Lane Change', int(w * scale), int(h * scale))

    # 输出视频
    out = None
    if SAVE_VIDEO:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"输出视频 → {output_path}")

    print(f"分辨率: {w}x{h} | FPS: {fps:.1f} | 总帧数: {total_frames}")
    print("按 'q' 退出\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, _ = tracker.track(
            frame, draw_trail=True, analyze_lane_change=True)

        # 叠加信息（半透明背景 + 清晰文字）
        fid = tracker.frame_id
        lc_count = sum(1 for v in tracker.lane_change_results.values() if v)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (w - 230, 5), (w - 5, 75), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0)
        cv2.putText(annotated_frame, f"Frame: {fid}/{total_frames}",
                    (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"LC: {lc_count}",
                    (w - 220, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow('Suspected Lane Change', annotated_frame)
        if out:
            out.write(annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC 退出
            break

    # ---- 摘要 ----
    lc_ids = [tid for tid, v in tracker.lane_change_results.items() if v]
    total = len(tracker.lane_change_results)
    print(f"\n=== 检测摘要 ===")
    print(f"追踪车辆总数: {total}")
    print(f"疑似变道车辆: {len(lc_ids)} 台 — ID: {lc_ids}")
    print(f"正常行驶: {total - len(lc_ids)} 台")

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    tracker.export_csv(csv_path)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("检测完成")


if __name__ == "__main__":
    main()
