# T4_lane_change.py (修改后)

"""
T-4: 疑似变道行为检测
功能：追踪 + 轨迹 + 方向向量 + 角度 + 变道判断 + CSV导出
文件名按PPT要求: p1_SuspectedLaneChange.py
"""

import cv2
import os
from yolo_tracker_base import YOLOTracker, VIDEOS_DIR, OUTPUT_DIR, PROJECT_ROOT

# ========== 配置 ==========
VIDEO_NAME = "Traffic-3.mp4"          # 视频文件名
SAVE_VIDEO = True                      # 是否保存结果视频
OUTPUT_NAME = "p1_SuspectedLaneChange.mp4"  # 输出视频文件名
CSV_NAME = "analysis_records.csv"     # CSV输出文件名

# 变道判断参数（可根据需要调整）
CHANGE_THRESHOLD = 25    # 累计变化超过此角度触发
CHANGE_WINDOW = 15       # 观察窗口帧数
CHANGE_MIN_FRAMES = 10   # 最少持续帧数
# ==========================

def main():
    # 获取视频完整路径
    video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print(f"请将视频文件放入 {VIDEOS_DIR} 文件夹")
        return
    
    tracker = YOLOTracker()
    
    # 设置变道判断参数
    tracker.change_threshold = CHANGE_THRESHOLD
    tracker.change_window = CHANGE_WINDOW
    tracker.change_min_frames = CHANGE_MIN_FRAMES
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频参数
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 窗口设置（PPT要求：等比例缩放）
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
    
    # 导出CSV（保存到输出目录）
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    tracker.export_csv(csv_path)
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("T-4 完成")

if __name__ == "__main__":
    main()