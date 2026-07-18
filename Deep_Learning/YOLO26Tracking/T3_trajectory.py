# T3_trajectory.py (修改后)

"""
T-3: 目标轨迹
功能：追踪 + 画轨迹线
"""

import cv2
import os
from yolo_tracker_base import YOLOTracker, VIDEOS_DIR, OUTPUT_DIR

# ========== 配置 ==========
VIDEO_NAME = "Traffic-1.mp4"      # 视频文件名
SAVE_VIDEO = False                  # 是否保存结果视频
OUTPUT_NAME = "T3_output.mp4"      # 输出视频文件名
TRAIL_LENGTH = 50                  # 轨迹最大长度（帧数）
# ==========================

def main():
    # 获取视频完整路径
    video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print(f"请将视频文件放入 {VIDEOS_DIR} 文件夹")
        return
    
    tracker = YOLOTracker()
    tracker.trail_length = TRAIL_LENGTH  # 设置轨迹长度
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    out = None
    if SAVE_VIDEO:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"结果视频将保存到: {output_path}")
    
    print("开始追踪轨迹，按 'q' 退出...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 追踪 + 画轨迹
        annotated_frame, _ = tracker.track(frame, draw_trail=True, analyze_lane_change=False)
        
        cv2.imshow("T-3: Tracking with Trail", annotated_frame)
        
        if out:
            out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("T-3 完成")

if __name__ == "__main__":
    main()