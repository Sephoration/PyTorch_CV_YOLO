# T5_counting.py

"""
T-5: 目标计数 (Object Counting)
功能：追踪 + 穿越中心水平线计数（向上/向下）
基于PPT中的 trigger() 函数实现

核心逻辑：
1. 在图像高度的中心位置画一条水平线
2. 判断每个目标的中心点位于线的哪一侧
3. 当目标穿越线时，根据方向进行计数
4. 使用 tracker_state 和 prev_tracker_state 管理状态，避免重复计数
"""

import cv2
import os
import numpy as np
from collections import namedtuple
from yolo_tracker_base import YOLOTracker, VIDEOS_DIR, OUTPUT_DIR

# ========== 配置 ==========
VIDEO_NAME = "Traffic-1.mp4"      # 视频文件名
SAVE_VIDEO = False                  # 是否保存结果视频
OUTPUT_NAME = "T5_output.mp4"      # 输出视频文件名

# 计数线偏移量（像素）
# offset = 0 表示正中央
# offset > 0 表示线上移，offset < 0 表示线下移
OFFSET = 0

# 显示窗口缩放宽度（像素）
DISPLAY_WIDTH = 1200

# 计数状态管理参数
MAX_LOST_FRAMES = 30  # 目标消失多少帧后完全清除状态
# ==========================

# 定义 Point 类（用于 is_in_line 函数）
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def is_in_line(pt1, pt2, pt):
    """
    判断某个点 pt 位于由两点 pt1 到 pt2 所构成的直线的哪一侧
    
    使用向量叉积计算：
    叉积 = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    
    Args:
        pt1: 线段起点 (Point)
        pt2: 线段终点 (Point)
        pt: 待判断的点 (Point)
    
    Returns:
        >0: 点在线的某一侧
        <0: 点在线的另一侧
        =0: 点在线上
    """
    x1, y1 = pt1.x, pt1.y
    x2, y2 = pt2.x, pt2.y
    x, y = pt.x, pt.y
    
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return np.sign(cross)


def trigger(detections, pt1, pt2, prev_tracker_state, tracker_state, 
            crossing_ids, in_count, out_count):
    """
    核心计数函数：判断目标是否穿越虚拟线，并统计穿越次数
    
    参数说明：
        detections: 当前帧检测到的目标列表
                    格式: [(xyxy, track_id), ...]
                    xyxy: (x1, y1, x2, y2)
        pt1, pt2: 线的起点和终点 (Point)
        prev_tracker_state: 历史状态字典（目标消失前保存的状态）
        tracker_state: 当前状态字典
                       {tracker_id: {'state': bool, 'direction': str}, ...}
                       state: True=线上方, False=线下方
                       direction: 'up' 或 'down'
        crossing_ids: 已穿越目标ID集合（用于额外防重复）
        in_count: 向上穿越计数（从下往上 → UP）
        out_count: 向下穿越计数（从上往下 → DOWN）
    
    返回值：
        (in_count, out_count) 更新后的计数
    """
    for xyxy, tracker_id in detections:
        x1, y1, x2, y2 = xyxy
        # 计算目标中心点
        center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
        
        # 判断中心点在线的那一侧（True表示在线的一侧，False表示另一侧）
        # 这里使用 >= 0 来定义某一侧（可根据需要调整符号）
        tracker_state_new = is_in_line(pt1, pt2, center) >= 0
        
        # ===== 情况1：第一次侦测到该目标，或目标之前消失后重新出现 =====
        if tracker_id not in tracker_state or tracker_state[tracker_id] is None:
            # 初始化状态
            tracker_state[tracker_id] = {
                'state': tracker_state_new,
                'direction': None
            }
            
            # 如果之前有历史记录（目标消失过），恢复其方向
            if (tracker_id in prev_tracker_state and 
                prev_tracker_state[tracker_id] is not None):
                tracker_state[tracker_id]['direction'] = prev_tracker_state[tracker_id]['direction']
        
        # ===== 情况2：目标仍在线的同一侧，没有穿越 =====
        elif tracker_state[tracker_id]['state'] == tracker_state_new:
            continue
        
        # ===== 情况3：目标完全穿越了线 =====
        else:
            # 从上往下穿越（state从True变为False）
            if tracker_state[tracker_id]['state'] and not tracker_state_new:
                # 只有方向不是'down'时才计数（避免重复计数）
                if tracker_state[tracker_id]['direction'] != 'down':
                    out_count += 1  # 向下穿越，OUT计数增加
                tracker_state[tracker_id]['direction'] = 'down'
            
            # 从下往上穿越（state从False变为True）
            elif not tracker_state[tracker_id]['state'] and tracker_state_new:
                if tracker_state[tracker_id]['direction'] != 'up':
                    in_count += 1   # 向上穿越，IN计数增加
                tracker_state[tracker_id]['direction'] = 'up'
            
            # 更新状态
            tracker_state[tracker_id]['state'] = tracker_state_new
    
    # ===== 处理已经消失的目标 =====
    # 获取当前帧所有检测到的tracker_id
    current_ids = [tid for _, tid in detections]
    
    for tracker_id in list(tracker_state.keys()):
        if tracker_id not in current_ids:
            # 目标消失，保存其状态到历史记录
            prev_tracker_state[tracker_id] = tracker_state[tracker_id]
            tracker_state[tracker_id] = None
    
    return in_count, out_count


def main():
    # 获取视频完整路径
    video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print(f"请将视频文件放入 {VIDEOS_DIR} 文件夹")
        return
    
    # 初始化追踪器
    tracker = YOLOTracker()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频参数
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    
    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 计算计数线的位置（图像高度的中心 ± 偏移量）
    line_y = frame_height // 2 + OFFSET
    pt1 = Point(0, line_y)
    pt2 = Point(frame_width, line_y)
    
    print(f"视频分辨率: {frame_width} x {frame_height}")
    print(f"计数线位置: Y = {line_y}")
    print(f"向上计数(UP) = 从下往上穿越")
    print(f"向下计数(DOWN) = 从上往下穿越")
    print("按 'q' 退出...")
    
    # 计算显示窗口大小（保持宽高比）
    display_width = DISPLAY_WIDTH
    display_height = int(frame_height * (display_width / frame_width))
    
    # 设置窗口
    cv2.namedWindow('T-5: Object Counting', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('T-5: Object Counting', display_width, display_height)
    cv2.moveWindow('T-5: Object Counting', 100, 50)
    
    # 视频写入器
    out = None
    if SAVE_VIDEO:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, 
                              (frame_width, frame_height))
        print(f"结果视频将保存到: {output_path}")
    
    # ========== 计数相关变量初始化 ==========
    in_count = 0      # 向上计数（UP）
    out_count = 0     # 向下计数（DOWN）
    tracker_state = {}        # 当前状态
    prev_tracker_state = {}   # 历史状态（目标消失时保存）
    crossing_ids = set()      # 已穿越目标ID集合（备用防重复）
    
    # 记录每个目标消失的帧数（用于清理长期消失的目标）
    lost_frames_counter = {}
    
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # 执行追踪（不画轨迹，不变道分析）
        annotated_frame, pred_boxes = tracker.track(frame, draw_trail=False, analyze_lane_change=False)
        
        # 将追踪结果转换为 trigger 函数需要的格式
        # pred_boxes 格式: [(x1, y1, x2, y2, lbl, track_id), ...]
        detections = []
        current_ids = []
        
        for box in pred_boxes:
            x1, y1, x2, y2, lbl, track_id = box
            detections.append(((x1, y1, x2, y2), track_id))
            current_ids.append(track_id)
            
            # 重置消失计数器
            lost_frames_counter[track_id] = 0
        
        # ===== 处理消失的目标（清理长期消失的ID）=====
        remove_ids = []
        for tid in list(tracker_state.keys()):
            if tid not in current_ids:
                lost_frames_counter[tid] = lost_frames_counter.get(tid, 0) + 1
                if lost_frames_counter[tid] > MAX_LOST_FRAMES:
                    remove_ids.append(tid)
        
        for tid in remove_ids:
            if tid in tracker_state:
                tracker_state.pop(tid)
            if tid in prev_tracker_state:
                prev_tracker_state.pop(tid)
            if tid in lost_frames_counter:
                lost_frames_counter.pop(tid)
            if tid in crossing_ids:
                crossing_ids.discard(tid)
        
        # ===== 调用 trigger 核心计数函数 =====
        in_count, out_count = trigger(
            detections, pt1, pt2,
            prev_tracker_state, tracker_state, crossing_ids,
            in_count, out_count
        )
        
        # ===== 在画面上绘制计数线和计数信息 =====
        # 画计数线（红色，粗细2）
        cv2.line(annotated_frame, (pt1.x, pt1.y), (pt2.x, pt2.y), (0, 0, 255), 2)
        
        # 在线的两端添加小圆点标记
        cv2.circle(annotated_frame, (pt1.x, pt1.y), 5, (0, 0, 255), -1)
        cv2.circle(annotated_frame, (pt2.x, pt2.y), 5, (0, 0, 255), -1)
        
        # 在线的左侧显示 "COUNT LINE" 文字
        cv2.putText(annotated_frame, "COUNT LINE", 
                    (pt1.x + 10, pt1.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示计数结果（PPT要求：UP / DOWN）
        # 注意：图像坐标系中，Y轴向下为正
        # UP = 从下往上穿越（IN_COUNT）
        # DOWN = 从上往下穿越（OUT_COUNT）
        text_up = f"UP: {in_count}"
        text_down = f"DOWN: {out_count}"
        
        # 左上角显示UP计数（绿色）
        cv2.putText(annotated_frame, text_up, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 左上角显示DOWN计数（红色）
        cv2.putText(annotated_frame, text_down, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # 可选：在线的旁边显示方向提示
        cv2.putText(annotated_frame, "UP ↑", (frame_width - 80, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_frame, "DOWN ↓", (frame_width - 100, line_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 显示帧数和当前活跃目标数（调试信息）
        active_count = sum(1 for v in tracker_state.values() if v is not None)
        debug_text = f"Frame: {frame_id} | Active IDs: {active_count}"
        cv2.putText(annotated_frame, debug_text, (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示画面
        cv2.imshow('T-5: Object Counting', annotated_frame)
        
        # 保存视频
        if out:
            out.write(annotated_frame)
        
        # 按键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            break
    
    # ========== 打印最终统计结果 ==========
    print("\n" + "="*50)
    print("目标计数结果统计")
    print("="*50)
    print(f"向上穿越计数 (UP / IN_COUNT):  {in_count}")
    print(f"向下穿越计数 (DOWN / OUT_COUNT): {out_count}")
    print(f"总计穿越: {in_count + out_count}")
    print(f"总处理帧数: {frame_id}")
    print("="*50)
    
    # 可选：保存计数结果到文本文件
    result_path = os.path.join(OUTPUT_DIR, "counting_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("目标计数结果\n")
        f.write(f"视频文件: {VIDEO_NAME}\n")
        f.write(f"计数线位置: Y = {line_y}\n")
        f.write(f"向上穿越 (UP): {in_count}\n")
        f.write(f"向下穿越 (DOWN): {out_count}\n")
        f.write(f"总计: {in_count + out_count}\n")
    print(f"计数结果已保存到: {result_path}")
    
    # 释放资源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("T-5 完成")


if __name__ == "__main__":
    main()