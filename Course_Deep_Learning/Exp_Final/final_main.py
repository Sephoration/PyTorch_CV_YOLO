# final_main.py
"""
期末报告 — 目标追踪与分类计数 + 敏感区域监控
功能整合：
  1. 动态轨迹颜色 (每 ID 不同色)
  2. 分车型计数 (car / bus / truck) 双向
  3. 敏感区域监控 (多边形 + 入侵报警)
  4. FPS 显示
  5. 键盘控制

操作键：
  q / ESC  → 退出
  c        → 切换计数线显示
  z        → 切换敏感区域显示
  t        → 切换彩色轨迹
  r        → 重置计数
"""

import cv2
import os
import sys

# 加入可能的影片来源路径
from yolo_tracker_base import (
    YOLOTracker, VIDEOS_DIR, OUTPUT_DIR, PROJECT_ROOT,
    Point, isInsidePolygon, drawAndFillPolygon, adaptive_resize
)

# ============================================================
# 影片搜寻（多来源）
# ============================================================
def find_video(filename):
    """从多个目录搜寻影片"""
    search_dirs = [
        VIDEOS_DIR,
        os.path.join(PROJECT_ROOT, '..', 'Exp_Midterm', 'videos'),
        os.path.join(PROJECT_ROOT, '..', '..', 'YOLO26Tracking', 'videos'),
    ]
    for d in search_dirs:
        abs_d = os.path.abspath(d)
        path = os.path.join(abs_d, filename)
        if os.path.exists(path):
            return path
    return None


# ============================================================
# 配置
# ============================================================
VIDEO_NAME = "Traffic-1.mp4"       # 启用影片
SAVE_VIDEO = True
OUTPUT_NAME = "final_result.mp4"

DISPLAY_WIDTH = 1280

# 计数参考线 Y 比例 (相对于画面高度)
COUNT_LINE_Y_RATIO = 0.60

# 敏感区域多边形（相对坐标比例）
ZONE_POLYGON_RATIO = [
    (0.15, 0.95),   # 左下
    (0.85, 0.95),   # 右下
    (0.75, 0.40),   # 右上
    (0.25, 0.40),   # 左上
]


def main():
    # ---- 寻找影片 ----
    video_path = find_video(VIDEO_NAME)
    if not video_path:
        print(f"影片不存在！请将 {VIDEO_NAME} 放入以下任一目录：")
        print(f"  - {VIDEOS_DIR}")
        print(f"  - Exp_Midterm/videos/")
        print(f"  - YOLO26Tracking/videos/")
        return

    print(f"使用影片: {video_path}")

    # ---- 初始化追踪器 ----
    tracker = YOLOTracker()
    tracker.colorful_trail = True   # 开启彩色轨迹

    # ---- 开启影片 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开影片")
        return

    # 读取第一帧获取参数
    ret, frame = cap.read()
    if not ret:
        print("无法读取影片帧")
        return
    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"分辨率: {frame_width}x{frame_height}  FPS: {fps:.1f}")

    # ---- 设定计数参考线 ----
    count_line_y = int(frame_height * COUNT_LINE_Y_RATIO)
    tracker.count_line = ((0, count_line_y), (frame_width, count_line_y))
    tracker.count_enabled = True

    # ---- 设定敏感区域多边形 ----
    zone_polygon = [
        (int(frame_width * rx), int(frame_height * ry))
        for rx, ry in ZONE_POLYGON_RATIO
    ]
    tracker.set_zone_polygon(zone_polygon)
    print(f"敏感区域顶点: {zone_polygon}")

    # ---- 视窗设定 ----
    win_name = 'Final Project - Tracking + Counting + Zone'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_WIDTH,
                     int(frame_height * (DISPLAY_WIDTH / frame_width)))
    cv2.moveWindow(win_name, 100, 50)

    # ---- 输出影片 ----
    out = None
    if SAVE_VIDEO:
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (frame_width, frame_height))
        print(f"输出影片 → {out_path}")

    # ---- 显示开关 ----
    show_count_line = True
    show_zone = True
    print(f"\n{'='*50}")
    print("操作键:")
    print("  q/ESC → 退出    c → 切换计数线")
    print("  z     → 切换区域  t → 切换彩色轨迹")
    print("  r     → 重置计数")
    print(f"{'='*50}\n")

    # ---- 主循环 ----
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 追踪 + 画轨迹
        annotated, pred_boxes = tracker.track(frame, draw_trail=True)

        # ---- 1. 敏感区域 ----
        if show_zone:
            annotated = tracker.draw_zone(annotated)
            current_inside = tracker.check_zone_intrusions(pred_boxes,
                                                           tracker.frame_id, fps)
            # 异常截图：新进入区域的目标自动裁切保存
            if current_inside:
                for box in pred_boxes:
                    x1, y1, x2, y2, lbl, track_id = box
                    if track_id in current_inside:
                        tracker.capture_snapshot(annotated, track_id, x1, y1, x2, y2, lbl)
                        state = tracker.zone_intrusions.get(track_id, {})
                        dur = state.get('total_frames_inside', 0) / fps
                        cv2.rectangle(annotated,
                                      (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 0, 255), 3)
                        cv2.putText(annotated, f"INTRUSION! {lbl}({track_id})",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(annotated, f"Time: {dur:.1f}s",
                                    (int(x1), int(y2) + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ---- 2. 计数线 ----
        if show_count_line:
            annotated = tracker.draw_count_line(annotated)

        # ---- 3. FPS ----
        annotated = tracker.draw_fps(annotated)

        # ---- 4. 计数统计面板（右下角） ----
        stats_x = frame_width - 260
        stats_y = 10
        overlay = annotated.copy()
        cv2.rectangle(overlay, (stats_x, stats_y),
                      (stats_x + 250, stats_y + 140), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)
        cv2.putText(annotated, "=== Vehicle Count ===",
                    (stats_x + 10, stats_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        yy = stats_y + 50
        for cls_name in ['car', 'bus', 'truck']:
            d = tracker.count_data.get(cls_name, {'in': 0, 'out': 0})
            text = f"{cls_name}: IN {d['in']}  OUT {d['out']}"
            color = (255, 0, 0) if cls_name == 'car' else \
                    (0, 0, 255) if cls_name == 'bus' else (0, 255, 255)
            cv2.putText(annotated, text, (stats_x + 10, yy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            yy += 28

        # ---- 5. 左下提示 ----
        cv2.putText(annotated, "'c':line 'z':zone 't':color 'r':reset 'q':quit",
                    (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1)

        # ---- 显示 ----
        display = adaptive_resize(annotated, DISPLAY_WIDTH)
        cv2.imshow(win_name, display)

        if out:
            out.write(annotated)

        # ---- 按键 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            show_count_line = not show_count_line
            print(f"计数线: {'显示' if show_count_line else '隐藏'}")
        elif key == ord('z'):
            show_zone = not show_zone
            print(f"敏感区域: {'显示' if show_zone else '隐藏'}")
        elif key == ord('t'):
            tracker.colorful_trail = not tracker.colorful_trail
            print(f"彩色轨迹: {'开启' if tracker.colorful_trail else '关闭'}")
        elif key == ord('r'):
            tracker.count_data = {k: {'in': 0, 'out': 0} for k in tracker.count_data}
            tracker._count_history.clear()
            print("[重置] 计数已归零")
            for cls_name in ['car', 'bus', 'truck']:
                tracker.count_data[cls_name] = {'in': 0, 'out': 0}

    # ---- 结束 ----
    print(f"\n{'='*50}")
    print("最终统计摘要")
    print(f"{'='*50}")
    print(tracker.get_count_summary())
    if tracker.zone_intrusions:
        print(f"\n敏感区域入侵: {len(tracker.zone_intrusions)} 个目标")
        for tid, st in tracker.zone_intrusions.items():
            print(f"  ID:{tid} ({st['label']}) - "
                  f"累计 {st['total_frames_inside']} 帧")
    print(f"{'='*50}")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("期末报告程序完成")


if __name__ == "__main__":
    main()
