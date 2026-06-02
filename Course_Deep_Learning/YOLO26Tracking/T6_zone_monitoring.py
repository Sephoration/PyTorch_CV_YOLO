# T6_zone_monitoring.py

"""
T-6: 敏感区域监控 (Sensitive Zone Monitoring)
功能：追踪 + 多边形敏感区域入侵检测 + 计时 + 截图

核心逻辑：
1. 在视频画面上定义一个多边形敏感区域
2. 使用射线投射算法（Ray Casting）判断目标中心点是否在区域内
3. 检测到入侵时显示红色警告，记录入侵持续时间
4. 自动保存入侵截图（含 track_id 和时间戳）
"""

import cv2
import os
import time
import numpy as np
from datetime import datetime
from yolo_tracker_base import YOLOTracker, VIDEOS_DIR, OUTPUT_DIR


# ========== 配置 ==========
VIDEO_NAME = "Traffic-1.mp4"        # 视频文件名
SAVE_VIDEO = False                   # 是否保存结果视频
OUTPUT_NAME = "T6_output.mp4"       # 输出视频文件名

# 显示窗口缩放宽度（像素）
DISPLAY_WIDTH = 1200

# 敏感区域颜色（半透明覆盖层颜色）
ZONE_COLOR = (0, 0, 255)            # 红色 (BGR)
ZONE_ALPHA = 0.3                    # 透明度 0~1

# 截图保存目录
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "intrusion_screenshots")

# 入侵检测参数
INTRUSION_COOLDOWN = 30             # 同一目标连续入侵至少间隔多少帧才再次截图
# ==========================


class Point:
    """二维点类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def isInsidePolygon(pt, polygon):
    """
    射线投射法（Ray Casting）判断点是否在多边形内部

    从目标点向右发射一条水平射线，统计与多边形边的交点数：
    - 奇数个交点 → 点在多边形内部
    - 偶数个交点 → 点在多边形外部

    Args:
        pt: 待判断的点 (Point)
        polygon: 多边形顶点列表 [(x1,y1), (x2,y2), ...]，按顺序排列

    Returns:
        True: 点在多边形内部
        False: 点在多边形外部
    """
    x, y = pt.x, pt.y
    n = len(polygon)
    inside = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # 检查水平射线是否与当前边相交
        # 条件：(y 在 y1 和 y2 之间) 且 (交点 x 在目标点 x 的右侧)
        if ((y1 > y) != (y2 > y)):
            # 计算水平射线与边的交点 x 坐标
            intersect_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersect_x:
                inside = not inside

    return inside


def drawAndFillPolygon(frame, polygon, color, alpha):
    """
    在画面上绘制半透明填充多边形

    Args:
        frame: 输入图像
        polygon: 多边形顶点列表 [(x1,y1), (x2,y2), ...]
        color: 填充颜色 (B, G, R)
        alpha: 透明度 (0~1)，越大越不透明

    Returns:
        绘制后的图像
    """
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32)

    # 填充多边形
    cv2.fillPoly(overlay, [pts], color)

    # 绘制多边形边框（更亮更醒目）
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)

    # 半透明叠加
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return result


def adaptive_resize(frame, max_width):
    """按最大宽度等比例缩放"""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame


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

    print(f"视频分辨率: {frame_width} x {frame_height}")
    print(f"按 'q' 退出，按 'r' 重置入侵记录")

    # ========== 定义敏感区域多边形 ==========
    # 使用相对坐标（比例），以适应不同分辨率
    # 四边形：左下、右下、右上、左上（顺时针或逆时针均可）
    # 以下区域覆盖画面中间偏下部分的车道区域
    ZONE_POLYGON = [
        (int(frame_width * 0.15), int(frame_height * 0.95)),   # 左下
        (int(frame_width * 0.85), int(frame_height * 0.95)),   # 右下
        (int(frame_width * 0.75), int(frame_height * 0.40)),   # 右上
        (int(frame_width * 0.25), int(frame_height * 0.40)),   # 左上
    ]

    # 如果希望将区域限制在更小的范围（如右侧车道），可改用以下配置：
    # ZONE_POLYGON = [
    #     (int(frame_width * 0.55), int(frame_height * 0.95)),
    #     (int(frame_width * 0.95), int(frame_height * 0.95)),
    #     (int(frame_width * 0.85), int(frame_height * 0.35)),
    #     (int(frame_width * 0.50), int(frame_height * 0.35)),
    # ]

    print(f"敏感区域顶点: {ZONE_POLYGON}")

    # 创建截图保存目录
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # 计算显示窗口大小
    display_width = DISPLAY_WIDTH
    display_height = int(frame_height * (display_width / frame_width))

    # 设置窗口
    cv2.namedWindow('T-6: Sensitive Zone Monitoring', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('T-6: Sensitive Zone Monitoring', display_width, display_height)
    cv2.moveWindow('T-6: Sensitive Zone Monitoring', 100, 50)

    # 视频写入器
    out = None
    if SAVE_VIDEO:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (frame_width, frame_height)
        )
        print(f"结果视频将保存到: {output_path}")

    # ========== 入侵状态管理 ==========
    # intrusion_state[track_id] = {
    #     'entered_frame': frame_id,       # 进入区域的帧号
    #     'last_alert_frame': frame_id,    # 上次触发警告的帧号
    #     'screenshots_taken': [],         # 已保存的截图文件名列表
    #     'total_frames_inside': 0,        # 在区域内的总帧数
    #     'label': 'car',                  # 目标类别名称
    # }
    intrusion_state = {}
    # 记录每个 track_id 对应的类别名称（用于最终统计）
    track_labels = {}
    # 当前帧在区域内的目标ID集合（用于跟踪状态变化）
    inside_ids = set()

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 执行追踪（画轨迹以便观察路径）
        annotated_frame, pred_boxes = tracker.track(
            frame, draw_trail=True, analyze_lane_change=False
        )

        # 绘制半透明敏感区域
        annotated_frame = drawAndFillPolygon(
            annotated_frame, ZONE_POLYGON, ZONE_COLOR, ZONE_ALPHA
        )

        # 添加区域标签
        zone_center_x = int(sum(p[0] for p in ZONE_POLYGON) / len(ZONE_POLYGON))
        zone_center_y = int(sum(p[1] for p in ZONE_POLYGON) / len(ZONE_POLYGON))
        cv2.putText(
            annotated_frame, "SENSITIVE ZONE",
            (zone_center_x - 70, zone_center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2
        )

        # ===== 入侵检测逻辑 =====
        current_inside_ids = set()

        for box in pred_boxes:
            x1, y1, x2, y2, lbl, track_id = box

            # 使用底部中心点作为检测点（脚的位置比中心点更准确）
            # PPT 进阶练习：从中心点改为底部中心
            check_point = Point(x=(x1 + x2) / 2, y=y2)

            # 判断是否在敏感区域内
            if isInsidePolygon(check_point, ZONE_POLYGON):
                current_inside_ids.add(track_id)

                # 初始化状态
                if track_id not in intrusion_state:
                    intrusion_state[track_id] = {
                        'entered_frame': frame_id,
                        'last_alert_frame': frame_id,
                        'screenshots_taken': [],
                        'total_frames_inside': 0,
                    }
                    # 记录类别名称
                track_labels[track_id] = lbl

                print(f"[入侵] Track {track_id} ({lbl}) 进入敏感区域 (Frame {frame_id})")

                # 更新状态
                state = intrusion_state[track_id]
                state['total_frames_inside'] += 1
                state['last_alert_frame'] = frame_id

                # 计算持续时间
                duration_frames = frame_id - state['entered_frame']
                duration_sec = duration_frames / fps

                # ===== 入侵显示 =====
                # 框变为红色
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 0, 255), 3
                )

                # 在框上方显示入侵警告
                alert_text = f"INTRUSION! {lbl}({track_id})"
                cv2.putText(
                    annotated_frame, alert_text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

                # 在框下方显示持续时间
                time_text = f"Time: {duration_sec:.1f}s"
                cv2.putText(
                    annotated_frame, time_text,
                    (int(x1), int(y2) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

                # ===== 自动截图保存 =====
                # 只在目标第一次进入区域时截图，避免重复保存
                if len(state['screenshots_taken']) == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_name = f"intrusion_{track_id}_{lbl}_frame{frame_id}_{timestamp}.jpg"
                    screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_name)

                    # 在截图上额外标注信息
                    screenshot = annotated_frame.copy()
                    cv2.putText(
                        screenshot, f"INTRUSION - Track {track_id} ({lbl})",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                    )
                    cv2.putText(
                        screenshot, f"Frame: {frame_id} | Time: {duration_sec:.1f}s",
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                    cv2.imwrite(screenshot_path, screenshot)
                    state['screenshots_taken'].append(screenshot_name)
                    print(f"[截图] Track {track_id} 入侵截图已保存: {screenshot_name}")

        # ===== 统计信息（画面左上角） =====
        active_intrusions = len(current_inside_ids)
        total_intrusions = len(intrusion_state)

        # 左上角显示统计
        cv2.putText(
            annotated_frame,
            f"Intruding Now: {active_intrusions}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255) if active_intrusions > 0 else (0, 255, 0), 2
        )
        cv2.putText(
            annotated_frame,
            f"Total Intrusions: {total_intrusions}",
            (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        # 显示帧号
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_id}",
            (10, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # 右下角显示操作提示
        cv2.putText(
            annotated_frame,
            "'q'=Quit",
            (frame_width - 80, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # 显示画面
        display_frame = adaptive_resize(annotated_frame, DISPLAY_WIDTH)
        cv2.imshow('T-6: Sensitive Zone Monitoring', display_frame)

        # 保存视频
        if out:
            out.write(annotated_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # q 或 ESC → 退出
            break
        elif key == ord('r'):               # r → 重置入侵记录
            intrusion_state.clear()
            track_labels.clear()
            inside_ids.clear()
            print("[重置] 入侵记录已清空")

    # ========== 打印最终统计结果 ==========
    print("\n" + "=" * 55)
    print("敏感区域监控结果统计")
    print("=" * 55)
    print(f"总处理帧数: {frame_id}")
    print(f"总入侵目标数: {len(intrusion_state)}")

    if intrusion_state:
        print("-" * 55)
        print(f"{'Track ID':<10} {'类别':<10} {'持续帧数':<10} {'截图数':<10}")
        print("-" * 55)
        for tid, state in intrusion_state.items():
            duration = frame_id - state['entered_frame']
            screenshot_count = len(state['screenshots_taken'])
            label = track_labels.get(tid, "unknown")
            print(f"{tid:<10} {label:<10} {duration:<10} {screenshot_count:<10}")

    print("=" * 55)
    print(f"截图保存目录: {SCREENSHOT_DIR}")
    print("T-6 完成")

    # 释放资源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
