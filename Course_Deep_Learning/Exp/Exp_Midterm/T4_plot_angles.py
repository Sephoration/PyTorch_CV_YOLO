# T4_plot_angles.py
"""
读取CSV，绘制指定车辆的角度变化折线图
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from yolo_tracker_base import OUTPUT_DIR

# ========== 配置 ==========
CSV_NAME = "LC_records.csv"
TRACK_IDS = [1, 2, 3]
SHOW_MARKER = True
SAVE_FIG = True
SAVE_NAME = "angle_change_chart.png"
# ==========================

def main():
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    if not os.path.exists(csv_path):
        alt = os.path.join(os.path.dirname(OUTPUT_DIR), CSV_NAME)
        csv_path = alt if os.path.exists(alt) else None
    if csv_path is None:
        print(f"文件不存在: {CSV_NAME}。请先运行 p1_SuspectedLaneChange.py")
        return

    print(f"读取: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"记录数: {len(df)}, track_id: {sorted(df['track_id'].unique())}")

    df = df.sort_values(['track_id', 'frame_id']).reset_index(drop=True)
    df_filtered = df[df['track_id'].isin(TRACK_IDS)].copy()
    if df_filtered.empty:
        print(f"指定的 track_id {TRACK_IDS} 不在 CSV 中")
        print(f"可用的: {sorted(df['track_id'].unique())}")
        return

    df_filtered['norm_frame'] = df_filtered.groupby('track_id')['frame_id'].transform(
        lambda x: x - x.min())

    plt.figure(figsize=(14, 8))
    for tid in TRACK_IDS:
        sub = df_filtered[df_filtered['track_id'] == tid]
        if sub.empty:
            continue
        plt.plot(sub['norm_frame'], sub['angle'],
                 marker='o' if SHOW_MARKER else None, markersize=2, linewidth=1.5,
                 label=f'Track ID {tid}')

    plt.title('Vehicle Direction Angle Over Time (Normalized Frames)')
    plt.xlabel('Normalized Frame ID')
    plt.ylabel('Angle (degrees)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_FIG:
        save_path = os.path.join(OUTPUT_DIR, SAVE_NAME)
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    plt.show()

    print("\n=== 统计 ===")
    for tid in TRACK_IDS:
        sub = df_filtered[df_filtered['track_id'] == tid]
        if not sub.empty:
            angles = sub['angle'].dropna()
            if len(angles) > 0:
                has_lane_ok = 'lane_ok' in sub.columns
                lane_true = sub['lane_ok'].sum() if has_lane_ok else 'N/A'
                print(f"Track {tid}: {len(angles)} 帧, "
                      f"角度范围 [{angles.min():.0f}, {angles.max():.0f}], "
                      f"变化量 {angles.max()-angles.min():.0f}°, "
                      f"lane_ok={lane_true}")

if __name__ == "__main__":
    main()
