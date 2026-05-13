# T4_plot_angles.py

"""
读取CSV文件，绘制指定车辆的角度变化折线图
用于T-4课后分析
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 从基础类导入路径配置
from yolo_tracker_base import OUTPUT_DIR, PROJECT_ROOT

# ========== 配置 ==========
CSV_NAME = "analysis_records.csv"

# 要观察的track_id（跑完T4后看CSV，填实际出现的ID）
TRACK_IDS = [1, 2, 3]

SHOW_MARKER = True
SAVE_FIG = True
SAVE_NAME = "angle_change_chart.png"
# ==========================

def find_csv_file():
    """查找CSV文件"""
    possible_paths = [
        os.path.join(OUTPUT_DIR, CSV_NAME),
        os.path.join(PROJECT_ROOT, CSV_NAME),
        CSV_NAME,
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    csv_path = find_csv_file()
    
    if csv_path is None:
        print(f"文件不存在: {CSV_NAME}")
        print(f"搜索路径:")
        print(f"  - {os.path.join(OUTPUT_DIR, CSV_NAME)}")
        print(f"  - {os.path.join(PROJECT_ROOT, CSV_NAME)}")
        print(f"  - {CSV_NAME}")
        print("\n请先运行 p1_SuspectedLaneChange.py 生成CSV文件")
        return
    
    print(f"读取CSV文件: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"读取到 {len(df)} 条记录")
    print(f"CSV中的track_id有: {sorted(df['track_id'].unique())}")
    
    df = df.sort_values(['track_id', 'frame_id']).reset_index(drop=True)
    
    df_filtered = df[df['track_id'].isin(TRACK_IDS)].copy()
    
    if df_filtered.empty:
        print(f"指定的track_id {TRACK_IDS} 在CSV中不存在")
        print(f"可用的track_id: {sorted(df['track_id'].unique())}")
        return
    
    df_filtered['frame_id_normalized'] = df_filtered.groupby('track_id')['frame_id'].transform(lambda x: x - x.min())
    
    plt.figure(figsize=(14, 8))
    
    for tid in TRACK_IDS:
        df_tid = df_filtered[df_filtered['track_id'] == tid].copy()
        if df_tid.empty:
            continue
        
        if SHOW_MARKER:
            plt.plot(df_tid['frame_id_normalized'], df_tid['angle'],
                    marker='o', markersize=2, linewidth=1.5,
                    label=f'Track ID {tid}')
        else:
            plt.plot(df_tid['frame_id_normalized'], df_tid['angle'],
                    linewidth=1.5, label=f'Track ID {tid}')
    
    plt.title('Vehicle Direction Angle Change Over Time (Normalized)')
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
    
    print("\n=== 角度变化统计 ===")
    for tid in TRACK_IDS:
        df_tid = df_filtered[df_filtered['track_id'] == tid]
        if not df_tid.empty:
            angles = df_tid['angle'].dropna()
            if len(angles) > 0:
                print(f"Track ID {tid}: 记录数={len(angles)}, "
                      f"角度范围=[{angles.min():.1f}, {angles.max():.1f}], "
                      f"变化量={angles.max() - angles.min():.1f}度")

if __name__ == "__main__":
    main()