# T4_plot_angles.py (修改后)

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
CSV_NAME = "analysis_records.csv"   # CSV文件名（在OUTPUT_DIR中）

# 要观察的track_id（根据CSV中的数据填写）
TRACK_IDS = [1, 2, 3, 4, 5]   # 请根据实际CSV中的track_id修改

SHOW_MARKER = True      # 是否显示数据点标记
SAVE_FIG = True         # 是否保存图片
SAVE_NAME = "angle_change_chart.png"  # 图片保存文件名
# ==========================

def find_csv_file():
    """查找CSV文件，支持多个可能的位置"""
    possible_paths = [
        os.path.join(OUTPUT_DIR, CSV_NAME),  # 输出目录
        os.path.join(PROJECT_ROOT, CSV_NAME),  # 项目根目录
        CSV_NAME,  # 当前工作目录
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    # 查找CSV文件
    csv_path = find_csv_file()
    
    if csv_path is None:
        print(f"文件不存在: {CSV_NAME}")
        print(f"搜索路径:")
        print(f"  - {os.path.join(OUTPUT_DIR, CSV_NAME)}")
        print(f"  - {os.path.join(PROJECT_ROOT, CSV_NAME)}")
        print(f"  - {CSV_NAME}")
        print("\n请先运行 T4_lane_change.py 生成CSV文件")
        return
    
    print(f"读取CSV文件: {csv_path}")
    
    # 读取CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"读取到 {len(df)} 条记录")
    print(f"CSV中的track_id有: {sorted(df['track_id'].unique())}")
    
    # 按track_id和frame_id排序
    df = df.sort_values(['track_id', 'frame_id']).reset_index(drop=True)
    
    # 筛选指定车辆
    df_filtered = df[df['track_id'].isin(TRACK_IDS)].copy()
    
    if df_filtered.empty:
        print(f"指定的track_id {TRACK_IDS} 在CSV中不存在")
        print(f"可用的track_id: {sorted(df['track_id'].unique())}")
        return
    
    # 归一化frame_id（让每辆车从0开始）
    df_filtered['frame_id_normalized'] = df_filtered.groupby('track_id')['frame_id'].transform(lambda x: x - x.min())
    
    # 创建画布
    plt.figure(figsize=(14, 8))
    
    # 为每辆车绘制角度曲线
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
    
    # 图表设置
    plt.title('Vehicle Direction Angle Change Over Time (Normalized)')
    plt.xlabel('Normalized Frame ID')
    plt.ylabel('Angle (degrees)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if SAVE_FIG:
        save_path = os.path.join(OUTPUT_DIR, SAVE_NAME)
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    
    # 显示
    plt.show()
    
    # 打印统计信息
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