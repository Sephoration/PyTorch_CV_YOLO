"""
手部关键点检测 - JSON到TXT转换验证脚本
功能：验证标注数据从JSON格式转换到YOLO TXT格式的正确性
验证内容包括：边界框坐标、关键点坐标和可见性
"""


import json
import os

def verify_conversion(json_path, txt_path):
    """
    验证JSON到TXT的转换是否正确
    """
    print(f"\n{'='*60}")
    print(f"验证转换: {os.path.basename(json_path)} -> {os.path.basename(txt_path)}")
    print('='*60)
    
    # 1. 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 2. 读取TXT文件
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_line = f.readline().strip()
    
    # 3. 解析TXT数据
    txt_parts = txt_line.split()
    if len(txt_parts) < 5:
        print(f"错误: TXT文件格式不正确，至少需要5个值，实际有{len(txt_parts)}个")
        return False
    
    # 类别ID应该是整数
    try:
        class_id = int(txt_parts[0])
        print(f"✓ 类别ID: {class_id} (应为整数)")
    except ValueError:
        print(f"✗ 类别ID不是整数: {txt_parts[0]}")
        return False
    
    # 4. 验证图像尺寸
    img_width = json_data['imageWidth']
    img_height = json_data['imageHeight']
    print(f"✓ 图像尺寸: {img_width} x {img_height}")
    
    # 5. 验证边界框
    bbox_norm = list(map(float, txt_parts[1:5]))
    print(f"\n边界框归一化值: {bbox_norm}")
    
    # 在JSON中查找边界框
    for shape in json_data['shapes']:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 确保x1<x2, y1<y2
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # 计算实际边界框
            x_center_actual = (x_min + x_max) / 2 / img_width
            y_center_actual = (y_min + y_max) / 2 / img_height
            width_actual = (x_max - x_min) / img_width
            height_actual = (y_max - y_min) / img_height
            
            print(f"实际边界框:")
            print(f"  x_center: {x_min:.2f} - {x_max:.2f} -> {x_center_actual:.6f}")
            print(f"  y_center: {y_min:.2f} - {y_max:.2f} -> {y_center_actual:.6f}")
            print(f"  width:    {x_max-x_min:.2f} -> {width_actual:.6f}")
            print(f"  height:   {y_max-y_min:.2f} -> {height_actual:.6f}")
            
            # 计算误差
            error_x = abs(bbox_norm[0] - x_center_actual)
            error_y = abs(bbox_norm[1] - y_center_actual)
            error_w = abs(bbox_norm[2] - width_actual)
            error_h = abs(bbox_norm[3] - height_actual)
            
            if error_x < 0.001 and error_y < 0.001 and error_w < 0.001 and error_h < 0.001:
                print("✓ 边界框转换正确")
            else:
                print(f"✗ 边界框转换误差: x:{error_x:.6f}, y:{error_y:.6f}, w:{error_w:.6f}, h:{error_h:.6f}")
            break
    
    # 6. 验证关键点
    keypoint_map = {
        'Mcp_1': 0,
        'Mcp_2': 1,
        'Mcp_3': 2,
        'Mcp_4': 3
    }
    
    num_keypoints = len(keypoint_map)
    expected_txt_length = 5 + num_keypoints * 3  # bbox + keypoints
    
    if len(txt_parts) != expected_txt_length:
        print(f"✗ TXT长度错误: 期望{expected_txt_length}个值，实际{len(txt_parts)}个")
        return False
    
    print(f"\n✓ TXT格式正确: {len(txt_parts)}个值 ({expected_txt_length}期望)")
    
    # 提取关键点数据
    keypoint_data = {}
    json_keypoints = {}
    
    # 从JSON提取关键点
    for shape in json_data['shapes']:
        if shape['shape_type'] == 'point':
            label = shape['label']
            if label in keypoint_map:
                x, y = shape['points'][0]
                x_norm = x / img_width
                y_norm = y / img_height
                json_keypoints[label] = (x_norm, y_norm)
    
    # 从TXT提取关键点
    txt_keypoints = []
    for i in range(num_keypoints):
        base_idx = 5 + i * 3
        kp_x = float(txt_parts[base_idx])
        kp_y = float(txt_parts[base_idx + 1])
        kp_v = float(txt_parts[base_idx + 2])
        txt_keypoints.append((kp_x, kp_y, kp_v))
    
    print("\n关键点验证:")
    for label, idx in sorted(keypoint_map.items(), key=lambda x: x[1]):
        if label in json_keypoints:
            json_x, json_y = json_keypoints[label]
            txt_x, txt_y, txt_v = txt_keypoints[idx]
            
            error_x = abs(json_x - txt_x)
            error_y = abs(json_y - txt_y)
            
            print(f"\n{label} (索引 {idx}):")
            print(f"  JSON: ({json_x:.6f}, {json_y:.6f})")
            print(f"  TXT:  ({txt_x:.6f}, {txt_y:.6f}, v={txt_v})")
            
            if error_x < 0.001 and error_y < 0.001:
                print(f"  ✓ 位置正确 (误差: x={error_x:.6f}, y={error_y:.6f})")
            else:
                print(f"  ✗ 位置误差较大 (误差: x={error_x:.6f}, y={error_y:.6f})")
            
            # 检查可见性
            if txt_v == 2.0:
                print(f"  ✓ 可见性正确 (v=2表示可见)")
            else:
                print(f"  ✗ 可见性错误 (应为2，实际为{txt_v})")
        else:
            print(f"\n{label}: ✗ 在JSON中未找到")
    
    return True

def batch_verify():
    """
    批量验证所有转换结果
    """
    labels_dir = os.path.join('data', 'labels')
    yolo_dir = os.path.join('data', 'yolo')
    
    if not os.path.exists(labels_dir):
        print(f"错误: 找不到labels目录: {labels_dir}")
        return
    
    if not os.path.exists(yolo_dir):
        print(f"错误: 找不到yolo目录: {yolo_dir}")
        return
    
    # 获取所有JSON文件
    import glob
    json_files = glob.glob(os.path.join(labels_dir, '*.json'))
    
    if not json_files:
        print(f"在 {labels_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件，开始验证...")
    
    verified_count = 0
    total_count = len(json_files)
    
    for json_file in json_files:
        base_name = os.path.basename(json_file).replace('.json', '.txt')
        txt_file = os.path.join(yolo_dir, base_name)
        
        if not os.path.exists(txt_file):
            print(f"\n✗ {os.path.basename(json_file)}: 对应的TXT文件不存在")
            continue
        
        if verify_conversion(json_file, txt_file):
            verified_count += 1
    
    print(f"\n{'='*60}")
    print(f"验证完成: {verified_count}/{total_count} 个文件通过验证")
    print('='*60)

if __name__ == '__main__':
    # 首先检查单个文件
    if os.path.exists('IMG_00024482.json') and os.path.exists('IMG_00024482.txt'):
        verify_conversion('IMG_00024482.json', 'IMG_00024482.txt')
    
    # 然后批量验证
    batch_verify()