import json
import os
import glob

# 关键点映射表，将关键点标签映射到固定顺序的索引
KEYPOINT_MAP = {
    'Mcp_1': 0,
    'Mcp_2': 1,
    'Mcp_3': 2,
    'Mcp_4': 3
}

NUM_KEYPOINTS = len(KEYPOINT_MAP)

def convert_json_to_txt(json_path):
    """
    将单个JSON文件转换为YOLO关键点TXT格式
    Args:
        json_path: JSON文件路径
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    bbox = None
    keypoints = {}
    
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 确保x1<x2, y1<y2
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # 计算边界框中心点和宽高
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # 归一化处理
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height
            
            # 确保数值在合理范围内
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            bbox = [x_center, y_center, width, height]
            
        elif shape['shape_type'] == 'point':
            label = shape['label']
            if label in KEYPOINT_MAP:
                x, y = shape['points'][0]
                # 归一化处理
                x /= img_width
                y /= img_height
                # 确保数值在合理范围内
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                keypoints[label] = [x, y]
    
    if bbox is None:
        print(f"警告：未在 {json_path} 中找到边界框")
        return None
    
    # 按照映射表顺序构建关键点列表，包含可见性
    keypoint_list = []
    for label, index in sorted(KEYPOINT_MAP.items(), key=lambda x: x[1]):
        if label in keypoints:
            x, y = keypoints[label]
            v = 2  # 2表示可见
        else:
            x, y = 0.0, 0.0
            v = 0  # 0表示无效
        keypoint_list.extend([x, y, v])
    
    # 构建YOLO格式行：类别ID x_center y_center width height 关键点1_x 关键点1_y 关键点1_v ...
    # 使用整数类别ID（假设hand类别为0）
    yolo_line = [0] + bbox + keypoint_list
    
    # 转换为字符串，类别ID保持整数，其他保留足够的小数位
    yolo_str = f'{yolo_line[0]:d} ' + ' '.join([f'{num:.6f}' for num in yolo_line[1:]])
    
    return yolo_str

def batch_convert():
    """
    批量转换所有JSON文件
    """
    input_dir = os.path.join(os.getcwd(), 'data', 'labels')
    output_dir = os.path.join(os.getcwd(), 'data', 'yolo')
    
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not json_files:
        print(f"未在 {input_dir} 中找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件，开始转换...")
    
    for json_file in json_files:
        yolo_str = convert_json_to_txt(json_file)
        
        if yolo_str:
            base_name = os.path.basename(json_file).replace('.json', '.txt')
            txt_file = os.path.join(output_dir, base_name)
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(yolo_str + '\n')
            
            print(f"转换完成：{os.path.basename(json_file)} -> {os.path.basename(txt_file)}")
            print(f"  内容：{yolo_str}")
    
    print("所有JSON文件转换完成！")

if __name__ == '__main__':
    batch_convert()