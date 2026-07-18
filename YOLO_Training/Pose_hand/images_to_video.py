import os
import random
import cv2
import numpy as np


def create_video_from_folder(folder_path, output_video="random_images_video.mp4", duration_per_image=2):
    """
    从单个文件夹读取所有图片，随机打乱顺序，合成视频
    
    参数:
    folder_path: 包含图片的文件夹路径
    output_video: 输出视频文件名
    duration_per_image: 每张图片显示的时长(秒)
    """
    # 参数设置
    fps = 1 / duration_per_image  # 例如 2 秒一帧 -> 0.5 fps

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在: {folder_path}")
        return

    # 获取文件夹名称
    folder_name = os.path.basename(folder_path)
    
    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"错误: 在 {folder_name} 中没有找到任何图片！")
        return

    # 创建完整的图片路径列表
    all_images = [os.path.join(folder_path, img_file) for img_file in image_files]
    
    # 随机打乱图片顺序
    random.shuffle(all_images)
    
    print(f"总共找到 {len(all_images)} 张图片，已随机打乱顺序")

    # 读取第一张图，获取尺寸
    first_img = cv2.imread(all_images[0])
    if first_img is None:
        print(f"错误: 无法读取第一张图片: {all_images[0]}")
        return
    
    height, width = first_img.shape[:2]
    print(f"图片尺寸: {width} x {height}")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 逐张处理并写入
    for idx, img_path in enumerate(all_images, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 调整尺寸以匹配视频大小
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # 写入视频
        video_writer.write(img)
        print(f"进度: {idx}/{len(all_images)} - {os.path.basename(img_path)}")

    # 释放资源
    video_writer.release()
    
    # 输出视频信息
    video_duration = len(all_images) * duration_per_image
    print(f"\n视频已生成: {output_video}")
    print(f"总共处理了 {len(all_images)} 张图片")
    print(f"视频时长: {video_duration} 秒")
    print(f"帧率: {fps} fps")


# 使用示例
if __name__ == "__main__":
    # 直接使用指定的图片文件夹路径
    folder_path = "videos/frames"
    
    # 可以自定义输出视频文件名，默认为 random_images_video.mp4
    output_video = "my_generated_video.mp4"
    
    # 可以自定义每张图片显示的时长(秒)，默认为 2 秒
    duration_per_image = 2
    
    print(f"使用图片文件夹: {folder_path}")
    print(f"输出视频文件: {output_video}")
    print(f"每张图片显示时长: {duration_per_image} 秒")
    print("开始生成视频...")
    
    # 调用函数创建视频
    create_video_from_folder(
        folder_path=folder_path,
        output_video=output_video,
        duration_per_image=duration_per_image
    )

