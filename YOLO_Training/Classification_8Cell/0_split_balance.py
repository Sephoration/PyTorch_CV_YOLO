"""
处理数据集 (8个类别 × 1500张)
训练集总计: 9600 张
验证集总计: 2400 张
"""

import os
import pathlib
import shutil
import cv2
import random
from sklearn.model_selection import train_test_split
import albumentations as A

# 路径配置
src = pathlib.Path('data/datasets')  # 原数据集路径（只读）
dst = pathlib.Path('data/datasets8')  # 新数据集路径（所有操作在这里进行）

# 数据集参数配置
TRAIN_COUNT = 1200  # 每个类别训练集图片数量
VAL_COUNT = 300  # 每个类别验证集图片数量
TOTAL_COUNT = TRAIN_COUNT + VAL_COUNT  # 每个类别总共1500张图片

# 数据增强配置
AUG = A.Compose([
    A.HorizontalFlip(p=0.5),  # 水平翻转，概率50%
    A.RandomRotate90(p=0.5),  # 随机旋转90度，概率50%
    A.RandomBrightnessContrast(p=0.3)  # 随机亮度对比度调整，概率30%
])


def augment_images(original_files, target_count, class_name):
    """
    对原始图片进行数据增强，生成目标数量的图片
    """
    if len(original_files) >= target_count:
        return original_files[:target_count]

    print(f"  需要增强: {len(original_files)} -> {target_count} 张")

    times, rem = divmod(target_count - len(original_files), len(original_files))
    augmented_files = []
    augmentation_count = 0

    for i, original_file in enumerate(original_files):
        img = cv2.imread(str(original_file))
        if img is None:
            print(f"  警告: 无法读取图片 {original_file}，跳过")
            continue

        aug_needed = times + (1 if i < rem else 0)

        for k in range(aug_needed):
            augmented = AUG(image=img)
            aug_filename = f"{original_file.stem}_aug{augmentation_count}{original_file.suffix}"
            aug_filepath = dst / 'temp_aug' / class_name / aug_filename

            aug_filepath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(aug_filepath), augmented['image'])
            augmented_files.append(aug_filepath)
            augmentation_count += 1

    return original_files + augmented_files


def copy_files_with_retry(file_list, target_dir, file_type):
    """
    复制文件并重试机制，确保文件确实被复制
    """
    success_count = 0
    for i, file_path in enumerate(file_list):
        try:
            # 生成目标文件名
            if hasattr(file_path, 'suffix'):
                # 如果是路径对象
                new_filename = f"{file_type}_{i:05d}{file_path.suffix}"
            else:
                # 如果是字符串路径
                file_path = pathlib.Path(file_path)
                new_filename = f"{file_type}_{i:05d}{file_path.suffix}"

            target_path = target_dir / new_filename

            # 复制文件
            shutil.copy2(file_path, target_path)

            # 验证文件是否确实存在
            if target_path.exists():
                success_count += 1
            else:
                print(f"  警告: 文件复制后不存在 {target_path}")

        except Exception as e:
            print(f"  复制文件失败 {file_path}: {e}")

    return success_count


def main():
    """主函数：执行数据集平衡和划分"""

    # 确保目标目录存在
    dst.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {dst}")

    # 清理之前的临时目录和目标目录
    temp_dir = dst / 'temp_aug'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # 清理之前的输出目录
    for subdir in ['train', 'val']:
        output_dir = dst / subdir
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历原数据集中的每个类别文件夹
    for cls_dir in src.iterdir():
        if not cls_dir.is_dir():
            continue

        print(f"\n正在处理类别: {cls_dir.name}")

        # 获取所有图片文件
        files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in image_extensions:
            files.extend(cls_dir.glob(f'*{ext}'))
            files.extend(cls_dir.glob(f'*{ext.upper()}'))

        print(f"  在原文件夹中找到 {len(files)} 张图片")

        if not files:
            print(f"  警告: {cls_dir.name} 中没有找到图片文件，跳过该类别")
            continue

        # 如果原图数量超过目标数量，随机选择TOTAL_COUNT张
        if len(files) > TOTAL_COUNT:
            files = random.sample(files, TOTAL_COUNT)
            print(f"  随机选择 {TOTAL_COUNT} 张原图")

        # 数据增强：确保总图片数量达到TOTAL_COUNT
        all_files = augment_images(files, TOTAL_COUNT, cls_dir.name)
        print(f"  增强后共有 {len(all_files)} 张图片")

        # 划分训练集和验证集 (1200:300)
        train_files, val_files = train_test_split(
            all_files,
            train_size=TRAIN_COUNT,
            test_size=VAL_COUNT,
            random_state=42
        )

        # 创建目标目录
        train_dir = dst / 'train' / cls_dir.name
        val_dir = dst / 'val' / cls_dir.name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        print(f"  开始复制文件...")

        # 复制训练集图片
        train_count = copy_files_with_retry(train_files, train_dir, 'train')

        # 复制验证集图片
        val_count = copy_files_with_retry(val_files, val_dir, 'val')

        print(f"  复制完成: 训练集 {train_count} 张, 验证集 {val_count} 张")

        # 立即验证当前类别的文件数量
        actual_train = len(list(train_dir.iterdir()))
        actual_val = len(list(val_dir.iterdir()))

        if actual_train == TRAIN_COUNT and actual_val == VAL_COUNT:
            print(f"  ✅ {cls_dir.name} 类别文件数量正确")
        else:
            print(
                f"  ⚠️  {cls_dir.name} 类别文件数量不正确: 训练集{actual_train}/{TRAIN_COUNT}, 验证集{actual_val}/{VAL_COUNT}")

    # 清理临时目录
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def print_final_statistics():
    """打印最终的数据集统计信息"""
    print('\n' + '=' * 50)
    print('最终统计结果:')
    print('=' * 50)

    train_dir = dst / 'train'
    val_dir = dst / 'val'

    total_train, total_val = 0, 0
    all_correct = True

    # 统计训练集
    print("\n训练集分布:")
    for cls_dir in sorted(train_dir.iterdir()):
        if cls_dir.is_dir():
            count = len([f for f in cls_dir.iterdir() if f.is_file()])
            total_train += count
            status = "✅" if count == TRAIN_COUNT else "❌"
            print(f"  {status} {cls_dir.name}: {count} 张 (目标: {TRAIN_COUNT})")
            if count != TRAIN_COUNT:
                all_correct = False

    # 统计验证集
    print("\n验证集分布:")
    for cls_dir in sorted(val_dir.iterdir()):
        if cls_dir.is_dir():
            count = len([f for f in cls_dir.iterdir() if f.is_file()])
            total_val += count
            status = "✅" if count == VAL_COUNT else "❌"
            print(f"  {status} {cls_dir.name}: {count} 张 (目标: {VAL_COUNT})")
            if count != VAL_COUNT:
                all_correct = False

    # 汇总信息
    print('\n' + '-' * 50)
    print(f"训练集总计: {total_train} 张")
    print(f"验证集总计: {total_val} 张")
    print(f"数据集总计: {total_train + total_val} 张")
    print(f"期望总计: {8 * TOTAL_COUNT} 张 (8个类别 × {TOTAL_COUNT}张)")

    if all_correct:
        print("🎉 所有类别都达到了目标数量！")
    else:
        print("❌ 部分类别未达到目标数量")


if __name__ == "__main__":
    print("开始处理数据集...")
    print("=" * 60)
    print(f"源目录: {src} (只读)")
    print(f"目标目录: {dst}")
    print(f"目标数量: 每类 {TRAIN_COUNT}训练 + {VAL_COUNT}验证 = {TOTAL_COUNT}张")
    print("=" * 60)

    # 执行主处理流程
    main()

    # 打印最终统计
    print_final_statistics()

    print(f"\n✅ 数据集处理完成！")
    print(f"📁 结果保存在: {dst}")
    print("🔒 原数据集未被修改")