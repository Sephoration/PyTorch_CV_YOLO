import os
import pandas as pd

# ==============================
# 修改 Pandas 默认显示格式，方便查看
# ==============================
pd.set_option("display.width", 5000)
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 500)
pd.set_option("expand_frame_repr", False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "csv", "yoga_pose_train.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "csv", "yoga_pose_test.csv")


if __name__ == '__main__':
    print("=" * 60)
    print("读取训练数据集")
    df_train = pd.read_csv(TRAIN_CSV_PATH)

    print("\n" + "=" * 60)
    print("读取测试数据集")
    df_test = pd.read_csv(TEST_CSV_PATH)

    # 查看前几笔数据
    print("\n" + "=" * 60)
    print("训练集前 5 笔数据：")
    print(df_train.head())

    # 数据集大小
    print("\n" + "=" * 60)
    print("训练集信息：")
    print(df_train.info(verbose=True))

    print("\n" + "=" * 60)
    print("测试集信息：")
    print(df_test.info(verbose=True))

    # 查看标签
    print("\n" + "=" * 60)
    print("训练集标签种类：")
    print(df_train["class_name"].unique())

    print("\n" + "=" * 60)
    print("训练集标签分布：")
    print(df_train["class_name"].value_counts())

    print("\n" + "=" * 60)
    print("测试集标签分布：")
    print(df_test["class_name"].value_counts())

    # 检查缺失值
    print("\n" + "=" * 60)
    print("训练集缺失值总数：", df_train.isnull().sum().sum())
    print("测试集缺失值总数：", df_test.isnull().sum().sum())
