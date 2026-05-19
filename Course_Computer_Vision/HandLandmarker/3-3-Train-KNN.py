import os
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

####################### 设定全局常数 #######################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "csv", "yoga_pose_train.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "csv", "yoga_pose_test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "yoga_pose_knn.pkl")
####################### 设定全局常数 #######################


# ==============================
# 读取 CSV，并分成 X / y
# ==============================
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # 标签：类别名称
    y = df["class_name"]

    # 特征：去掉 class_name 和 file_name
    X = df.drop(["class_name", "file_name"], axis=1)

    return X, y


if __name__ == '__main__':
    # 确保模型文件夹存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("读取训练数据集")
    X_train, y_train = load_dataset(TRAIN_CSV_PATH)

    print("读取测试数据集")
    X_test, y_test = load_dataset(TEST_CSV_PATH)

    # 查看训练与测试基本信息
    print("-" * 60)
    print(f"训练样本数：{len(X_train)}")
    print(f"测试样本数：{len(X_test)}")
    print(f"特征维度：{X_train.shape[1]}")
    print(f"类别名称：{sorted(y_train.unique())}")

    # ==============================
    # 建立 KNN 分类器
    # ==============================
    print("-" * 60)
    print("开始训练 KNN 分类器")
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    model.fit(X_train, y_train)

    # 查看训练后的内部状态
    print("-" * 60)
    print("训练完成")
    print(f"实际使用的 algorithm：{model._fit_method}")
    print(f"训练样本数：{model.n_samples_fit_}")
    print(f"类别列表：{model.classes_}")

    # ==============================
    # 保存模型
    # ==============================
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("-" * 60)
    print(f"模型已保存：{MODEL_PATH}")

    # ==============================
    # 测试模型
    # ==============================
    print("=" * 60)
    print("开始测试模型")
    y_pred = model.predict(X_test)

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    print("-" * 60)
    print(f"测试准确率：{acc:.4f}")

    # 分类报告
    print("-" * 60)
    print("分类报告：")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    print("-" * 60)
    print("混淆矩阵：")
    labels = sorted(y_train.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("标签顺序", labels)
    print(cm)
    print("=" * 60)
