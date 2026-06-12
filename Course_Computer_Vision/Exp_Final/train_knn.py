import os
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "csv", "hand_gesture_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_gesture_knn.pkl")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("加载数据集...")
    df = pd.read_csv(CSV_PATH)

    y = df["class_name"]
    X = df.drop(["class_name", "file_name"], axis=1)

    print(f"样本总数：{len(X)}")
    print(f"特征维度：{X.shape[1]}")
    print(f"类别数：{len(y.unique())}  类别：{sorted(y.unique())}")
    print(f"各类样本数：\n{y.value_counts().sort_index()}")

    print("\n" + "=" * 60)
    print("训练 KNN 分类器...")
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X, y)

    cv_scores = cross_val_score(knn, X, y, cv=5)
    print(f"5折交叉验证平均准确率：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    y_pred = knn.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n训练集准确率：{acc:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    print(f"模型已保存：{MODEL_PATH}")

    print("\n" + "=" * 60)
    print("分类报告：")
    print(classification_report(y, y_pred))

    print("混淆矩阵：")
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    print("     ", "  ".join(f"{l:>4}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i]}  ", "  ".join(f"{v:>4}" for v in row))

    print("\n" + "=" * 60)
    print("尝试训练 SVM 对比...")
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X, y)
    svm_acc = accuracy_score(y, svm.predict(X))
    svm_cv = cross_val_score(svm, X, y, cv=5).mean()
    print(f"SVM 训练集准确率：{svm_acc:.4f}")
    print(f"SVM 5折交叉验证：{svm_cv:.4f}")

    if svm_cv > cv_scores.mean():
        svm_path = os.path.join(MODEL_DIR, "hand_gesture_svm.pkl")
        with open(svm_path, "wb") as f:
            pickle.dump(svm, f)
        print(f"SVM 表现更好，已保存：{svm_path}")
    else:
        print("KNN 表现更好，保留 KNN 模型")


if __name__ == '__main__':
    main()
