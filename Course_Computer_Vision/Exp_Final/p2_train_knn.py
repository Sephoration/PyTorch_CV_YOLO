import os
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 六个模型的保存路径
MODELS = {
    "hand_gesture_knn.pkl":          ("csv", "hand_gesture_data.csv",          "knn"),
    "hand_gesture_svm.pkl":          ("csv", "hand_gesture_data.csv",          "svm"),
    "hand_gesture_knn_hand.pkl":     ("csv", "hand_gesture_data_hand.csv",     "knn"),
    "hand_gesture_svm_hand.pkl":     ("csv", "hand_gesture_data_hand.csv",     "svm"),
    "hand_gesture_knn_hand_v2.pkl":  ("csv", "hand_gesture_data_hand_v2.csv",  "knn"),
    "hand_gesture_svm_hand_v2.pkl":  ("csv", "hand_gesture_data_hand_v2.csv",  "svm"),
}


def load_data(csv_name):
    csv_path = os.path.join(BASE_DIR, "csv", csv_name)
    df = pd.read_csv(csv_path)
    y = df["class_name"]
    X = df.drop(["class_name", "file_name"], axis=1)
    return X, y


def train_and_save(X, y, model_name, algo):
    print(f"\n{'=' * 60}")
    print(f"训练 {model_name} ({algo.upper()})")
    print(f"样本数：{len(X)}  特征维度：{X.shape[1]}  类别数：{len(y.unique())}")

    if algo == "knn":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ])
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel="rbf", C=10, gamma="scale", probability=True))
        ])

    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5折交叉验证平均准确率：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"训练集准确率：{acc:.4f}")

    model_path = os.path.join(MODEL_DIR, model_name)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存：{model_path}")

    print("\n分类报告：")
    print(classification_report(y, y_pred))
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    header = "     " + "".join(f"{l:>4}" for l in labels)
    print(f"\n混淆矩阵（行列对应 labels={labels}）：")
    print(header)
    for i, row in enumerate(cm):
        print(f"真{i}  " + "".join(f"{v:4}" for v in row))

    return cv_scores.mean()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_scores = []

    for model_name, (csv_dir, csv_name, algo) in MODELS.items():
        X, y = load_data(csv_name)
        score = train_and_save(X, y, model_name, algo)
        all_scores.append((model_name, score))

    print(f"\n{'=' * 60}")
    print("模型对比（5折交叉验证平均准确率）：")
    for name, score in all_scores:
        print(f"  {name:<35} {score:.4f}")


if __name__ == '__main__':
    main()
