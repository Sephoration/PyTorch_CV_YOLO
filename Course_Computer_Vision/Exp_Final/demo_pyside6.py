import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
KNN_PATH = os.path.join(MODEL_DIR, "hand_gesture_knn.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "hand_gesture_svm.pkl")
PROC_W, PROC_H = 320, 240
LOCK_DURATION = 1.5


class GestureClassifier:
    def __init__(self):
        self.knn = self.svm = None
        self.current = "knn"
        for name, path in [("knn", KNN_PATH), ("svm", SVM_PATH)]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    setattr(self, name, pickle.load(f))

    def normalize(self, landmarks):
        """与 extract_features.py 完全一致的归一化"""
        wrist = landmarks[0]
        mcp = landmarks[9]
        sx = mcp.x - wrist.x; sy = mcp.y - wrist.y; sz = mcp.z - wrist.z
        scale = np.sqrt(sx*sx + sy*sy + sz*sz)
        if scale < 1e-6:
            scale = 1.0
        feats = []
        for lm in landmarks:
            feats.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
        return np.array(feats).reshape(1, -1)

    def predict(self, landmarks):
        model = getattr(self, self.current, None)
        if model is None or landmarks is None:
            return None, None
        feats = self.normalize(landmarks)
        pred = model.predict(feats)[0]
        proba = model.predict_proba(feats)[0] if self.current == "knn" and hasattr(model, "predict_proba") else None
        return int(pred), proba


class InferenceWorker(QThread):
    result = Signal(object, object)  # frame, pred_data | frame, None

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.running = False
        self.frame_skip = 2
        self.fc = 0

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        task_path = os.path.join(MODEL_DIR, "hand_landmarker.task")
        if not os.path.exists(task_path):
            return
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1, min_hand_detection_confidence=0.75,
        )
        landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        conn = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        prev_pred = -1
        lock_start = 0.0
        locked_pred = -1

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            self.fc += 1
            h, w = frame.shape[:2]
            pred_data = None

            if self.fc % (self.frame_skip + 1) == 0:
                small = cv2.resize(frame, (PROC_W, PROC_H))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                ts = int(time.time() * 1000)
                results = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if results and results.hand_landmarks:
                    hl = results.hand_landmarks[0]
                    for c in conn:
                        p0, p1 = hl[c.start], hl[c.end]
                        cv2.line(frame, (int(p0.x*w), int(p0.y*h)), (int(p1.x*w), int(p1.y*h)), (0,255,0), 2)
                    for lm in hl:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,0,255), -1)

                    pred, proba = self.classifier.predict(hl)
                    if pred is not None:
                        now = time.time()
                        # 老师风格的锁定逻辑
                        if pred == prev_pred:
                            if pred != locked_pred:
                                if lock_start == 0:
                                    lock_start = now
                                elif now - lock_start >= LOCK_DURATION:
                                    locked_pred = pred
                        else:
                            lock_start = 0
                            locked_pred = -1
                        prev_pred = pred
                        show = locked_pred if locked_pred >= 0 else pred
                        progress = 0.0
                        if lock_start > 0 and pred == prev_pred and pred != locked_pred:
                            progress = min(1.0, (now - lock_start) / LOCK_DURATION)
                        pred_data = (show, locked_pred >= 0, progress, proba)

            self.result.emit(frame, pred_data)

        cap.release()
        landmarker.close()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("静态手势识别系统 (0-9)")
        self.setFixedSize(960, 680)

        self.classifier = GestureClassifier()
        self.worker = None
        self.last_frame = None
        self.pred_data = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # 标题栏
        title = QLabel("静态手势识别系统   |   基于 MediaPipe + KNN")
        title.setStyleSheet("background:#1a1a22; color:#00c8a0; font-size:16px; font-weight:bold; padding:12px 20px;")
        root.addWidget(title)

        # 主体
        body = QHBoxLayout()
        root.addLayout(body)

        # 左侧：摄像头
        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("background:#000; border:1px solid #333;")
        body.addWidget(self.cam_label)

        # 右侧信息面板
        info = QWidget()
        info.setFixedWidth(300)
        info.setStyleSheet("background:#121216;")
        il = QVBoxLayout(info)
        il.setContentsMargins(16, 20, 16, 20)

        # 识别数字大图
        self.num_label = QLabel("--")
        self.num_label.setStyleSheet("font-size:72px; font-weight:bold; color:#fff; background:#1a1a22; border-radius:12px; padding:20px;")
        self.num_label.setAlignment(Qt.AlignCenter)
        self.num_label.setFixedHeight(140)
        il.addWidget(self.num_label)

        # 置信度
        self.conf_label = QLabel("置信度: --")
        self.conf_label.setStyleSheet("font-size:14px; color:#00c8a0; padding:4px 0;")
        il.addWidget(self.conf_label)

        # 锁定状态
        self.lock_label = QLabel("状态: 等待手势...")
        self.lock_label.setStyleSheet("font-size:13px; color:#aaa; padding:4px 0;")
        il.addWidget(self.lock_label)

        # 进度条
        self.progress_label = QLabel("")
        self.progress_label.setFixedHeight(20)
        self.progress_label.setStyleSheet("background:#2a2a30; border-radius:4px;")
        il.addWidget(self.progress_label)

        il.addSpacing(20)

        # 模型选择
        ml = QHBoxLayout()
        ml.addWidget(QLabel("模型:"))
        ml.itemAt(0).widget().setStyleSheet("color:#aaa; font-size:13px;")
        self.model_box = QComboBox()
        self.model_box.setStyleSheet("background:#2a2a30; color:#fff; font-size:13px; padding:6px 10px; border:1px solid #444; border-radius:4px;")
        if self.classifier.knn: self.model_box.addItem("KNN","knn")
        if self.classifier.svm: self.model_box.addItem("SVM","svm")
        self.model_box.currentIndexChanged.connect(self.on_model)
        ml.addWidget(self.model_box)
        ml.addStretch()
        il.addLayout(ml)

        il.addSpacing(10)

        # 按钮
        self.btn = QPushButton("开启摄像头")
        self.btn.setStyleSheet("QPushButton{background:#00c8a0;color:#fff;font-size:14px;font-weight:bold;padding:10px;border:none;border-radius:6px;}QPushButton:hover{background:#00e0b0;}QPushButton:checked{background:#e05555;}")
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.toggle)
        il.addWidget(self.btn)

        il.addStretch()

        # 底部引导
        guide = QLabel("【手势操作】 保持 1 秒锁定  0-9 数字识别")
        guide.setStyleSheet("background:#1a1a22; color:#666; font-size:12px; padding:8px 20px;")
        root.addWidget(guide)

        body.addWidget(info)

        # 定时刷新UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)

    def toggle(self, checked):
        if checked:
            self.start()
        else:
            self.stop()

    def start(self):
        if self.worker: return
        self.worker = InferenceWorker(self.classifier)
        self.worker.result.connect(self.on_result)
        self.worker.start()
        self.btn.setText("关闭摄像头")

    def stop(self):
        if self.worker:
            self.worker.stop(); self.worker = None
        self.cam_label.clear()
        self.btn.setText("开启摄像头")
        self.num_label.setText("--")
        self.conf_label.setText("置信度: --")
        self.lock_label.setText("状态: 等待手势...")
        self.progress_label.setText("")

    def on_result(self, frame, pred_data):
        self.last_frame = frame
        self.pred_data = pred_data

    def refresh(self):
        # 更新画面
        if self.last_frame is not None:
            rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.cam_label.setPixmap(QPixmap.fromImage(qimg))

        # 更新识别结果
        if self.pred_data is not None:
            pred, locked, progress, proba = self.pred_data
            if pred is not None:
                self.num_label.setText(str(pred))
                if proba is not None:
                    conf = float(np.max(proba))
                    self.conf_label.setText(f"置信度: {conf:.1%}")
                else:
                    self.conf_label.setText("置信度: --")
                if locked:
                    self.lock_label.setText(f"✅ 已锁定: 手势 {pred}")
                    self.lock_label.setStyleSheet("font-size:13px; color:#00ff88; padding:4px 0;")
                elif progress > 0:
                    self.lock_label.setText(f"⏳ 锁定中 {int(progress*100)}%")
                    self.lock_label.setStyleSheet("font-size:13px; color:#ffcc00; padding:4px 0;")
                else:
                    self.lock_label.setText(f"👆 检测到手势 {pred}，保持锁定")
                    self.lock_label.setStyleSheet("font-size:13px; color:#aaa; padding:4px 0;")

                # 进度条
                bar_w = int(280 * progress)
                self.progress_label.setText(f"{'█' * (bar_w // 10)}{'░' * (28 - bar_w // 10)}")
            else:
                self.num_label.setText("--")
        else:
            self.num_label.setText("--")

    def on_model(self, idx):
        name = self.model_box.itemData(idx)
        if name and self.worker:
            self.worker.classifier.current = name

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("QMainWindow{background:#1a1a20;}")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
