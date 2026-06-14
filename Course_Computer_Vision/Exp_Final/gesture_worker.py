# gesture_worker.py
"""QThread 摄像头推理 — 画面只保留骨架 + 底部进度条(含数字标识) + FPS"""
import time, cv2, numpy as np
from collections import deque, Counter
from PySide6.QtCore import QThread, Signal
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier

LOCK_DURATION = 2.5
SMOOTH_WINDOW = 7
MIN_CONFIDENCE = 0.3
FRAME_SKIP = 1


class GestureWorker(QThread):
    frame_ready = Signal(np.ndarray)
    result_ready = Signal(dict)

    def __init__(self):
        super().__init__()
        self._running = False
        self.detection_con = 0.7
        self.show_skeleton = True
        self.show_lock_bar = True

    def run(self):
        self._running = True
        classifier = GestureClassifier()
        tracker = HandTracker(detection_con=self.detection_con)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，降低延迟
        if not cap.isOpened():
            self.result_ready.emit({"status": "摄像头打开失败"})
            tracker.close()
            return

        # 丢弃前几帧（刚启动时曝光不稳定）
        for _ in range(5):
            cap.read()

        # 用多帧真实画面预热 MediaPipe，消除首次手部进入卡顿
        for _ in range(10):
            ret, warm = cap.read()
            if ret:
                tracker.detect(warm)
            self.msleep(10)

        pred_history = deque(maxlen=SMOOTH_WINDOW)
        lock_start = 0.0
        locked_pred = -1
        fps_time = time.time()
        fps = 0.0
        fc = 0
        last_result = None

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.msleep(5)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            fc += 1
            now = time.time()

            if fc % 15 == 0:
                dt = now - fps_time
                fps = 15 / dt if dt > 0 else 0
                fps_time = now

            # ---- 检测（隔帧）----
            if fc % (FRAME_SKIP + 1) == 0:
                tracker.detect(frame)

                if tracker.results and tracker.results.hand_landmarks:
                    hand_lms = tracker.results.hand_landmarks[0]
                    pred, proba = classifier.predict(hand_lms)

                    if pred is not None and proba is not None:
                        top_conf = float(np.max(proba))
                        if top_conf >= MIN_CONFIDENCE:
                            pred_history.append(pred)
                        else:
                            pred_history.clear()

                        smooth_pred = (Counter(pred_history).most_common(1)[0][0]
                                       if len(pred_history) >= 3 else pred)

                        if locked_pred >= 0:
                            # 已锁定：手势不变则保持锁定，变了才解锁
                            if smooth_pred == locked_pred:
                                pass
                            else:
                                locked_pred = -1
                                lock_start = 0
                        elif (smooth_pred == pred and
                              len(pred_history) >= SMOOTH_WINDOW):
                            # 未锁定 + 手势稳定 → 计时
                            if lock_start == 0:
                                lock_start = now
                            elif now - lock_start >= LOCK_DURATION:
                                locked_pred = smooth_pred
                        else:
                            lock_start = 0

                        display = locked_pred if locked_pred >= 0 else smooth_pred
                        locked = locked_pred >= 0
                        lp = (min(1.0, (now - lock_start) / LOCK_DURATION)
                              if lock_start > 0 and not locked else 0.0)

                        last_result = {
                            "pred": display, "locked": locked,
                            "lock_progress": lp, "confidence": top_conf,
                            "proba": proba, "smooth": smooth_pred,
                        }
                    else:
                        pred_history.clear()
                        lock_start = 0; locked_pred = -1; last_result = None
                else:
                    pred_history.clear()
                    lock_start = 0; locked_pred = -1; last_result = None

            # ---- 绘制骨架 ----
            if self.show_skeleton:
                frame = tracker.draw(frame)

            res = last_result

            # ---- 底部进度条（含数字标识）----
            if self.show_lock_bar:
                bar_w = min(w - 40, 300)
                bar_x = (w - bar_w) // 2
                bar_y = h - 55
                bar_h = 16

                if res and res["lock_progress"] > 0 and not res["locked"]:
                    # 进度条背景
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                    # 进度填充
                    fill_w = int((bar_w - 4) * res["lock_progress"])
                    cv2.rectangle(frame, (bar_x + 2, bar_y + 2),
                                  (bar_x + 2 + fill_w, bar_y + bar_h - 2),
                                  (0, 220, 220), -1)
                    # 边框
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 2)

                    # 进度条左边显示当前预测数字
                    cur = res.get("smooth", res["pred"])
                    if cur is not None:
                        num_x = bar_x - 35
                        num_y = bar_y + bar_h // 2 + 8
                        cv2.putText(frame, str(cur), (num_x, num_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 220), 3)

                elif res and res["locked"]:
                    # 已锁定：绿色满条 + 数字
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (0, 80, 0), -1)
                    cv2.rectangle(frame, (bar_x + 2, bar_y + 2),
                                  (bar_x + bar_w - 2, bar_y + bar_h - 2),
                                  (0, 255, 0), -1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), 2)

                    if res["pred"] is not None:
                        num_x = bar_x - 35
                        num_y = bar_y + bar_h // 2 + 8
                        cv2.putText(frame, str(res["pred"]), (num_x, num_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # ---- FPS ----
            cv2.putText(frame, f"{fps:.0f} FPS", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            self.frame_ready.emit(frame)
            if res:
                self.result_ready.emit({
                    "pred": res["pred"], "locked": res["locked"],
                    "lock_progress": res["lock_progress"],
                    "confidence": res["confidence"],
                    "proba": res["proba"].tolist() if res["proba"] is not None else None,
                    "model": classifier.current, "fps": round(fps, 1),
                })
            else:
                self.result_ready.emit({
                    "pred": None, "locked": False, "lock_progress": 0.0,
                    "confidence": 0.0, "proba": None,
                    "model": classifier.current, "fps": round(fps, 1),
                })

        cap.release()
        tracker.close()

    def stop(self):
        self._running = False
