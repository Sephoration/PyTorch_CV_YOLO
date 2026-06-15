# gesture_worker.py
"""QThread 摄像头推理 — 画面只保留骨架 + 底部进度条(含数字标识) + FPS"""
import os, time, cv2, numpy as np
from collections import deque, Counter
from PySide6.QtCore import QThread, Signal
from p3_hand_tracker import HandTracker
from p3_gesture_classifier import GestureClassifier

LOCK_DURATION = 0.9
SMOOTH_WINDOW = 7
MIN_CONFIDENCE = 0.25
FRAME_SKIP = 2          # 隔 2 帧检测（每 3 帧一次检测），大幅提升 FPS
UNLOCK_THRESHOLD = 3   # 连续 N 帧不同手势才解锁，防噪声误断
HAND_HOLD = 0.3        # 手消失后锁定进度保持时间（秒）


class GestureWorker(QThread):
    frame_ready = Signal(np.ndarray)
    result_ready = Signal(dict)

    def __init__(self, classifier=None):
        super().__init__()
        self._running = False
        self.classifier = classifier
        self.detection_con = 0.7
        self.show_skeleton = True
        self.show_lock_bar = True

    def run(self):
        self._running = True
        classifier = self.classifier or GestureClassifier()

        # 先开摄像头（快），再加载模型（慢），摄像头灯先亮
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)      # 最小缓冲，降低延迟
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if actual_w < 300:
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        if not cap.isOpened():
            self.result_ready.emit({"status": "摄像头打开失败"})
            return

        tracker = HandTracker(detection_con=self.detection_con)

        # 用假图预热跟踪模型 + 分类器，消除首次手部进入卡顿
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(5):
            tracker.detect(blank)
        # 预热分类器：生成虚拟 MediaPipe landmark 对象
        try:
            dummy = type('LM', (), {'x': 0.5, 'y': 0.5, 'z': 0.0})()
            classifier.predict([dummy] * 21)
        except Exception:
            pass

        # 丢弃前几帧（刚启动时曝光不稳定）
        for _ in range(2):
            cap.read()

        pred_history = deque(maxlen=SMOOTH_WINDOW)
        lock_start = 0.0
        locked_pred = -1
        unlock_counter = 0     # 连续"不同手势"计数
        fps_time = time.time()
        fps = 0.0
        fc = 0
        last_result = None
        last_detect_ms = 0.0
        last_classify_ms = 0.0
        last_hand_time = 0.0
        held_result = None
        held_lock_progress = 0.0

        # ---- 性能分析器（每 N 帧打印一次耗时明细）----
        PROFILER_INTERVAL = 30
        prof = {"read": 0.0, "detect": 0.0, "classify": 0.0, "draw": 0.0, "count": 0}

        camera_lost = False
        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            t1 = time.time()
            if not ret:
                if not camera_lost:
                    camera_lost = True
                    self.result_ready.emit({"status": "未找到摄像头"})
                self.msleep(5)
                continue
            camera_lost = False

            h, w = frame.shape[:2]
            fc += 1
            now = time.time()

            # ---- FPS（按时间间隔更新，比按帧数更均匀）----
            if now - fps_time >= 1.0:
                fps = fc / (now - fps_time) if now > fps_time else 0
                fc = 0
                fps_time = now

            prof["read"] += t1 - t0
            # ---- 检测（隔帧）----
            if fc % (FRAME_SKIP + 1) == 0:
                t2 = time.time()
                tracker.detect(frame)

                t3 = time.time()
                last_detect_ms = round((t3 - t2) * 1000, 1)
                prof["detect"] += t3 - t2

                if tracker.results and tracker.results.hand_landmarks:
                    hand_lms = tracker.results.hand_landmarks[0]
                    pred, proba = classifier.predict(hand_lms)
                    t4 = time.time()
                    last_classify_ms = round((t4 - t3) * 1000, 1)
                    prof["classify"] += t4 - t3

                    if pred is not None and proba is not None:
                        top_conf = float(np.max(proba))
                        if top_conf >= MIN_CONFIDENCE:
                            pred_history.append(pred)
                        else:
                            pred_history.clear()

                        smooth_pred = (Counter(pred_history).most_common(1)[0][0]
                                       if len(pred_history) >= 3 else pred)

                        if locked_pred >= 0:
                            # 已锁定：连续"不同"超过阈值才解锁，防噪声误断
                            if smooth_pred == locked_pred:
                                unlock_counter = 0
                            else:
                                unlock_counter += 1
                                if unlock_counter >= UNLOCK_THRESHOLD:
                                    locked_pred = -1
                                    lock_start = 0
                                    unlock_counter = 0
                        elif (smooth_pred == pred and
                              len(pred_history) >= SMOOTH_WINDOW):
                            # 未锁定 + 手势稳定 → 计时
                            if lock_start == 0:
                                lock_start = now
                            elif now - lock_start >= LOCK_DURATION:
                                locked_pred = smooth_pred
                                unlock_counter = 0
                        else:
                            lock_start = 0
                            unlock_counter = 0

                        display = locked_pred if locked_pred >= 0 else smooth_pred
                        locked = locked_pred >= 0

                        last_result = {
                            "pred": display, "locked": locked,
                            "confidence": top_conf,
                            "proba": proba, "smooth": smooth_pred,
                            "lock_start": lock_start,
                        }
                    else:
                        pred_history.clear()
                        lock_start = 0; locked_pred = -1; unlock_counter = 0
                        last_result = None
                else:
                    pred_history.clear()
                    lock_start = 0; locked_pred = -1; unlock_counter = 0
                    last_result = None

            # ---- 每帧计算锁定进度（让进度条动画平滑）----
            if last_result and last_result.get("pred") is not None:
                res = last_result
                last_hand_time = now
                if res.get("locked"):
                    lock_progress = 1.0
                elif res.get("lock_start", 0) > 0:
                    lock_progress = min(1.0, (now - res["lock_start"]) / LOCK_DURATION)
                else:
                    lock_progress = 0.0
                held_result = res
                held_lock_progress = lock_progress
            elif now - last_hand_time < HAND_HOLD:
                res = held_result
                lock_progress = held_lock_progress
            else:
                res = None
                lock_progress = 0.0

            # ---- 绘制骨架 ----
            if self.show_skeleton:
                frame = tracker.draw(frame)

            # ---- 翻转为镜像显示（检测用原始帧，用户看镜像）----
            show = cv2.flip(frame, 1)

            # ---- 底部进度条（含数字标识）----
            if self.show_lock_bar:
                bar_w = min(w - 40, 300)
                bar_x = (w - bar_w) // 2
                bar_y = h - 55
                bar_h = 16

                if res and lock_progress > 0 and not res.get("locked"):
                    # 进度条背景
                    cv2.rectangle(show, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                    # 进度填充
                    fill_w = int((bar_w - 4) * lock_progress)
                    cv2.rectangle(show, (bar_x + 2, bar_y + 2),
                                  (bar_x + 2 + fill_w, bar_y + bar_h - 2),
                                  (0, 220, 220), -1)
                    # 边框
                    cv2.rectangle(show, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 2)

                    # 进度条左边显示当前预测数字
                    cur = res.get("smooth", res["pred"])
                    if cur is not None:
                        num_x = bar_x - 35
                        num_y = bar_y + bar_h // 2 + 8
                        cv2.putText(show, str(cur), (num_x, num_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 220), 3)

                elif res and res.get("locked"):
                    # 已锁定：绿色满条 + 数字
                    cv2.rectangle(show, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (0, 80, 0), -1)
                    cv2.rectangle(show, (bar_x + 2, bar_y + 2),
                                  (bar_x + bar_w - 2, bar_y + bar_h - 2),
                                  (0, 255, 0), -1)
                    cv2.rectangle(show, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), 2)

                    if res["pred"] is not None:
                        num_x = bar_x - 35
                        num_y = bar_y + bar_h // 2 + 8
                        cv2.putText(show, str(res["pred"]), (num_x, num_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # ---- FPS + 模型（左上 / 右上）----
            cv2.putText(show, f"{fps:.0f} FPS", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            model_text = classifier.meta[classifier.current]["display"] if (classifier.current and classifier.current in classifier.meta) else "N/A"
            (tw, th), _ = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(show, model_text, (w - tw - 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            prof["draw"] += time.time() - t1
            prof["count"] += 1
            if prof["count"] >= PROFILER_INTERVAL:
                prof["read"] = prof["detect"] = prof["classify"] = prof["draw"] = 0.0
                prof["count"] = 0

            self.frame_ready.emit(show)
            if res:
                self.result_ready.emit({
                    "pred": res["pred"], "locked": res.get("locked", False),
                    "lock_progress": lock_progress,
                    "confidence": res.get("confidence", 0.0),
                    "proba": res["proba"].tolist() if res.get("proba") is not None else None,
                    "model": classifier.current, "fps": round(fps, 1),
                    "detect_ms": last_detect_ms, "classify_ms": last_classify_ms,
                })
            else:
                self.result_ready.emit({
                    "pred": None, "locked": False, "lock_progress": 0.0,
                    "confidence": 0.0, "proba": None,
                    "model": classifier.current, "fps": round(fps, 1),
                    "detect_ms": last_detect_ms, "classify_ms": last_classify_ms,
                })

        cap.release()
        tracker.close()

    def stop(self):
        self._running = False
