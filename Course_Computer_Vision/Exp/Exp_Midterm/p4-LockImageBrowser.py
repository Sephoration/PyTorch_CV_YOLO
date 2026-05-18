"""
p4-LockImageBrowser.py
手势锁定控制图片浏览器

功能：
  伸出并保持手指数锁定后执行操作，防止误触
  [1] 锁定 -> Prev    [2] 锁定 -> Next
  [3] 锁定 -> Rotate  [4] 锁定两次 -> Delete
  5 -> Clear   |   5-1-2 -> Exit
"""

import cv2
import time
import os
import shutil
import numpy as np
import HandTrackingModule as htm


# ==================== 手指检测 ====================

def count_fingers(lmList, hand_label):
    if len(lmList) < 21:
        return 0
    fingers = []
    tipIds = [4, 8, 12, 16, 20]
    if hand_label == "Left":
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)


# ==================== 图片浏览器 ====================

class ImageBrowser:
    def __init__(self, folder):
        self.folder = folder
        self.image_files = []
        self.current_index = 0
        self.rotation = 0
        self.load_images()

    def load_images(self):
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.image_files = sorted([
            f for f in os.listdir(self.folder)
            if f.lower().endswith(valid_ext)
        ])

    def get_current_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            return None
        path = os.path.join(self.folder, self.image_files[self.current_index])
        img = cv2.imread(path)
        if img is None:
            return None
        if self.rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.rotation = 0

    def prev_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.rotation = 0

    def rotate_image(self):
        self.rotation = (self.rotation + 90) % 360

    def delete_current(self):
        if not self.image_files:
            return False
        path = os.path.join(self.folder, self.image_files[self.current_index])
        if os.path.exists(path):
            deleted_folder = os.path.join(self.folder, "deleted")
            os.makedirs(deleted_folder, exist_ok=True)
            shutil.move(path, os.path.join(deleted_folder, self.image_files[self.current_index]))
            self.load_images()
            if self.current_index >= len(self.image_files):
                self.current_index = max(0, len(self.image_files) - 1)
            self.rotation = 0
            return True
        return False

    def get_info_text(self):
        if not self.image_files:
            return ""
        return f"{self.current_index + 1} / {len(self.image_files)}"


# ==================== 主程序 ====================

W, H = 960, 680
CAM_W, CAM_H = 200, 150
LOCK_DURATION = 1.5

FUNC_MAP = {1: "Prev", 2: "Next", 3: "Rotate", 4: "Delete", 5: "Clear"}

STATE_COLORS = {
    "idle": (180, 180, 180), "tracking": (0, 230, 255),
    "locked": (0, 255, 128), "arming": (255, 0, 0), "clear": (0, 255, 0),
}


def main():
    IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "images")

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"[INFO] Created image folder: {IMAGE_FOLDER}")
        input("Press Enter after adding images...")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    cap.set(3, 640)
    cap.set(4, 480)

    htm.ensure_model(htm.MODEL_PATH, htm.MODEL_URL)
    detector = htm.HandDetector(detectionCon=0.75)
    browser = ImageBrowser(IMAGE_FOLDER)

    if not browser.image_files:
        print(f"No images in {IMAGE_FOLDER}")
        cap.release()
        return

    # ---- state ----
    tracked = -1
    track_start = 0
    delete_armed = False
    msg = ""
    msg_color = (200, 200, 200)
    msg_timer = 0
    state = "idle"

    # 5-1-2 exit
    seq = []
    last_seq = 0
    exit_armed = False
    exit_armed_start = 0

    cv2.namedWindow("Gesture Lock Image Browser", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Lock Image Browser", W, H)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        now = time.time()

        # ---- hand detection ----
        frame = detector.findHands(frame, draw=True, flip=False)
        lmList = detector.findPosition(frame, draw=False)

        finger_count = 0
        hand_label = None

        if len(lmList) >= 21:
            if detector.results and detector.results.handedness:
                hand_label = detector.results.handedness[0][0].category_name
            finger_count = count_fingers(lmList, hand_label)

        # ======== 5-1-2 exit (track finger transitions with lock) ========
        if finger_count in (1, 2, 5):
            if finger_count != last_seq:
                seq.append(finger_count)
                if len(seq) > 3:
                    seq.pop(0)
                if seq == [5, 1]:
                    exit_armed = True
                    exit_armed_start = now
                    state = "arming"
                    msg = "Hold 2 to Exit"
                    msg_color = (255, 0, 255)
                    msg_timer = now
                elif seq == [5, 1, 2]:
                    seq = []
                last_seq = finger_count
        elif finger_count == 0:
            last_seq = 0
            if exit_armed:
                exit_armed = False
                exit_armed_start = 0

        if exit_armed and finger_count == 2:
            if now - exit_armed_start >= LOCK_DURATION:
                print("5-1-2 exit")
                break
            else:
                state = "tracking"

        # ======== lock logic ========

        if finger_count == 5:
            tracked = -1
            track_start = 0
            delete_armed = False
            msg = ">> Cleared <<"
            msg_color = (0, 255, 0)
            msg_timer = now
            state = "clear"

        elif 1 <= finger_count <= 4:
            if finger_count == tracked:
                elapsed = now - track_start
                if elapsed >= LOCK_DURATION:
                    if finger_count == 4:
                        if delete_armed:
                            ok = browser.delete_current()
                            msg = ">> Deleted <<" if ok else ">> Failed <<"
                            msg_color = (0, 0, 255)
                            delete_armed = False
                            state = "locked"
                        else:
                            delete_armed = True
                            msg = "Lock 4 again to delete"
                            msg_color = (0, 0, 255)
                            state = "arming"
                        tracked = -1
                        track_start = 0
                    else:
                        if finger_count == 1:
                            browser.prev_image()
                            msg = ">> Prev <<"
                        elif finger_count == 2:
                            browser.next_image()
                            msg = ">> Next <<"
                        elif finger_count == 3:
                            browser.rotate_image()
                            msg = ">> Rotated <<"
                        msg_color = (0, 255, 255)
                        tracked = -1
                        track_start = 0
                        state = "locked"
                    msg_timer = now
                else:
                    state = "tracking"
                    msg = ""
            else:
                tracked = finger_count
                track_start = now
                if finger_count != 4:
                    delete_armed = False
                msg = ""
                state = "tracking"
        else:
            tracked = -1
            track_start = 0
            state = "idle"
            if msg and now - msg_timer > 2.5:
                msg = ""

        if delete_armed and state != "tracking":
            state = "arming"

        # ======== UI ========
        canvas = np.full((H, W, 3), 28, dtype=np.uint8)

        current_img = browser.get_current_image()

        if current_img is not None:
            h_img, w_img = current_img.shape[:2]
            scale = min((W - 20) / w_img, (H - 130) / h_img)
            dw, dh = int(w_img * scale), int(h_img * scale)
            display = cv2.resize(current_img, (dw, dh))
            x0 = (W - dw) // 2
            y0 = (H - 105 - dh) // 2 + 40
            canvas[y0:y0+dh, x0:x0+dw] = display
        else:
            cv2.putText(canvas, "No Image", (W // 2 - 70, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)

        # ---- camera pip ----
        cam = cv2.resize(frame, (CAM_W, CAM_H))
        canvas[0:CAM_H, W - CAM_W:W] = cam

        # ---- top status (compact single line) ----
        state_name = state.upper()
        state_color = STATE_COLORS.get(state, (180, 180, 180))

        if state == "idle" or finger_count == 0:
            right_text = "Waiting..."
        elif finger_count == 5:
            right_text = "5 -> Clear"
        else:
            right_text = f"{finger_count} -> {FUNC_MAP.get(finger_count, '')}"

        line1 = f"{state_name}  |  {right_text}"

        cv2.putText(canvas, line1, (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        # ---- image info ----
        info = browser.get_info_text()
        cv2.putText(canvas, f"Image  {info}", (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # ---- action msg ----
        if msg:
            cv2.putText(canvas, msg, (15, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2)

        # ---- progress bar ----
        if state == "tracking" and finger_count == tracked and 1 <= finger_count <= 4:
            elapsed = now - track_start
            if elapsed < LOCK_DURATION:
                ratio = elapsed / LOCK_DURATION
                bx, by, bw, bh = 20, H - 50, 280, 12
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
                fill = int(bw * ratio)
                bar_color = (0, 200, 255) if ratio < 1.0 else (0, 255, 80)
                cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), bar_color, -1)
                cv2.putText(canvas, f"LOCK {int(ratio * 100)}%",
                            (bx + bw + 10, by + 11), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (220, 220, 220), 2)

        if exit_armed and finger_count == 2:
            elapsed = now - exit_armed_start
            if elapsed < LOCK_DURATION:
                ratio = elapsed / LOCK_DURATION
                bx, by, bw, bh = 20, H - 50, 280, 12
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
                fill = int(bw * ratio)
                bar_color = (255, 0, 255) if ratio < 1.0 else (0, 255, 80)
                cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), bar_color, -1)
                cv2.putText(canvas, f"EXIT LOCK {int(ratio * 100)}%",
                            (bx + bw + 10, by + 11), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (220, 220, 220), 2)

        # ---- guide ----
        cv2.putText(canvas, "1:Prev  2:Next  3:Rotate  4:Delete(x2)  5:Clear  [5-1]->Hold 2:Exit",
                    (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.imshow("Gesture Lock Image Browser", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
