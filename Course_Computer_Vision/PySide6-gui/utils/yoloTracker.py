import os
import torch
import cv2
from ultralytics import YOLO

# ====== 1) 动态路径计算与预设目标 ======
# 模型存放在 utils/models/ 或项目根目录的 models/ 目录下
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models", "yolo26s.pt")
# DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models", "yolo26s.pt"))
DEFAULT_OBJ_LIST = ['person', 'car', 'bus', 'truck']

class BaseTracker:
    def __init__(self, conf=0.25, iou=0.70):
        self.img_size = 640
        self.conf = conf  # 接收外部传入
        self.iou = iou

    def init_model(self):
        # 初始化模型的方法：由子类负责真正实现
        raise NotImplementedError("Subclasses must implement init_model().")

    # 新增 current_obj_list 参数
    def draw_bboxes(self, im, pred_boxes, current_obj_list):
        # 预设常见类别的区分颜色 (绿, 蓝, 红, 黄)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
        for box in pred_boxes:
            x1, y1, x2, y2, lbl, _, track_id = box
            class_idx = current_obj_list.index(lbl) if lbl in current_obj_list else -1
            if class_idx != -1:
                color = colors[class_idx]
            else:
                color = (0, 0, 0)  # Default color if class not in OBJ_LIST
            thickness = 2

            # Draw the bounding box
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # Add text label with track_id in magenta color
            text = f'{lbl} (ID:{track_id})'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # 计算颜色亮度
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            # 根据亮度自动选择文字颜色
            text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)

            # 半透明文字背景框位置
            padding = 5  # 文字框加 padding
            text_x = int(x1)
            text_y = int(y1) - 5
            box_start = (text_x, text_y - text_size[1] - 2 * padding)
            box_end = (text_x + text_size[0] + 2 * padding, text_y)

            overlay = im.copy()
            cv2.rectangle(overlay, box_start, box_end, color, -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
            # cv2.rectangle(im, box_start, box_end, color, -1)

            # 绘制文字
            text_pos = (text_x + padding, text_y - padding)
            cv2.putText(im, text, text_pos,
                        font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        return im

class YOLOTracker(BaseTracker):
    def __init__(self, model_path=DEFAULT_MODEL_PATH, obj_list=None, conf=0.25, iou=0.70):
        super().__init__(conf, iou)  # 传递给父类

        self.weights = model_path
        self.obj_list = obj_list if obj_list is not None else DEFAULT_OBJ_LIST
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.names = None

        self.init_model()

    def init_model(self):
        """挂载 YOLO 神经网络模型"""
        # 如果模型文件不存在，Ultralytics 会自动提示或尝试下载
        self.model = YOLO(self.weights)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(f"啟動模型為：{self.weights}")
        print(f"[INFO] YOLO Tracker 引擎就绪. 硬件设备: {self.device}")

    def update_params(self, conf=None, iou=None, obj_list=None):
        """提供给外界 (UI层) 的热调参接口"""
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        if obj_list is not None:
            self.obj_list = obj_list

    def track(self, im):
        results = self.model.track(im, tracker="bytetrack.yaml", persist=True, imgsz=self.img_size,
                                   conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        detected_boxes = results[0].boxes
        pred_boxes = []

        # 增加安全检查：如果这一帧什么都没检测到，直接返回原图和空列表
        if detected_boxes is None or len(detected_boxes) == 0:
            return im, pred_boxes

        for box in detected_boxes:
            # 健壮性检查：如果当前目标没有分配到 track_id，则跳过或设为 -1
            if box.id is None:
                continue  # 教学演示中通常直接 continue，确保画面上只出现稳定追踪的 ID

            class_id = box.cls.int().cpu().item()
            lbl = self.names[class_id]

            # 类别过滤：只追踪我们需要的列表目标
            if not lbl in self.obj_list:
                continue

            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = box.conf.cpu().item()
            track_id = box.id.int().cpu().item()
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))

        # im = self.draw_bboxes(im, pred_boxes)
        im = self.draw_bboxes(im, pred_boxes, self.obj_list)

        return im, pred_boxes

    def close(self):
        """安全释放生命周期资源，避免 GPU OOM (Out Of Memory)"""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    tracker = YOLOTracker()

    # 读取测试图片
    img_bgr = cv2.imread('../images/bus.jpg')
    # img_bgr = cv2.imread('images/zidane.jpg')

    # 检查图片是否读取成功
    if img_bgr is None:
        print("图片读取失败，请检查路径是否正确。")
    else:
        # 执行追踪
        img_bgr, pred_boxes = tracker.track(img_bgr)
        # print(pred_boxes)

        # 显示图片
        cv2.imshow('Tracked Image', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()