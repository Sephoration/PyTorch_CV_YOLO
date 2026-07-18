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
    """追踪器基础类：负责定义通用属性与核心绘图逻辑"""
    def __init__(self, conf=0.25, iou=0.70):
        # 推理参数
        self.img_size = 640
        self.conf = conf
        self.iou = iou

    def init_model(self):
        # 初始化模型的方法：由子类负责真正实现
        raise NotImplementedError("Subclasses must implement init_model().")

    def draw_bboxes(self, im, pred_boxes, current_obj_list):
        """
        高阶视觉美化版 BBox 绘制 (支持半透明底色与自适应文字颜色)
        """
        # 预设常见类别的区分颜色 (绿, 蓝, 红, 黄)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

        for box in pred_boxes:
            x1, y1, x2, y2, lbl, _, track_id = box
            class_idx = current_obj_list.index(lbl) if lbl in current_obj_list else -1

            # 动态颜色分配：超出预设颜色表则给予默认黑色
            if class_idx != -1 and class_idx < len(colors):
                color = colors[class_idx]
            else:
                color = (0, 0, 0)

            thickness = 2

            # 1. 绘制边界框
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # 2. 准备文字标签与字体参数
            text = f'{lbl} (ID:{track_id})'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # 3. 计算颜色亮度 (Luminance) 决定文字是黑色还是白色，极具工业水准的设计！
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)

            # 4. 绘制带有 Alpha 融合的半透明文字背景框
            padding = 5
            text_x = int(x1)
            text_y = int(y1) - 5
            box_start = (text_x, text_y - text_size[1] - 2 * padding)
            box_end = (text_x + text_size[0] + 2 * padding, text_y)

            overlay = im.copy()
            cv2.rectangle(overlay, box_start, box_end, color, -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

            # 5. 绘制文字
            text_pos = (text_x + padding, text_y - padding)
            cv2.putText(im, text, text_pos,
                        font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        return im


class YOLOTracker(BaseTracker):
    """YOLO 核心追踪器引擎"""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, obj_list=None, conf=0.25, iou=0.70):
        # 先继承父类中的共用参数
        super().__init__(conf, iou)

        self.weights = model_path
        self.obj_list = obj_list if obj_list is not None else DEFAULT_OBJ_LIST
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.names = None

        # 实例化时立即初始化模型
        self.init_model()

    def init_model(self):
        """挂载 YOLO 神经网络模型"""
        # 如果模型文件不存在，Ultralytics 会自动提示或尝试下载
        self.model = YOLO(self.weights)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(f"启动模型为：{self.weights}")
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
        """核心追踪推论管道"""
        results = self.model.track(im, tracker="bytetrack.yaml", persist=True, imgsz=self.img_size,
                                   conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        detected_boxes = results[0].boxes
        pred_boxes = []

        # 增加安全检查：如果这一帧什么都没检测到，直接返回原图和空列表
        if detected_boxes is None or len(detected_boxes) == 0:
            return im, pred_boxes

        for box in detected_boxes:
            # 健壮性检查：跳过没有分配到稳定 track_id 的对象
            if box.id is None:
                continue

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

        # 呼叫父类的绘画功能
        im = self.draw_bboxes(im, pred_boxes, self.obj_list)

        return im, pred_boxes

    def close(self):
        """安全释放生命周期资源，避免 GPU OOM (Out Of Memory)"""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==========================================
# 调试入口：测试 utils/YOLOTracker.py 是否可独立正常运行
# ==========================================
if __name__ == '__main__':
    # 实例化追踪引擎
    tracker = YOLOTracker()

    # 读取测试图片 (请确保当前目录下有 images/bus.jpg)
    img_path = os.path.join(os.path.dirname(__file__), "../images/bus.jpg")
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"图片读取失败，请检查路径: {img_path}")
    else:
        # 执行追踪
        img_bgr, pred_boxes = tracker.track(img_bgr)
        print(f"检测到符合条件的目标数量: {len(pred_boxes)}")

        # 显示图片
        cv2.imshow('YOLO Tracker Standalone Test', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tracker.close()