import sys
from PySide6.QtWidgets import QApplication
from yolo_gui.main_window import YOLOMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = YOLOMainWindow()
    window.show()

    sys.exit(app.exec())