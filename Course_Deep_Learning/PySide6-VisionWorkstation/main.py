# main.py
"""计算机视觉综合实验工作站 - 入口（PPT Slide 24-37）"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from gui.main_window import VisionWorkstationGUI


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    window = VisionWorkstationGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
