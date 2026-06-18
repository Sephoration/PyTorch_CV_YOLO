import sys
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# ==========================================================
# 自主选择布局区域（想要哪个版本，就取消哪一行的注释）
# ==========================================================
from gui.main_window import VisionWorkstationGUI as VisionGUI
# from gui.main_window_1 import VisionWorkstationGUI as VisionGUI
# from gui.main_window_2 import VisionWorkstationGUI as VisionGUI # 老师推荐版
# from gui.main_window_3 import VisionWorkstationGUI as VisionGUI
# from gui.main_window_4 import VisionWorkstationGUI as VisionGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    # 无论上面选哪个，这里实例化的名字都叫 VisionGUI()
    window = VisionGUI()
    window.show()

    sys.exit(app.exec())