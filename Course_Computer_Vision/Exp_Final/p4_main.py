import os
os.environ['GLOG_minloglevel'] = '2'
import sys
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from p3_gui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())