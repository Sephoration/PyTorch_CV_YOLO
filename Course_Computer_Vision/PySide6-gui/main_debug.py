# main.py
import sys
import traceback
from PySide6.QtWidgets import QApplication

# 🎯 强力诊断：拦截导入阶段的错误
try:
    # 请确保您的 main_window_2.py 确实保存在 gui 文件夹下
    from gui.main_window_2 import VisionWorkstationGUI

    print("【雷达提示】成功导入 gui/main_window_2.py")
except Exception as e:
    print("\n❌【严重错误】在导入 GUI 模块时发生崩溃，请检查文件名和路径：")
    traceback.print_exc()
    input("\n按下回车键退出程序...")
    sys.exit(1)

if __name__ == "__main__":
    # 🎯 强力诊断：拦截初始化和运行阶段的错误
    try:
        app = QApplication(sys.argv)

        print("【雷达提示】正在实例化主窗口 VisionWorkstationGUI...")
        window = VisionWorkstationGUI()

        # ====================================================================
        # ⚠️【关键检查点】
        # 如果您的 main.py 原本在这里写了类似下面的代码，请务必全部注释掉：
        # window.v_thread.start()
        # window.v_thread.change_media_source(0)
        # 因为 main_window_2.py 里面已经没有 v_thread 了！
        # ====================================================================

        print("【雷达提示】正在展现 GUI 视窗...")
        window.show()

        print("【雷达提示】PySide6 核心事件循环已成功接通！")
        sys.exit(app.exec())

    except Exception as e:
        print("\n❌【严重错误】程序在主循环启动前发生致命崩溃：")
        traceback.print_exc()
        input("\n按下回车键退出程序...")