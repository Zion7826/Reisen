# -*- coding: utf-8 -*-
"""玻璃瓶缺陷检测 GUI 主入口"""

import sys
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtCore import Qt

from glass_bottle_defect_gui.presentation.main_window import MainWindow
from glass_bottle_defect_gui.infrastructure.logger import logger


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # 窗口居中显示
    screen = QDesktopWidget().screenGeometry()
    window.move((screen.width() - window.width()) // 2, 
                (screen.height() - window.height()) // 2)
    
    # 强制显示并激活窗口
    window.show()
    window.raise_()
    window.activateWindow()
    window.setWindowState(Qt.WindowActive)
    
    logger.info("玻璃瓶缺陷检测系统已启动")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
