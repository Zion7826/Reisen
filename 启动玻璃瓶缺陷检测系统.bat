@echo off
chcp 65001 >nul 2>&1
title 玻璃瓶缺陷检测系统

echo.
echo ========================================
echo   玻璃瓶缺陷检测系统
echo ========================================
echo.
echo 正在启动 GUI 界面...
echo.

python -m glass_bottle_defect_gui.main

if errorlevel 1 (
    echo.
    echo ========================================
    echo   启动失败！
    echo ========================================
    echo.
    echo 可能的原因：
    echo 1. Python 未安装或未添加到 PATH
    echo 2. 依赖包未安装
    echo.
    echo 解决方法：
    echo 运行以下命令安装依赖：
    echo   pip install -r requirements.txt
    echo.
    echo 或单独安装：
    echo   pip install PyQt5 opencv-python ultralytics torch torchvision pandas openpyxl
    echo.
    pause
)