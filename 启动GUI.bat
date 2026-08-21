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

"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe" -m glass_bottle_defect_gui.main

if errorlevel 1 (
    echo.
    echo 启动失败！请检查依赖是否已安装
    pause
)
