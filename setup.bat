@echo off
chcp 65001 >nul
REM ================================================================
REM  一键创建虚拟环境并安装依赖 (Windows)
REM  默认安装 CPU 版 PyTorch；如需 CUDA 版请修改下方参数
REM ================================================================

setlocal

REM ---- 配置区：按需修改 ----
REM PyTorch 版本：cpu / cu118 / cu121
set "TORCH_VARIANT=cpu"
REM 虚拟环境目录名
set "VENV_NAME=.venv"
REM --------------------------

echo.
echo ================================================================
echo   Python 虚拟环境一键创建
echo   PyTorch 版本 : %TORCH_VARIANT%
echo   虚拟环境名   : %VENV_NAME%
echo ================================================================
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未检测到 python 命令，请先安装 Python 3.9+ 并加入 PATH。
    pause
    exit /b 1
)

REM 调用主脚本
python create_venv.py --torch %TORCH_VARIANT% --venv-name %VENV_NAME%

if errorlevel 1 (
    echo.
    echo [ERROR] 创建过程出现错误，请查看上方日志。
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   全部完成！
echo   激活虚拟环境请运行: activate.bat
echo ================================================================
pause
endlocal