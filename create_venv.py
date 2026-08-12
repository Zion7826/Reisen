# -*- coding: utf-8 -*-
"""
================================================================================
 Windows 平台 Python 虚拟环境一键创建脚本
 依赖：PyTorch / OpenCV / ultralytics(YOLOv8) / PyQt5
================================================================================
 功能说明：
   1. 检查本机 Python 版本（要求 >= 3.9）
   2. 在当前目录下创建虚拟环境 .venv
   3. 升级 pip / setuptools / wheel
   4. 安装 PyTorch（可选 CPU 或 CUDA 11.8 / 12.1 版本）
   5. 安装 requirements.txt 中的其余依赖
   6. 逐项验证关键包是否可正常 import
   7. 生成一键激活的 activate.bat

 用法：
   python create_venv.py                # 默认安装 CPU 版 PyTorch
   python create_venv.py --torch cpu    # 显式指定 CPU 版
   python create_venv.py --torch cu118  # 安装 CUDA 11.8 版 PyTorch
   python create_venv.py --torch cu121  # 安装 CUDA 12.1 版 PyTorch
   python create_venv.py --venv-name myenv  # 自定义虚拟环境目录名
================================================================================
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# ---------------------------- 全局配置 ----------------------------
MIN_PYTHON = (3, 9)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENV_NAME = ".venv"

# PyTorch 安装源（按 --torch 参数选择）
TORCH_INDEX = {
    "cpu":   "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
}

# 关键包验证清单（包名 -> import 名）
VERIFY_PACKAGES = {
    "torch":       "torch",
    "torchvision": "torchvision",
    "cv2":         "cv2",
    "ultralytics": "ultralytics",
    "PyQt5":       "PyQt5",
    "PyInstaller": "PyInstaller",
}


# ---------------------------- 工具函数 ----------------------------
def run(cmd, check=True, shell=False, capture=False):
    """统一执行子进程命令，打印命令本身，失败时抛错。"""
    print(f"\n[CMD] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        shell=shell,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        if capture:
            print(result.stdout)
            print(result.stderr)
        raise RuntimeError(f"命令执行失败，返回码 {result.returncode}")
    return result


def step(msg):
    """打印步骤分隔线。"""
    line = "=" * 70
    print(f"\n{line}\n>> {msg}\n{line}")


def info(msg):
    print(f"[INFO] {msg}")


def ok(msg):
    print(f"[ OK ] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


# ---------------------------- 步骤实现 ----------------------------
def check_python_version():
    """检查 Python 版本是否满足最低要求。"""
    step("步骤 1/6：检查 Python 版本")
    cur = sys.version_info[:3]
    info(f"当前 Python 版本：{cur[0]}.{cur[1]}.{cur[2]}")
    if (cur[0], cur[1]) < MIN_PYTHON:
        raise RuntimeError(
            f"Python 版本过低，要求 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}，"
            f"当前为 {cur[0]}.{cur[1]}.{cur[2]}"
        )
    ok(f"Python 版本满足要求 (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")


def create_venv(venv_name):
    """创建虚拟环境，返回 venv 路径与 pip 路径。"""
    step("步骤 2/6：创建虚拟环境")
    venv_path = SCRIPT_DIR / venv_name
    if venv_path.exists():
        warn(f"目录已存在：{venv_path}")
        choice = input("是否删除并重建？[y/N]: ").strip().lower()
        if choice == "y":
            import shutil
            shutil.rmtree(venv_path, ignore_errors=True)
            info(f"已删除旧目录：{venv_path}")
        else:
            warn("保留现有虚拟环境，跳过创建步骤。")
            return venv_path, get_pip_path(venv_path)

    run([sys.executable, "-m", "venv", str(venv_path)])
    ok(f"虚拟环境已创建：{venv_path}")
    return venv_path, get_pip_path(venv_path)


def get_pip_path(venv_path):
    """返回 venv 内的 python.exe 路径（Windows）。"""
    py = venv_path / "Scripts" / "python.exe"
    if not py.exists():
        raise RuntimeError(f"未找到虚拟环境 Python：{py}")
    return py


def upgrade_pip(python_exe):
    """升级 pip / setuptools / wheel。"""
    step("步骤 3/6：升级 pip 及基础工具")
    run([str(python_exe), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"])
    ok("pip / setuptools / wheel 已升级")


def install_torch(python_exe, torch_variant):
    """安装 PyTorch 指定版本。"""
    step(f"步骤 4/6：安装 PyTorch ({torch_variant})")
    index_url = TORCH_INDEX[torch_variant]
    # 显式指定 torch + torchvision，避免 ultralytics 拉错版本
    cmd = [
        str(python_exe), "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", index_url,
    ]
    run(cmd)
    ok(f"PyTorch ({torch_variant}) 安装完成")


def install_requirements(python_exe):
    """安装 requirements.txt 中的其余依赖。"""
    step("步骤 5/6：安装其余依赖 (opencv / ultralytics / pyqt5 ...)")
    req_file = SCRIPT_DIR / "requirements.txt"
    if not req_file.exists():
        warn(f"未找到 {req_file}，跳过。")
        return
    run([str(python_exe), "-m", "pip", "install", "-r", str(req_file)])
    ok("requirements.txt 依赖安装完成")

    # 安装 PyInstaller（打包工具）
    run([str(python_exe), "-m", "pip", "install", "pyinstaller==6.11.0"])
    ok("PyInstaller 安装完成")


def verify_installation(python_exe):
    """逐项验证关键包是否可正常导入并打印版本。"""
    step("步骤 6/6：验证安装结果")
    verify_script = (
        "import importlib, sys\n"
        "results = []\n"
        "for pip_name, import_name in [\n"
        + "".join(f"    ('{k}', '{v}'),\n" for k, v in VERIFY_PACKAGES.items())
        + "]:\n"
        "    try:\n"
        "        m = importlib.import_module(import_name)\n"
        "        ver = getattr(m, '__version__', '未知')\n"
        "        results.append((pip_name, True, ver))\n"
        "    except Exception as e:\n"
        "        results.append((pip_name, False, str(e)))\n"
        "for name, ok, ver in results:\n"
        "    flag = 'OK ' if ok else 'FAIL'\n"
        "    print(f'  [{flag}] {name:15s} -> {ver}')\n"
        "sys.exit(0 if all(r[1] for r in results) else 1)\n"
    )
    result = run([str(python_exe), "-c", verify_script], check=False, capture=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("部分关键包验证失败，请检查上方输出")
    ok("所有关键包验证通过")


def generate_activate_bat(venv_path):
    """生成一键激活脚本。"""
    bat_path = SCRIPT_DIR / "activate.bat"
    content = (
        "@echo off\n"
        f":: 一键激活虚拟环境：{venv_path.name}\n"
        f'call "{venv_path}\\Scripts\\activate.bat"\n'
        "echo.\n"
        f"echo 已激活虚拟环境: {venv_path.name}\n"
        "echo Python 路径:\n"
        "where python\n"
        "echo.\n"
        "cmd /k\n"
    )
    bat_path.write_text(content, encoding="utf-8")
    ok(f"激活脚本已生成：{bat_path}")


def print_summary(venv_path, torch_variant):
    """打印最终汇总信息。"""
    step("安装完成 - 使用说明")
    print(f"虚拟环境目录 : {venv_path}")
    print(f"PyTorch 版本  : {torch_variant}")
    print()
    print("后续使用方式：")
    print(f"  1) 双击运行 activate.bat  或")
    print(f'  2) 在命令行执行: {venv_path}\\Scripts\\activate.bat')
    print()
    print("快速测试 YOLOv8：")
    print("  python -c \"from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8 OK')\"")
    print()
    print("快速测试 PyQt5：")
    print("  python -c \"from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')\"")


# ---------------------------- 主入口 ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Windows 平台 Python 虚拟环境一键创建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--torch",
        choices=["cpu", "cu118", "cu121"],
        default="cpu",
        help="选择 PyTorch 安装版本：cpu(默认) / cu118(CUDA 11.8) / cu121(CUDA 12.1)",
    )
    parser.add_argument(
        "--venv-name",
        default=DEFAULT_VENV_NAME,
        help=f"虚拟环境目录名（默认：{DEFAULT_VENV_NAME}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    try:
        check_python_version()
        venv_path, python_exe = create_venv(args.venv_name)
        upgrade_pip(python_exe)
        install_torch(python_exe, args.torch)
        install_requirements(python_exe)
        verify_installation(python_exe)
        generate_activate_bat(venv_path)
        print_summary(venv_path, args.torch)
    except KeyboardInterrupt:
        warn("用户中断操作。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n总耗时：{elapsed:.1f} 秒")


if __name__ == "__main__":
    main()