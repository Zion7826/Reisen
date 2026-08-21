import subprocess
import sys
import os
import time

python_exe = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"
work_dir = r"D:\demo"

# 启动训练
proc = subprocess.Popen(
    [python_exe, "train_continue.py"],
    cwd=work_dir,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
)

print(f"训练已启动，PID: {proc.pid}")

# 等待几秒钟
time.sleep(10)

# 检查进程是否还在运行
if proc.poll() is None:
    print("训练进程正在运行")
else:
    print(f"进程已退出，返回码: {proc.poll()}")