import subprocess
import time

python_exe = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"

proc = subprocess.Popen(
    [python_exe, "app.py"],
    cwd=r"D:\dlgd-pharma",
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
)

print(f"dlgd-pharma 已启动，PID: {proc.pid}")
time.sleep(8)

if proc.poll() is None:
    print("进程正在运行")
else:
    print(f"进程已退出，返回码: {proc.poll()}")