import subprocess
import time

python_exe = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"

proc = subprocess.Popen(
    [python_exe, "web_dual_detect.py"],
    cwd=r"D:\demo",
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
)

print(f"双模式检测系统已启动，PID: {proc.pid}")
time.sleep(8)

if proc.poll() is None:
    print("进程正在运行")
else:
    print(f"进程已退出，返回码: {proc.poll()}")