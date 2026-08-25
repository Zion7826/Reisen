import subprocess
import sys
import os
import time

python_exe = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"
train_script = r"D:\dlgd-pharma\train\train_unetpp.py"
work_dir = r"D:\dlgd-pharma"

log_file = open(r"D:\dlgd-pharma\train\unetpp_train_log.txt", "w", encoding="utf-8")

proc = subprocess.Popen(
    [python_exe, train_script],
    cwd=work_dir,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    close_fds=True,
)

print(f"UNet++ training started, PID: {proc.pid}")

with open(r"D:\dlgd-pharma\train\unetpp_pid.txt", "w") as f:
    f.write(str(proc.pid))
