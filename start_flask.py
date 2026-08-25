import subprocess
import sys
import time
import os

python_path = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"
app_path = r"D:\dlgd-pharma\app.py"
work_dir = r"D:\dlgd-pharma"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

proc = subprocess.Popen(
    [python_path, app_path],
    cwd=work_dir,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
)

print(f"Flask PID: {proc.pid}")
time.sleep(12)

import urllib.request
try:
    resp = urllib.request.urlopen("http://localhost:5000/", timeout=10)
    print(f"HTTP Status: {resp.status}")
except Exception as e:
    print(f"Connection Error: {e}")