import subprocess
import os
import time

python_exe = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"
app_path = r"D:\dlgd-pharma\app.py"
work_dir = r"D:\dlgd-pharma"

log_file = open(r"D:\dlgd-pharma\flask_out.txt", "w", encoding="utf-8")

proc = subprocess.Popen(
    [python_exe, app_path],
    cwd=work_dir,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    close_fds=True,
)

print(f"Flask started, PID: {proc.pid}")
time.sleep(8)

import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:5000', timeout=5)
    print(f"Status: {r.status}, Content-Length: {len(r.read())} bytes")
except Exception as e:
    print(f"Error: {e}")
