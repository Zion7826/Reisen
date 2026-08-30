import requests
import os
import zipfile
import io

mirrors = [
    'https://github.com/lonlonago/Glass-Bottle-Defects-Identification-Dataset-VOC-YOLO-Format-with-2716-Images-in-3-Categories/archive/refs/heads/main.zip',
    'https://codeload.github.com/lonlonago/Glass-Bottle-Defects-Identification-Dataset-VOC-YOLO-Format-with-2716-Images-in-3-Categories/zip/refs/heads/main',
    'https://hub.fastgit.xyz/lonlonago/Glass-Bottle-Defects-Identification-Dataset-VOC-YOLO-Format-with-2716-Images-in-3-Categories/archive/refs/heads/main.zip',
]

save_path = r'D:\dlgd-pharma\data\bottle_defects_2716.zip'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

for i, url in enumerate(mirrors):
    print(f'尝试镜像 {i+1}: {url[:80]}...')
    try:
        resp = requests.get(url, timeout=60, stream=True, verify=False)
        if resp.status_code == 200:
            total = 0
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        if total % (1024*1024) < 8192:
                            print(f'  下载中: {total / 1024 / 1024:.1f} MB')
            print(f'  下载完成: {total / 1024 / 1024:.1f} MB')
            break
        else:
            print(f'  HTTP {resp.status_code}')
    except Exception as e:
        print(f'  失败: {e}')
else:
    print('所有镜像都失败')