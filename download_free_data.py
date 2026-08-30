import requests
import os

datasets = [
    ('BTAD', 'http://avires.dimi.uniud.it/papers/btad/btad.zip', r'D:\dlgd-pharma\data\btad.zip'),
    ('VisA', 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar', r'D:\dlgd-pharma\data\VisA.tar'),
]

for name, url, save_path in datasets:
    print(f'尝试下载 {name}...')
    try:
        resp = requests.get(url, timeout=30, stream=True, verify=False)
        if resp.status_code == 200:
            total = int(resp.headers.get('content-length', 0))
            print(f'  文件大小: {total/1024/1024:.1f} MB')
            downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (10*1024*1024) < 1024*1024:
                            print(f'  下载中: {downloaded/1024/1024:.0f} MB / {total/1024/1024:.0f} MB')
            print(f'  完成: {downloaded/1024/1024:.1f} MB')
        else:
            print(f'  HTTP {resp.status_code}')
    except Exception as e:
        print(f'  失败: {e}')
    print()