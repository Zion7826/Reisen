import requests
import os
import zipfile

repos = [
    ('andreabazerla/anomaly-detection', 'main'),
    ('sachiin044/InspAIct', 'main'),
    ('BimsaraS99/industrial-anomaly-detection-autoencoder', 'main'),
]

base_dir = r'D:\dlgd-pharma\data'
os.makedirs(base_dir, exist_ok=True)

for repo, branch in repos:
    url = f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}'
    repo_name = repo.split('/')[1]
    save_path = os.path.join(base_dir, f'{repo_name}.zip')
    print(f'下载 {repo}...')
    try:
        resp = requests.get(url, timeout=120, stream=True, verify=False)
        if resp.status_code == 200:
            total = 0
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            print(f'  完成: {total / 1024 / 1024:.1f} MB')
            
            try:
                with zipfile.ZipFile(save_path, 'r') as z:
                    names = z.namelist()
                    img_count = sum(1 for n in names if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
                    print(f'  图片数: {img_count}, 总文件数: {len(names)}')
                    z.extractall(os.path.join(base_dir, repo_name))
            except Exception as e:
                print(f'  解压失败: {e}')
        else:
            print(f'  HTTP {resp.status_code}')
    except Exception as e:
        print(f'  失败: {e}')
    print()