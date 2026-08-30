import requests
import os

url = 'http://localhost:5000/upload'
test_dir = r'D:\bottle\test'

tests = [
    ('broken_large', '000.png'),
    ('broken_large', '001.png'),
    ('broken_large', '002.png'),
    ('broken_small', '000.png'),
    ('broken_small', '001.png'),
    ('broken_small', '002.png'),
    ('contamination', '000.png'),
    ('contamination', '001.png'),
    ('contamination', '002.png'),
    ('good', '000.png'),
    ('good', '001.png'),
    ('good', '002.png'),
]

print(f'{"真实类别":<15} {"图片":<10} {"分数":>8} {"缺陷":>6} {"检测类别":<25} {"特征"}')
print('-' * 100)

for cls, fname in tests:
    fpath = os.path.join(test_dir, cls, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'rb') as f:
        resp = requests.post(url, files={'files': (fname, f, 'image/png')}, timeout=30)
    data = resp.json()
    if data['success']:
        r = data['results'][0]
        score = r['anomaly_score']
        defective = r['is_defective']
        counts = r['class_counts']
        detected = ', '.join(f'{k}:{v}' for k, v in counts.items() if v > 0) or '无'
        print(f'{cls:<15} {fname:<10} {score:>8.2f} {"是" if defective else "否":>6} {detected:<25}')