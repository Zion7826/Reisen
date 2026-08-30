import requests
import os

url = 'http://localhost:5000/upload'
test_dir = r'D:\bottle\test'
test_images = [
    ('broken_large', '000.png'),
    ('broken_small', '000.png'),
    ('contamination', '000.png'),
    ('good', '000.png'),
]

for cls, fname in test_images:
    fpath = os.path.join(test_dir, cls, fname)
    if not os.path.exists(fpath):
        print(f'{cls}/{fname}: FILE NOT FOUND')
        continue
    with open(fpath, 'rb') as f:
        resp = requests.post(url, files={'files': (fname, f, 'image/png')}, timeout=30)
    data = resp.json()
    if data['success']:
        r = data['results'][0]
        score = r['anomaly_score']
        defective = r['is_defective']
        det_count = r['detection_count']
        counts = r['class_counts']
        print(f'{cls}/{fname}: score={score}, defective={defective}, detections={det_count}, counts={counts}')
    else:
        print(f'{cls}/{fname}: ERROR - {data.get("error")}')