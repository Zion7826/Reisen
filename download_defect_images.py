# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷图片自动采集脚本
从百度图片搜索下载缺陷图片，用于训练数据集准备

用法：
    python download_defect_images.py --defect crack --count 20
    python download_defect_images.py --defect bubble --count 20
    python download_defect_images.py --defect scratch --count 20
    python download_defect_images.py --defect stain --count 20

注意：
    - 百度图片有反爬机制，需要带 User-Agent 和 Referer
    - 下载的图片仅供学习研究使用
    - 图片下载后需人工筛选和标注
"""

import argparse
import os
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

# 百度图片搜索配置
BAIDU_IMAGE_SEARCH_URL = "https://image.baidu.com/search/index"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://image.baidu.com/",
}

# 缺陷类型关键词映射
DEFECT_KEYWORDS = {
    "crack": "玻璃瓶 裂纹 破裂",
    "bubble": "玻璃瓶 气泡 气孔",
    "scratch": "玻璃瓶 划痕 刮痕",
    "stain": "玻璃瓶 污渍 污染 污点",
}


def download_image(url: str, save_path: str, timeout: int = 10) -> bool:
    """下载单张图片"""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                # 检查是否为图片
                content_type = response.headers.get("Content-Type", "")
                if "image" in content_type or url.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    with open(save_path, "wb") as f:
                        f.write(response.read())
                    return True
        return False
    except Exception as e:
        print(f"  [失败] {url[:60]}... -> {e}")
        return False


def search_baidu_images(keyword: str, max_count: int = 20) -> list:
    """从百度图片搜索获取图片URL列表（简化版，直接构造搜索URL）"""
    # 注意：百度图片的真实图片URL藏在JSON数据里，需要解析
    # 这里用简化方案：直接访问搜索页，解析HTML中的图片URL
    image_urls = []

    # 构造搜索URL
    params = {
        "tn": "baiduimage",
        "word": keyword,
        "pn": 0,  # 起始位置
        "rn": max_count,  # 每页数量
    }
    search_url = f"{BAIDU_IMAGE_SEARCH_URL}?{urllib.parse.urlencode(params)}"

    print(f"搜索关键词: {keyword}")
    print(f"搜索URL: {search_url}")

    try:
        req = urllib.request.Request(search_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="ignore")

            # 方法1：解析 data-imgurl 属性（百度图片的真实URL）
            pattern1 = r'data-imgurl=["\']([^"\']+)["\']'
            urls1 = re.findall(pattern1, html)
            image_urls.extend(urls1)

            # 方法2：解析 objURL（原图URL）
            pattern2 = r'"objURL":"([^"]+)"'
            urls2 = re.findall(pattern2, html)
            image_urls.extend(urls2)

            # 方法3：解析 thumbURL（缩略图URL）
            pattern3 = r'"thumbURL":"([^"]+)"'
            urls3 = re.findall(pattern3, html)
            image_urls.extend(urls3)

        print(f"找到 {len(image_urls)} 个候选图片URL")
        return list(set(image_urls))[:max_count]  # 去重并限制数量

    except Exception as e:
        print(f"搜索失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="玻璃瓶缺陷图片采集工具")
    parser.add_argument(
        "--defect",
        type=str,
        required=True,
        choices=["crack", "bubble", "scratch", "stain"],
        help="缺陷类型",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="下载数量（默认20张）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\demo\datasets\glass_bottle\images\raw",
        help="输出目录",
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取关键词
    keyword = DEFECT_KEYWORDS.get(args.defect, args.defect)
    print(f"\n{'=' * 60}")
    print(f"缺陷类型: {args.defect}")
    print(f"搜索关键词: {keyword}")
    print(f"目标数量: {args.count} 张")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 60}\n")

    # 搜索图片
    image_urls = search_baidu_images(keyword, args.count * 2)  # 多搜一些，因为部分会下载失败

    if not image_urls:
        print("未找到任何图片URL")
        return

    # 下载图片
    success_count = 0
    for i, url in enumerate(image_urls):
        if success_count >= args.count:
            break

        # 处理URL（百度图片URL可能有转义）
        url = url.replace("\\/", "/")

        # 生成文件名
        ext = ".jpg"
        if ".png" in url:
            ext = ".png"
        elif ".jpeg" in url:
            ext = ".jpeg"

        filename = f"{args.defect}_{success_count + 1:03d}{ext}"
        save_path = output_dir / filename

        print(f"[{i + 1}/{len(image_urls)}] 下载: {url[:80]}...")

        if download_image(url, str(save_path)):
            success_count += 1
            print(f"  [成功] 保存到 {filename} ({success_count}/{args.count})")
        else:
            print(f"  [跳过]")

        # 短暂延迟，避免请求过快
        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"下载完成: 成功 {success_count} 张，保存到 {output_dir}")
    print(f"注意：下载的图片需要人工筛选和标注")
    print(f"下一步：使用 labelImg 或其他标注工具进行标注")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
