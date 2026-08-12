# -*- coding: utf-8 -*-
"""检测报告导出器模块。

将 DetectionResult 转换为结构化 HTML 报告并原子写入文件。
采用"先写临时文件再原子重命名"策略保证写入一致性。
仅依赖 Python 标准库，零额外依赖。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..domain.defect_type_registry import DefectTypeRegistry
from ..domain.models import DetectionResult
from ..infrastructure.exceptions import ReportBuildError

logger = logging.getLogger("glass_bottle_gui")


class ReportExporter:
    """检测报告导出器，组装报告内容并原子写入 HTML 文件。"""

    def __init__(self, registry: DefectTypeRegistry) -> None:
        """初始化报告导出器。

        Args:
            registry: 缺陷类型注册表，用于查询中文名
        """
        self._registry = registry

    def export(
        self,
        result: DetectionResult,
        path: str,
        fmt: str = "html",
    ) -> str:
        """导出检测报告到文件。

        采用原子写入策略：先写 .tmp 临时文件，成功后 os.replace 原子替换。

        Args:
            result: 检测结果对象
            path: 目标文件路径
            fmt: 报告格式（目前仅支持 html）

        Returns:
            实际写入的文件路径

        Raises:
            ReportBuildError: 报告组装失败
            PermissionError: 写入权限不足
        """
        if fmt.lower() != "html":
            raise ReportBuildError(reason=f"不支持的报告格式: {fmt}")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # 组装报告内容
        try:
            content = self._build_content(result)
        except Exception as e:
            logger.error(f"报告内容组装失败: {e}")
            raise ReportBuildError(reason=str(e)) from e

        # 原子写入：先写临时文件再重命名
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(target))
        except PermissionError:
            self._cleanup_tmp(tmp_path)
            raise
        except Exception as e:
            self._cleanup_tmp(tmp_path)
            raise ReportBuildError(reason=f"文件写入失败: {e}") from e

        logger.info(f"报告已导出: {target}")
        return str(target)

    def _build_content(self, result: DetectionResult) -> str:
        """将 DetectionResult 转换为 HTML 报告字符串。

        Args:
            result: 检测结果对象

        Returns:
            HTML 格式报告字符串
        """
        time_str = result.detect_time.strftime("%Y-%m-%d %H:%M:%S")
        defect_status = "含缺陷" if result.has_defect else "无缺陷"

        # 缺陷清单表格行
        if result.defects:
            rows_html = ""
            for idx, defect in enumerate(result.defects, start=1):
                chinese_name = self._registry.get_chinese_name(defect.type_key)
                rows_html += (
                    f"<tr>"
                    f"<td>{idx}</td>"
                    f"<td>{chinese_name}</td>"
                    f"<td>{defect.type_key}</td>"
                    f"<td>{defect.confidence:.2f}</td>"
                    f"</tr>"
                )
        else:
            rows_html = (
                '<tr><td colspan="4" style="text-align:center;color:#999;">'
                "未检测到缺陷</td></tr>"
            )

        # 置信度统计
        avg_conf = result.avg_confidence()
        max_conf = result.max_confidence()
        count = result.defect_count()

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>玻璃瓶缺陷检测报告</title>
    <style>
        body {{ font-family: "Microsoft YaHei", sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .info-item {{ margin: 8px 0; }}
        .info-label {{ font-weight: bold; color: #555; }}
        .status-badge {{
            display: inline-block; padding: 4px 12px; border-radius: 4px;
            color: white; font-weight: bold;
        }}
        .status-defect {{ background-color: #e74c3c; }}
        .status-ok {{ background-color: #27ae60; }}
        .stats {{ margin-top: 15px; }}
        .stats span {{ margin-right: 30px; }}
    </style>
</head>
<body>
    <h1>玻璃瓶缺陷检测报告</h1>

    <h2>基本信息</h2>
    <div class="info-item"><span class="info-label">检测时间：</span>{time_str}</div>
    <div class="info-item"><span class="info-label">图片来源：</span>{result.source}</div>
    <div class="info-item"><span class="info-label">图片名称：</span>{result.image_name}</div>
    <div class="info-item">
        <span class="info-label">检测结果：</span>
        <span class="status-badge {'status-defect' if result.has_defect else 'status-ok'}">{defect_status}</span>
    </div>

    <h2>缺陷清单</h2>
    <table>
        <thead>
            <tr>
                <th>序号</th>
                <th>缺陷类型</th>
                <th>类型键</th>
                <th>置信度</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <h2>置信度统计</h2>
    <div class="stats">
        <span><span class="info-label">缺陷数量：</span>{count}</span>
        <span><span class="info-label">平均置信度：</span>{avg_conf:.2f}</span>
        <span><span class="info-label">最高置信度：</span>{max_conf:.2f}</span>
    </div>

    <hr>
    <p style="color:#999;font-size:12px;">报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>"""
        return html

    @staticmethod
    def _cleanup_tmp(tmp_path: Path) -> None:
        """清理临时文件。"""
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass