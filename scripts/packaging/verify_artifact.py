# -*- coding: utf-8 -*-
"""独立产物验证脚本。

对已构建的打包产物单独执行验证，无需重新打包。

用法：
    python scripts/packaging/verify_artifact.py --artifact dist/glass_bottle_defect
    python scripts/packaging/verify_artifact.py --artifact dist/glass_bottle_defect.exe
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config.pack_config import PackConfig
from .config.validation_report import ValidationReport
from .services.artifact_validator import ArtifactValidator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("packaging")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="打包产物独立验证脚本")
    parser.add_argument("--artifact", required=True, help="待验证的 exe 或产物目录路径")
    parser.add_argument("--name", default="glass_bottle_defect", help="产物名称")
    return parser.parse_args()


def verify_all(artifact_path: Path, name: str) -> ValidationReport:
    """执行完整验证。

    Args:
        artifact_path: 产物路径
        name: 产物名称

    Returns:
        ValidationReport 验证报告
    """
    config = PackConfig.default()
    config = __import__("dataclasses").replace(config, name=name)

    validator = ArtifactValidator(config)
    return validator.validate(artifact_path)


def main() -> int:
    """主入口。

    Returns:
        退出码（0=全部通过，1=存在失败）
    """
    args = parse_args()
    artifact_path = Path(args.artifact)

    if not artifact_path.exists():
        logger.error(f"产物路径不存在: {artifact_path}")
        return 1

    report = verify_all(artifact_path, args.name)

    if report.overall_passed:
        logger.info("验证全部通过")
        return 0
    else:
        logger.warning("验证未完全通过，请检查报告")
        return 1


if __name__ == "__main__":
    sys.exit(main())