# -*- coding: utf-8 -*-
"""PyInstaller 打包主编排脚本。

按状态机顺序串联：生成配置 → 执行打包 → 组织分发 → 验证产物。
支持 onedir/onefile 双模式，命令行参数可覆盖默认配置。

用法：
    python scripts/packaging/build_exe.py                           # 默认 onedir 模式
    python scripts/packaging/build_exe.py --mode onefile            # onefile 模式
    python scripts/packaging/build_exe.py --name myapp --skip-verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config.build_result import BuildResult
from .config.pack_config import PackConfig, PackMode
from .exceptions import (
    BuildFailureError,
    DistributionError,
    SpecGenerationError,
    ValidationError,
)
from .services.artifact_validator import ArtifactValidator
from .services.build_runner import BuildRunner
from .services.distribution_organizer import DistributionOrganizer
from .services.spec_generator import SpecGenerator

logger = logging.getLogger("packaging")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PyInstaller 打包主编排脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["onedir", "onefile"], default="onedir",
        help="打包模式 (默认: onedir)",
    )
    parser.add_argument("--name", default="glass_bottle_defect", help="产物名称")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--data-yaml", type=str, default=None, help="数据配置路径")
    parser.add_argument("--skip-verify", action="store_true", help="跳过产物验证")
    parser.add_argument("--skip-compress", action="store_true", help="跳过压缩")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PackConfig:
    """根据命令行参数构造 PackConfig。"""
    config = PackConfig.default()
    mode = PackMode.ONEDIR if args.mode == "onedir" else PackMode.ONEFILE
    overrides = {
        "mode": mode,
        "name": args.name,
        "skip_verify": args.skip_verify,
        "skip_compress": args.skip_compress,
    }
    if args.model:
        overrides["model_path"] = Path(args.model)
    if args.data_yaml:
        overrides["data_yaml_path"] = Path(args.data_yaml)
    overrides["spec_file_path"] = config.entry_script.parent.parent / f"{args.name}.spec"
    return replace(config, **overrides)


def run_pipeline(config: PackConfig) -> BuildResult:
    """执行打包主流程。

    Args:
        config: 打包配置

    Returns:
        BuildResult 构建结果
    """
    logger.info("=" * 60)
    logger.info("PyInstaller 打包流程启动")
    logger.info(f"模式: {config.mode.value} | 名称: {config.name}")
    logger.info("=" * 60)

    # 阶段 1：生成 spec 配置
    logger.info("[1/4] 生成 spec 配置...")
    spec_generator = SpecGenerator(config)
    spec_file = spec_generator.generate()

    # 阶段 2：执行打包
    logger.info("[2/4] 执行 PyInstaller 打包...")
    build_runner = BuildRunner(config)
    result = build_runner.run(spec_file)

    # 阶段 3：组织分发
    logger.info("[3/4] 组织绿色分发...")
    organizer = DistributionOrganizer(config)
    dist_path = organizer.organize(result.artifact_path)
    logger.info(f"分发包: {dist_path}")

    # 阶段 4：验证产物
    if not config.skip_verify:
        logger.info("[4/4] 验证产物...")
        validator = ArtifactValidator(config)
        report = validator.validate(result.artifact_path)
        if not report.overall_passed:
            logger.warning("产物验证未完全通过，请检查验证报告")
    else:
        logger.info("[4/4] 跳过产物验证")

    logger.info("=" * 60)
    logger.info(f"打包完成！产物: {dist_path}")
    logger.info("=" * 60)

    return result


def main() -> int:
    """主入口。

    Returns:
        退出码（0=成功，1=spec错误，2=构建失败，3=分发失败，4=验证失败）
    """
    args = parse_args()

    try:
        config = build_config(args)
    except Exception as e:
        logger.error(f"参数构造失败: {e}")
        return 1

    try:
        run_pipeline(config)
        return 0
    except (FileNotFoundError, SpecGenerationError) as e:
        logger.error(f"spec 生成失败: {e}")
        return 1
    except BuildFailureError as e:
        logger.error(f"打包构建失败: {e}")
        return 2
    except DistributionError as e:
        logger.error(f"分发组织失败: {e}")
        return 3
    except ValidationError as e:
        logger.error(f"产物验证失败: {e}")
        return 4
    except Exception as e:
        logger.error(f"未预期异常: {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())