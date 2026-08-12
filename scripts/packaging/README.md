# PyInstaller 打包工具

将 PyQt5 GUI 程序打包为 Windows 绿色免安装 exe 可执行文件。

## 前置条件

- Python 3.11
- 已安装全部运行时依赖（PyQt5、ultralytics、torch、opencv-python、numpy）
- 已安装 PyInstaller（`pip install pyinstaller==6.11.0`）
- 入口文件 `glass_bottle_defect_gui/main.py` 存在

## 打包命令

```bash
# onedir 模式（默认，启动快，便于排错）
python scripts/packaging/build_exe.py

# onefile 模式（单文件，启动稍慢）
python scripts/packaging/build_exe.py --mode onefile

# 自定义参数
python scripts/packaging/build_exe.py --name myapp --model path/to/best.pt --skip-verify
```

## 产物位置

- onedir 模式：`dist/glass_bottle_defect/`（目录）+ `dist/glass_bottle_defect.zip`（压缩包）
- onefile 模式：`dist/glass_bottle_defect.exe` + `dist/glass_bottle_defect.zip`

## 独立验证

```bash
python scripts/packaging/verify_artifact.py --artifact dist/glass_bottle_defect
```

## 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 启动报 ModuleNotFoundError | 隐藏导入遗漏 | 在 `config/hidden_imports.py` 追加模块名 |
| Qt 插件缺失，界面无法显示 | PyQt5 平台插件未打包 | 检查 `dependency_collector.py` 的插件路径 |
| 模型加载失败，进入降级模式 | best.pt 未打包或路径错误 | 确认权重文件存在，检查 `sys._MEIPASS` 路径解析 |
| 磁盘空间不足 | 打包需要较大临时空间 | 清理 build/ 目录，确保磁盘剩余 > 2GB |
| 中文路径报错 | 路径含特殊字符 | 尽量使用英文路径 |