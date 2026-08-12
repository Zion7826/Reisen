# -*- coding: utf-8 -*-
"""主窗口模块。

构建"工具栏 + 原图区 + 结果图区 + 缺陷信息区 + 状态栏"五区布局，
串联图片上传→检测→渲染→表格流程，以及摄像头实时检测与报告导出。
实现检测状态机：Idle → ImageLoaded → Detecting → ResultShown。
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..controller.camera_worker import CameraWorker
from ..controller.detection_worker import DetectionWorker
from ..controller.image_upload_controller import ImageUploadController
from ..domain.models import DetectionResult
from ..infrastructure.error_handler import ErrorHandler
from ..service.detection_service import DetectionService
from ..service.report_exporter import ReportExporter
from .result_renderer import ResultRenderer

logger = logging.getLogger("glass_bottle_gui")


class DetectState(Enum):
    """检测状态机枚举。"""
    IDLE = auto()
    IMAGE_LOADED = auto()
    DETECTING = auto()
    RESULT_SHOWN = auto()


class MainWindow(QMainWindow):
    """玻璃瓶缺陷检测系统主窗口。"""

    def __init__(
        self,
        detection_service: DetectionService,
        report_exporter: ReportExporter,
        result_renderer: ResultRenderer,
        image_upload_controller: ImageUploadController,
        error_handler: ErrorHandler,
    ) -> None:
        """初始化主窗口并注入依赖。

        Args:
            detection_service: 检测服务
            report_exporter: 报告导出器
            result_renderer: 结果渲染器
            image_upload_controller: 图片上传控制器
            error_handler: 异常处理器
        """
        super().__init__()
        self._service = detection_service
        self._report_exporter = report_exporter
        self._renderer = result_renderer
        self._upload_controller = image_upload_controller
        self._error_handler = error_handler

        # 状态
        self._state = DetectState.IDLE
        self._current_image: Optional[np.ndarray] = None
        self._current_image_name: str = ""
        self._last_result: Optional[DetectionResult] = None
        self._detection_worker: Optional[DetectionWorker] = None
        self._camera_worker: Optional[CameraWorker] = None
        self._camera_active = False

        self._setup_ui()
        self._update_button_states()

    # ----------------------------------------------------------
    # UI 构建
    # ----------------------------------------------------------
    def _setup_ui(self) -> None:
        """构建五区布局。"""
        self.setWindowTitle("玻璃瓶缺陷检测系统")
        self.setMinimumSize(1000, 700)

        # 工具栏
        self._setup_toolbar()

        # 中央内容区
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 对比区：原图 + 结果图 左右并列
        compare_layout = QHBoxLayout()
        self._original_label = QLabel("原图")
        self._original_label.setAlignment(Qt.AlignCenter)
        self._original_label.setMinimumSize(480, 360)
        self._original_label.setStyleSheet("border: 1px solid #ccc; background: #f9f9f9;")

        self._result_label = QLabel("检测结果")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setMinimumSize(480, 360)
        self._result_label.setStyleSheet("border: 1px solid #ccc; background: #f9f9f9;")

        compare_layout.addWidget(self._original_label)
        compare_layout.addWidget(self._result_label)
        main_layout.addLayout(compare_layout)

        # 缺陷信息表格
        self._defect_table = QTableWidget(0, 3)
        self._defect_table.setHorizontalHeaderLabels(["序号", "缺陷类型", "置信度"])
        self._defect_table.horizontalHeader().setStretchLastSection(True)
        self._defect_table.setFixedHeight(200)
        main_layout.addWidget(self._defect_table)

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self.show_status("就绪")

    def _setup_toolbar(self) -> None:
        """构建工具栏。"""
        toolbar = QToolBar("工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 上传图片按钮
        self._upload_action = QAction("上传图片", self)
        self._upload_action.triggered.connect(self.on_upload_clicked)
        toolbar.addAction(self._upload_action)

        toolbar.addSeparator()

        # 启动检测按钮
        self._detect_action = QAction("启动检测", self)
        self._detect_action.triggered.connect(self.on_detect_clicked)
        toolbar.addAction(self._detect_action)

        toolbar.addSeparator()

        # 摄像头设备索引下拉
        toolbar.addWidget(QLabel(" 摄像头: "))
        self._camera_combo = QComboBox()
        for i in range(4):
            self._camera_combo.addItem(f"摄像头 {i}", i)
        toolbar.addWidget(self._camera_combo)

        # 启动摄像头按钮
        self._camera_start_action = QAction("启动摄像头", self)
        self._camera_start_action.triggered.connect(self.on_camera_start_clicked)
        toolbar.addAction(self._camera_start_action)

        # 停止摄像头按钮
        self._camera_stop_action = QAction("停止摄像头", self)
        self._camera_stop_action.triggered.connect(self.on_camera_stop_clicked)
        toolbar.addAction(self._camera_stop_action)

        toolbar.addSeparator()

        # 导出报告按钮
        self._export_action = QAction("导出报告", self)
        self._export_action.triggered.connect(self.on_export_clicked)
        toolbar.addAction(self._export_action)

    # ----------------------------------------------------------
    # 状态机与按钮联动
    # ----------------------------------------------------------
    def _set_state(self, state: DetectState) -> None:
        """更新状态机并联动按钮。"""
        self._state = state
        self._update_button_states()

    def _update_button_states(self) -> None:
        """根据当前状态更新按钮启用/禁用。"""
        # 摄像头模式下的特殊处理
        if self._camera_active:
            self._upload_action.setEnabled(False)
            self._detect_action.setEnabled(False)
            self._camera_start_action.setEnabled(False)
            self._camera_stop_action.setEnabled(True)
            self._export_action.setEnabled(False)
            self._camera_combo.setEnabled(False)
            return

        # 正常模式
        self._camera_stop_action.setEnabled(False)
        self._camera_combo.setEnabled(True)

        if self._state == DetectState.IDLE:
            self._upload_action.setEnabled(True)
            self._detect_action.setEnabled(False)
            self._camera_start_action.setEnabled(True)
            self._export_action.setEnabled(False)
        elif self._state == DetectState.IMAGE_LOADED:
            self._upload_action.setEnabled(True)
            self._detect_action.setEnabled(True)
            self._camera_start_action.setEnabled(True)
            self._export_action.setEnabled(False)
        elif self._state == DetectState.DETECTING:
            self._upload_action.setEnabled(False)
            self._detect_action.setEnabled(False)
            self._camera_start_action.setEnabled(False)
            self._export_action.setEnabled(False)
        elif self._state == DetectState.RESULT_SHOWN:
            self._upload_action.setEnabled(True)
            self._detect_action.setEnabled(True)
            self._camera_start_action.setEnabled(True)
            self._export_action.setEnabled(True)

    # ----------------------------------------------------------
    # 事件处理：图片上传
    # ----------------------------------------------------------
    def on_upload_clicked(self) -> None:
        """处理上传图片点击：选择→校验→加载→展示原图。"""
        path = self._upload_controller.select_image(self)
        if path is None:
            return

        try:
            self._upload_controller.validate_format(path)
        except Exception as e:
            self._error_handler.handle(e, self)
            return

        try:
            image = self._upload_controller.load_image(path)
        except Exception as e:
            self._error_handler.handle(e, self)
            return

        self._current_image = image
        self._current_image_name = path.split("/")[-1].split("\\")[-1]
        self._last_result = None

        # 展示原图预览
        pixmap = self._renderer._ndarray_to_qpixmap(image)
        self._original_label.setPixmap(pixmap)
        self._result_label.setText("点击「启动检测」开始检测")
        self._result_label.clear()

        self._set_state(DetectState.IMAGE_LOADED)
        self.show_status(f"已加载图片: {self._current_image_name}")

    # ----------------------------------------------------------
    # 事件处理：单张图片检测
    # ----------------------------------------------------------
    def on_detect_clicked(self) -> None:
        """处理启动检测点击：创建 DetectionWorker 异步执行。"""
        if self._current_image is None:
            return

        self._set_state(DetectState.DETECTING)
        self.show_status("检测中...")
        self._error_handler.log_operation("检测启动", {"image": self._current_image_name})

        self._detection_worker = DetectionWorker(
            self._service,
            self._current_image,
            "上传",
            self._current_image_name,
        )
        self._detection_worker.result_ready.connect(self.update_result_view)
        self._detection_worker.error_occurred.connect(self._on_detect_error)
        self._detection_worker.start()

    def _on_detect_error(self, message: str) -> None:
        """检测失败回调：提示并恢复状态。"""
        self._error_handler.handle_message(message, self)
        self._set_state(DetectState.IMAGE_LOADED)
        self.show_status("检测失败")

    # ----------------------------------------------------------
    # 事件处理：摄像头
    # ----------------------------------------------------------
    def on_camera_start_clicked(self) -> None:
        """处理启动摄像头点击。"""
        camera_index = self._camera_combo.currentData()
        self._camera_active = True
        self._update_button_states()
        self.show_status(f"摄像头启动中... (索引 {camera_index})")
        self._error_handler.log_operation("摄像头启动", {"index": camera_index})

        self._camera_worker = CameraWorker(
            self._service,
            camera_index=camera_index,
            target_fps=config.TARGET_FPS,
        )
        self._camera_worker.frame_ready.connect(self.on_camera_frame)
        self._camera_worker.error_occurred.connect(self._on_camera_error)
        self._camera_worker.start()

    def on_camera_stop_clicked(self) -> None:
        """处理停止摄像头点击。"""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker = None

        self._camera_active = False
        self._set_state(DetectState.IDLE)
        self.show_status("摄像头已停止")
        self._original_label.setText("原图")
        self._original_label.clear()
        self._result_label.setText("检测结果")
        self._result_label.clear()
        self._defect_table.setRowCount(0)

    def on_camera_frame(self, frame: np.ndarray, result: DetectionResult) -> None:
        """摄像头帧回调：刷新对比区与缺陷表格。"""
        self._current_image = frame
        self._current_image_name = result.image_name
        self._last_result = result
        self.update_result_view(result)

    def _on_camera_error(self, message: str) -> None:
        """摄像头错误回调。"""
        self._error_handler.handle_message(message, self)
        if self._camera_worker is not None:
            self._camera_worker = None
        self._camera_active = False
        self._set_state(DetectState.IDLE)
        self.show_status("摄像头已停止")

    # ----------------------------------------------------------
    # 事件处理：报告导出
    # ----------------------------------------------------------
    def on_export_clicked(self) -> None:
        """处理导出报告点击。"""
        if self._last_result is None:
            self._error_handler.handle_message("暂无检测结果可导出", self)
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出检测报告", "检测报告.html", config.REPORT_FILE_FILTER
        )
        if not path:
            return

        self._error_handler.log_operation("报告导出", {"path": path})

        try:
            saved_path = self._report_exporter.export(self._last_result, path)
            self.show_status(f"报告导出成功: {saved_path}")
            QMessageBox.information(self, "提示", f"报告已导出至:\n{saved_path}")
        except Exception as e:
            self._error_handler.handle(e, self)

    # ----------------------------------------------------------
    # 结果展示
    # ----------------------------------------------------------
    def update_result_view(self, result: DetectionResult) -> None:
        """刷新对比区与缺陷表格。

        Args:
            result: 检测结果
        """
        self._last_result = result

        if self._current_image is not None:
            try:
                orig_pixmap, result_pixmap = self._renderer.render_comparison(
                    self._current_image, result.defects
                )
                self._original_label.setPixmap(orig_pixmap)
                self._result_label.setPixmap(result_pixmap)
            except Exception as e:
                logger.error(f"结果渲染失败: {e}")
                self._error_handler.handle_message("结果渲染失败，请重试", self)
                return

        self.update_defect_table(result.defects)

        if not self._camera_active:
            self._set_state(DetectState.RESULT_SHOWN)
            if result.has_defect:
                self.show_status(
                    f"检测完成: 发现 {result.defect_count()} 个缺陷"
                )
            else:
                self.show_status("检测完成: 未发现缺陷")

    def update_defect_table(self, defects: list) -> None:
        """刷新缺陷信息表格。

        Args:
            defects: DefectItem 列表
        """
        self._defect_table.setRowCount(0)

        if not defects:
            self._defect_table.setRowCount(1)
            item = QTableWidgetItem("未检测到缺陷")
            self._defect_table.setItem(0, 0, item)
            self._defect_table.setSpan(0, 0, 1, 3)
            return

        self._defect_table.setRowCount(len(defects))
        for row, defect in enumerate(defects):
            chinese_name = self._renderer._registry.get_chinese_name(defect.type_key)
            self._defect_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self._defect_table.setItem(row, 1, QTableWidgetItem(chinese_name))
            self._defect_table.setItem(row, 2, QTableWidgetItem(f"{defect.confidence:.2f}"))

    def show_status(self, text: str) -> None:
        """状态栏反馈。"""
        self._status_bar.showMessage(text)

    # ----------------------------------------------------------
    # 窗口关闭清理
    # ----------------------------------------------------------
    def closeEvent(self, event) -> None:
        """窗口关闭时清理摄像头资源。"""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker.wait(2000)
            self._camera_worker = None
        if self._detection_worker is not None and self._detection_worker.isRunning():
            self._detection_worker.wait(2000)
        super().closeEvent(event)