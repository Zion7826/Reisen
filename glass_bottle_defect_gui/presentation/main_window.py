# -*- coding: utf-8 -*-
"""主窗口"""

import os
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QGroupBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

import cv2
import pandas as pd

from glass_bottle_defect_gui.service.detection_service import DetectionService
from glass_bottle_defect_gui.controller.detection_worker import DetectionWorker
from glass_bottle_defect_gui.controller.camera_worker import CameraWorker
from glass_bottle_defect_gui.domain.models import DetectionResponse
from glass_bottle_defect_gui.config import RESULTS_DIR
from glass_bottle_defect_gui.infrastructure.logger import logger


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.service = DetectionService()
        self.camera_worker = None
        self.detection_worker = None
        self.current_image_path = None
        self.current_response = None
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("玻璃瓶缺陷检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 主部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 左侧面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # 右侧面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def create_left_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 图片检测组
        image_group = QGroupBox("图片检测")
        image_layout = QVBoxLayout()
        image_group.setLayout(image_layout)

        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_image)
        image_layout.addWidget(self.upload_btn)

        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.detect_image)
        self.detect_btn.setEnabled(False)
        image_layout.addWidget(self.detect_btn)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        image_layout.addWidget(self.save_btn)

        layout.addWidget(image_group)

        # 摄像头检测组
        camera_group = QGroupBox("摄像头检测")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)

        self.camera_btn = QPushButton("打开摄像头")
        self.camera_btn.setCheckable(True)
        self.camera_btn.clicked.connect(self.toggle_camera)
        camera_layout.addWidget(self.camera_btn)

        layout.addWidget(camera_group)

        # 置信度阈值
        conf_group = QGroupBox("置信度阈值")
        conf_layout = QVBoxLayout()
        conf_group.setLayout(conf_layout)

        self.conf_label = QLabel("0.25")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_slider)

        layout.addWidget(conf_group)

        # 检测结果统计
        stats_group = QGroupBox("检测结果")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.stats_label = QLabel("等待检测...")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # 导出报告
        export_group = QGroupBox("导出报告")
        export_layout = QVBoxLayout()
        export_group.setLayout(export_layout)

        self.export_btn = QPushButton("导出检测报告")
        self.export_btn.clicked.connect(self.export_report)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        layout.addWidget(export_group)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 选项卡
        self.tab_widget = QTabWidget()

        # 图片显示选项卡
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        self.image_label.setText("请上传图片")
        self.tab_widget.addTab(self.image_label, "图片显示")

        # 检测详情选项卡
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(5)
        self.detail_table.setHorizontalHeaderLabels(["序号", "类别", "中文名", "置信度", "位置"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tab_widget.addTab(self.detail_table, "检测详情")

        # 日志选项卡
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.tab_widget.addTab(self.log_text, "运行日志")

        layout.addWidget(self.tab_widget)
        return panel

    def update_conf_label(self):
        """更新置信度标签"""
        value = self.conf_slider.value() / 100
        self.conf_label.setText(f"{value:.2f}")

    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.current_image_path = file_path
            self.load_and_display_image(file_path)
            self.detect_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.stats_label.setText("等待检测...")
            self.detail_table.setRowCount(0)
            self.log_message(f"已上传图片: {file_path}")

    def load_and_display_image(self, image_path: str):
        """加载并显示图片"""
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def detect_image(self):
        """检测图片"""
        if not self.current_image_path:
            return

        self.detect_btn.setEnabled(False)
        self.log_message("开始检测...")

        conf_threshold = self.conf_slider.value() / 100
        self.detection_worker = DetectionWorker(
            self.current_image_path, self.service, conf_threshold
        )
        self.detection_worker.finished.connect(self.on_detection_finished)
        self.detection_worker.error.connect(self.on_detection_error)
        self.detection_worker.start()

    def on_detection_finished(self, response: DetectionResponse):
        """检测完成回调"""
        self.current_response = response
        self.detect_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        # 更新统计信息
        if response.total_count > 0:
            stats_text = f"检测到 {response.total_count} 个缺陷\n"
            for class_name, count in response.class_counts.items():
                stats_text += f"{class_name}: {count} 个\n"
            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("未检测到缺陷")

        # 更新详情表格
        self.update_detail_table(response)

        # 绘制检测结果
        image = self.service.load_image(self.current_image_path)
        result_image = self.service.draw_detections(image, response)
        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        self.log_message(f"检测完成，共检测到 {response.total_count} 个缺陷")

    def on_detection_error(self, error_msg: str):
        """检测错误回调"""
        self.detect_btn.setEnabled(True)
        QMessageBox.critical(self, "检测错误", error_msg)
        self.log_message(f"检测失败: {error_msg}")

    def update_detail_table(self, response: DetectionResponse):
        """更新详情表格"""
        self.detail_table.setRowCount(len(response.detections))
        for row, det in enumerate(response.detections):
            self.detail_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.detail_table.setItem(row, 1, QTableWidgetItem(det.class_name))
            self.detail_table.setItem(row, 2, QTableWidgetItem(det.class_name_cn))
            self.detail_table.setItem(row, 3, QTableWidgetItem(f"{det.confidence:.2f}"))
            bbox_str = f"({det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]})"
            self.detail_table.setItem(row, 4, QTableWidgetItem(bbox_str))

    def save_result(self):
        """保存检测结果"""
        if not self.current_response or not self.current_image_path:
            return

        image = self.service.load_image(self.current_image_path)
        result_image = self.service.draw_detections(image, self.current_response)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RESULTS_DIR / f"result_{timestamp}.jpg"
        cv2.imwrite(str(save_path), result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        QMessageBox.information(self, "保存成功", f"结果已保存至: {save_path}")
        self.log_message(f"结果已保存: {save_path}")

    def toggle_camera(self):
        """切换摄像头"""
        if self.camera_btn.isChecked():
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """启动摄像头"""
        self.camera_btn.setText("关闭摄像头")
        self.upload_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)

        conf_threshold = self.conf_slider.value() / 100
        self.camera_worker = CameraWorker(0, self.service, conf_threshold)
        self.camera_worker.frame_ready.connect(self.on_camera_frame)
        self.camera_worker.detection_ready.connect(self.on_camera_detection)
        self.camera_worker.error.connect(self.on_camera_error)
        self.camera_worker.start()

        self.log_message("摄像头已启动")

    def stop_camera(self):
        """停止摄像头"""
        self.camera_btn.setText("打开摄像头")
        self.upload_btn.setEnabled(True)
        self.detect_btn.setEnabled(self.current_image_path is not None)

        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None

        self.log_message("摄像头已关闭")

    def on_camera_frame(self, qt_image: QImage):
        """摄像头帧回调"""
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def on_camera_detection(self, response: DetectionResponse):
        """摄像头检测回调"""
        if response.total_count > 0:
            stats_text = f"检测到 {response.total_count} 个缺陷\n"
            for class_name, count in response.class_counts.items():
                stats_text += f"{class_name}: {count} 个\n"
            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("未检测到缺陷")

    def on_camera_error(self, error_msg: str):
        """摄像头错误回调"""
        QMessageBox.critical(self, "摄像头错误", error_msg)
        self.stop_camera()
        self.log_message(f"摄像头错误: {error_msg}")

    def export_report(self):
        """导出检测报告"""
        if not self.current_response:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = RESULTS_DIR / f"detection_report_{timestamp}.xlsx"

            # 准备数据
            data = []
            for det in self.current_response.detections:
                data.append({
                    "序号": len(data) + 1,
                    "类别": det.class_name,
                    "中文名": det.class_name_cn,
                    "置信度": f"{det.confidence:.2f}",
                    "位置": f"({det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]})"
                })

            df = pd.DataFrame(data)
            df.to_excel(report_path, index=False)

            QMessageBox.information(self, "导出成功", f"报告已导出至: {report_path}")
            self.log_message(f"报告已导出: {report_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出报告失败: {e}")
            self.log_message(f"导出报告失败: {e}")

    def log_message(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """关闭事件"""
        if self.camera_worker:
            self.camera_worker.stop()
        event.accept()