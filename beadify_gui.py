"""拼豆图纸生成器 GUI - Beadify GUI"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
#     "PySide6",
# ]
# ///

import sys
import typing
from pathlib import Path

# Allow importing beadify.py from same directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, Signal, QObject, QThread, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSplitter, QStackedWidget, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QPushButton, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QFileDialog,
    QGroupBox, QStatusBar, QProgressBar, QGridLayout,
)

import beadify


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def pil_to_pixmap(pil_img: Image.Image) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    data = pil_img.convert("RGBA").tobytes()
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# ImagePanel — zoomable image display
# ---------------------------------------------------------------------------

class ImagePanel(QGraphicsView):
    """Zoomable, pannable image display.

    Zoom is clamped so the image can never be smaller than fit-in-view,
    but can be enlarged beyond the display area (scrollbars appear).
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom_level: float = 1.0  # 1.0 = fit-in-view
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet("background: #f0f0f0; border: 1px solid #ccc;")

    def set_image(self, pil_img: Image.Image) -> None:
        pixmap = pil_to_pixmap(pil_img)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom_level = 1.0
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_image(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._zoom_level = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if not self._pixmap_item:
            return
        if event.angleDelta().y() > 0:
            # Zoom in — no upper limit
            self._zoom_level *= 1.15
            self.scale(1.15, 1.15)
        else:
            # Zoom out — clamp at fit-in-view (1.0)
            new_level = self._zoom_level / 1.15
            if new_level <= 1.0:
                self._zoom_level = 1.0
                self.resetTransform()
                self.fitInView(self._pixmap_item,
                               Qt.AspectRatioMode.KeepAspectRatio)
            else:
                self._zoom_level = new_level
                self.scale(1 / 1.15, 1 / 1.15)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        if self._pixmap_item and self._zoom_level <= 1.0:
            self.fitInView(self._pixmap_item,
                           Qt.AspectRatioMode.KeepAspectRatio)


# ---------------------------------------------------------------------------
# BeadEngine — cached pipeline
# ---------------------------------------------------------------------------

class BeadEngine:
    """Wraps the beadify.py pipeline with stage caching."""

    def __init__(self) -> None:
        # Discover available brand color files
        colors_dir = Path(__file__).parent / "colors"
        self.brands: list[tuple[str, str]] = []  # (display_name, json_path)
        if colors_dir.is_dir():
            for p in sorted(colors_dir.glob("*.json")):
                import json as _json
                with open(p, "r", encoding="utf-8") as f:
                    d = _json.load(f)
                name = d.get("brand", p.stem)
                self.brands.append((name, str(p)))
        # Fallback to legacy file
        if not self.brands:
            legacy = Path(__file__).parent / "bead_colors.json"
            if legacy.exists():
                self.brands.append(("Mard", str(legacy)))
        self.json_path = self.brands[0][1] if self.brands else ""
        # Stage 1
        self.bead_lab: np.ndarray | None = None
        self.bead_labels: list[str] = []
        self.label_to_rgb: dict[str, tuple[int, int, int]] = {}
        self._extended: bool = False
        # Stage 2
        self.img_rgba: np.ndarray | None = None
        self.img_pil: Image.Image | None = None
        self.grid_lab: np.ndarray | None = None
        self.grid_mask: np.ndarray | None = None
        self.row_bounds: list[int] = []
        self.col_bounds: list[int] = []
        self._image_path: str = ""
        self._sample_ratio: float = 0.5
        # Stage 3
        self.grid_indices_raw: np.ndarray | None = None
        # Stage 4
        self.grid_indices: np.ndarray | None = None
        self._tolerance: float = 12.0  # hybrid default
        self._metric: str = "hybrid"
        # Stage 5 (border) — stored separately so stage 6 can re-run cleanly
        self.bordered_indices: np.ndarray | None = None
        self.bordered_mask: np.ndarray | None = None
        self.border_flag: np.ndarray | None = None
        # Stage 6 (connection)
        self.final_indices: np.ndarray | None = None
        self.final_mask: np.ndarray | None = None
        self.connect_flag: np.ndarray | None = None
        # Settings
        self._border_enabled: bool = False
        self._border_label: str = "H2"
        self._connect_enabled: bool = False
        self._connect_width: int = 3
        self._cell_size: int = 60
        self._origin: str = "bl"
        self._guide_lines: bool = False

    def _ensure_label(self, label: str) -> None:
        """Ensure a bead label exists in both bead_labels and label_to_rgb."""
        if label not in self.label_to_rgb:
            import json
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            hexval = data["label_to_hex"].get(label)
            if hexval:
                self.label_to_rgb[label] = beadify.hex_to_rgb(hexval)
        if label not in self.bead_labels:
            self.bead_labels.append(label)

    def compute(self, from_stage: int = 1,
                progress_cb: None | typing.Callable = None) -> dict:
        """Run pipeline from given stage. Returns result dict."""
        results: dict = {}

        def _progress(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        if from_stage <= 1:
            _progress("加载豆色…")
            self.bead_lab, self.bead_labels, self.label_to_rgb = \
                beadify.load_bead_colors(self.json_path, self._extended)
            # Ensure border/connection labels are available for rendering
            self._ensure_label("H1")
            self._ensure_label("H2")

        if from_stage <= 2 and self._image_path:
            _progress("加载图片 & 检测网格…")
            self.img_pil = Image.open(self._image_path).convert("RGBA")
            self.img_rgba = np.array(self.img_pil)
            block_size = beadify.detect_block_size(self.img_rgba)
            row_bounds, col_bounds = beadify.detect_grid(self.img_rgba, block_size)
            self.row_bounds = row_bounds
            self.col_bounds = col_bounds
            self.grid_lab, self.grid_mask = beadify.extract_block_colors(
                self.img_rgba, row_bounds, col_bounds,
                sample_ratio=self._sample_ratio)

        if from_stage <= 3 and self.grid_lab is not None:
            _progress("匹配豆色…")
            self.grid_indices_raw, self.grid_lab = beadify.map_to_bead_colors(
                self.grid_lab, self.grid_mask, self.bead_lab, self.bead_labels,
                metric=self._metric)

        if from_stage <= 4 and self.grid_indices_raw is not None:
            _progress(f"合并颜色 (容差={self._tolerance})…")
            if self._tolerance > 0:
                self.grid_indices = beadify.consolidate_colors(
                    self.grid_indices_raw, self.grid_mask, self.grid_lab,
                    self.bead_lab, self.bead_labels, self._tolerance,
                    metric=self._metric)
            else:
                self.grid_indices = self.grid_indices_raw.copy()

        if from_stage <= 5 and self.grid_indices is not None:
            if self._border_enabled:
                _progress("添加包边…")
                self.bordered_indices, self.bordered_mask, self.border_flag = \
                    beadify.add_border(self.grid_indices, self.grid_mask,
                                     self.bead_labels, self._border_label)
            else:
                self.bordered_indices = self.grid_indices.copy()
                self.bordered_mask = self.grid_mask.copy()
                self.border_flag = np.zeros_like(self.grid_mask)

        if from_stage <= 6 and self.bordered_indices is not None:
            if self._connect_enabled:
                _progress("连接主体…")
                self.final_indices, self.final_mask, self.connect_flag = \
                    beadify.connect_bodies(self.bordered_indices, self.bordered_mask,
                                         self.bead_labels, "H1",
                                         self._connect_width)
            else:
                self.final_indices = self.bordered_indices.copy()
                self.final_mask = self.bordered_mask.copy()
                self.connect_flag = np.zeros_like(self.bordered_mask)

        _progress("完成")
        # Collect stats
        if self.grid_indices is not None and self.grid_mask is not None:
            used = set(int(x) for x in self.grid_indices[self.grid_mask])
            results["n_colors"] = len(used)
            results["n_beads"] = int(self.grid_mask.sum())
        if self.final_mask is not None:
            results["n_rows"] = self.final_mask.shape[0]
            results["n_cols"] = self.final_mask.shape[1]
        return results

    def render_all(self) -> dict[str, Image.Image]:
        """Render all output images. Returns dict of name -> PIL Image."""
        imgs: dict[str, Image.Image] = {}
        if self.final_indices is None or self.final_mask is None:
            return imgs

        imgs["color"] = beadify.render_color_image(
            self.final_indices, self.final_mask,
            self.label_to_rgb, self.bead_labels, self._cell_size)
        imgs["pattern"] = beadify.render_output(
            self.final_indices, self.final_mask,
            self.bead_labels, self.label_to_rgb,
            self._cell_size, self._origin,
            guide_lines=self._guide_lines)
        return imgs

    def render_for_tolerance(self, tolerance: float) -> tuple[
        Image.Image, Image.Image, np.ndarray, int
    ]:
        """Render color+pattern images for a specific tolerance.
        Returns (color_img, pattern_img, indices, n_colors)."""
        if self.grid_indices_raw is None:
            raise ValueError("No image loaded")
        if tolerance > 0:
            indices = beadify.consolidate_colors(
                self.grid_indices_raw, self.grid_mask, self.grid_lab,
                self.bead_lab, self.bead_labels, tolerance,
                metric=self._metric)
        else:
            indices = self.grid_indices_raw.copy()
        n_colors = len(set(int(x) for x in indices[self.grid_mask]))
        color_img = beadify.render_color_image(
            indices, self.grid_mask, self.label_to_rgb,
            self.bead_labels, self._cell_size)
        pattern_img = beadify.render_output(
            indices, self.grid_mask, self.bead_labels,
            self.label_to_rgb, self._cell_size, self._origin)
        return color_img, pattern_img, indices, n_colors


# ---------------------------------------------------------------------------
# ComputeWorker — background thread
# ---------------------------------------------------------------------------

class ComputeWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)

    def __init__(self, engine: BeadEngine, from_stage: int):
        super().__init__()
        self.engine = engine
        self.from_stage = from_stage

    @Slot()
    def run(self) -> None:
        try:
            results = self.engine.compute(self.from_stage,
                                          progress_cb=self.progress.emit)
            imgs = self.engine.render_all()
            results["images"] = imgs
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit({"error": str(e)})


# ---------------------------------------------------------------------------
# SettingsPanel
# ---------------------------------------------------------------------------

class SettingsPanel(QWidget):
    settings_changed = Signal(str)  # emits setting name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # File selection
        file_group = QGroupBox("图片")
        fl = QVBoxLayout(file_group)
        self.file_btn = QPushButton("打开图片…")
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        fl.addWidget(self.file_btn)
        fl.addWidget(self.file_label)
        layout.addWidget(file_group)

        # Parameters
        param_group = QGroupBox("参数")
        pl = QFormLayout(param_group)

        self.brand_combo = QComboBox()
        pl.addRow("色号标准:", self.brand_combo)

        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0, 50)
        self.tolerance_spin.setValue(12.0)  # hybrid default
        self.tolerance_spin.setSingleStep(0.5)
        self.tolerance_spin.setDecimals(1)
        pl.addRow("容差:", self.tolerance_spin)

        self.extended_check = QCheckBox("扩展色 (P/Q/R/T/Y/ZG)")
        pl.addRow(self.extended_check)

        self.sample_ratio_spin = QDoubleSpinBox()
        self.sample_ratio_spin.setRange(0.1, 1.0)
        self.sample_ratio_spin.setValue(0.5)
        self.sample_ratio_spin.setSingleStep(0.1)
        self.sample_ratio_spin.setDecimals(1)
        pl.addRow("取色比例:", self.sample_ratio_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["混合 (推荐)", "CIE76 (Lab欧氏)", "CIEDE2000"])
        pl.addRow("色差算法:", self.metric_combo)

        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(20, 120)
        self.cell_size_spin.setValue(60)
        pl.addRow("格子大小:", self.cell_size_spin)

        self.origin_combo = QComboBox()
        self.origin_combo.addItems(["左下", "左上", "右下", "右上"])
        pl.addRow("坐标原点:", self.origin_combo)

        self.guide_lines_check = QCheckBox("辅助计数线")
        pl.addRow(self.guide_lines_check)
        layout.addWidget(param_group)

        # Border & Connection
        bc_group = QGroupBox("包边 & 连接")
        bl = QVBoxLayout(bc_group)

        self.border_check = QCheckBox("启用包边")
        bl.addWidget(self.border_check)
        border_row = QHBoxLayout()
        border_row.addWidget(QLabel("包边颜色:"))
        self.border_color_combo = QComboBox()
        self.border_color_combo.addItems(["H2 (白)", "H1 (透明)"])
        border_row.addWidget(self.border_color_combo)
        bl.addLayout(border_row)

        self.connect_check = QCheckBox("启用连接")
        bl.addWidget(self.connect_check)
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("宽度:"))
        self.connect_width_spin = QSpinBox()
        self.connect_width_spin.setRange(2, 10)
        self.connect_width_spin.setValue(3)
        conn_row.addWidget(self.connect_width_spin)
        bl.addLayout(conn_row)
        layout.addWidget(bc_group)

        # View mode
        view_group = QGroupBox("视图")
        vl = QVBoxLayout(view_group)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["默认视图", "容差探索"])
        vl.addWidget(self.view_combo)
        layout.addWidget(view_group)

        # Export
        self.export_btn = QPushButton("导出图纸…")
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        layout.addStretch()

        # Connect signals
        self.brand_combo.currentIndexChanged.connect(
            lambda: self.settings_changed.emit("brand"))
        self.tolerance_spin.valueChanged.connect(
            lambda: self.settings_changed.emit("tolerance"))
        self.extended_check.toggled.connect(
            lambda: self.settings_changed.emit("extended"))
        self.sample_ratio_spin.valueChanged.connect(
            lambda: self.settings_changed.emit("sample_ratio"))
        self.metric_combo.currentIndexChanged.connect(
            lambda: self.settings_changed.emit("metric"))
        self.cell_size_spin.valueChanged.connect(
            lambda: self.settings_changed.emit("cell_size"))
        self.origin_combo.currentIndexChanged.connect(
            lambda: self.settings_changed.emit("origin"))
        self.guide_lines_check.toggled.connect(
            lambda: self.settings_changed.emit("guide_lines"))
        self.border_check.toggled.connect(
            lambda: self.settings_changed.emit("border"))
        self.border_color_combo.currentIndexChanged.connect(
            lambda: self.settings_changed.emit("border"))
        self.connect_check.toggled.connect(
            lambda: self.settings_changed.emit("connection"))
        self.connect_width_spin.valueChanged.connect(
            lambda: self.settings_changed.emit("connection"))

    @property
    def origin_code(self) -> str:
        return ["bl", "tl", "br", "tr"][self.origin_combo.currentIndex()]

    @property
    def metric_code(self) -> str:
        return ["hybrid", "cie76", "ciede2000"][self.metric_combo.currentIndex()]

    @property
    def border_label(self) -> str:
        return "H2" if self.border_color_combo.currentIndex() == 0 else "H1"


# ---------------------------------------------------------------------------
# DefaultView — 3-panel comparison
# ---------------------------------------------------------------------------

class DefaultView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 2×2 grid using nested splitters
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(top_splitter)
        v_splitter.addWidget(bottom_splitter)

        self.panels: list[tuple[QLabel, ImagePanel]] = []
        titles = ["原图", "网格检测", "拼豆色图", "拼豆图纸"]
        targets = [top_splitter, top_splitter, bottom_splitter, bottom_splitter]
        for title, target in zip(titles, targets):
            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(2, 2, 2, 2)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
            panel = ImagePanel()
            cl.addWidget(lbl)
            cl.addWidget(panel, stretch=1)
            target.addWidget(container)
            self.panels.append((lbl, panel))

        layout.addWidget(v_splitter, stretch=1)

        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setStyleSheet(
            "font-size: 14px; padding: 4px;"
            " background: palette(midlight); color: palette(text);")
        layout.addWidget(self.stats_label)

    def set_original(self, img: Image.Image) -> None:
        self.panels[0][1].set_image(img)

    def set_grid_overlay(self, img: Image.Image) -> None:
        self.panels[1][1].set_image(img)

    def set_color(self, img: Image.Image) -> None:
        self.panels[2][1].set_image(img)

    def set_pattern(self, img: Image.Image) -> None:
        self.panels[3][1].set_image(img)

    def set_stats(self, n_colors: int, n_beads: int,
                  n_cols: int = 0, n_rows: int = 0) -> None:
        parts = [f"颜色: {n_colors}", f"豆数: {n_beads}"]
        if n_cols > 0 and n_rows > 0:
            parts.append(f"豆板: {n_cols}×{n_rows}")
        self.stats_label.setText("  |  ".join(parts))

    def clear(self) -> None:
        for _, panel in self.panels:
            panel.clear_image()
        self.stats_label.setText("")


# ---------------------------------------------------------------------------
# ToleranceView — comparison grid with merge highlighting
# ---------------------------------------------------------------------------

class ToleranceView(QWidget):
    compute_requested = Signal(list)  # list of tolerance values

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("容差值:"))
        self.tol_spins: list[QDoubleSpinBox] = []
        for default_val in [5.0, 10.0, 15.0]:
            spin = QDoubleSpinBox()
            spin.setRange(0, 50)
            spin.setValue(default_val)
            spin.setSingleStep(0.5)
            spin.setDecimals(1)
            ctrl.addWidget(spin)
            self.tol_spins.append(spin)

        self.highlight_check = QCheckBox("高亮融合区域")
        ctrl.addWidget(self.highlight_check)

        self.compute_btn = QPushButton("对比")
        ctrl.addWidget(self.compute_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Grid: 2 rows x 3 cols
        self.grid = QGridLayout()
        self.color_panels: list[ImagePanel] = []
        self.pattern_panels: list[ImagePanel] = []
        self.color_labels: list[QLabel] = []
        self.pattern_labels: list[QLabel] = []

        for i in range(3):
            # Color label + panel
            cl = QLabel("")
            cl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.setStyleSheet("font-weight: bold; font-size: 14px;")
            cp = ImagePanel()
            self.grid.addWidget(cl, 0, i)
            self.grid.addWidget(cp, 1, i)
            self.color_labels.append(cl)
            self.color_panels.append(cp)

            # Pattern label + panel
            pl = QLabel("")
            pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pl.setStyleSheet("font-weight: bold; font-size: 14px;")
            pp = ImagePanel()
            self.grid.addWidget(pl, 2, i)
            self.grid.addWidget(pp, 3, i)
            self.pattern_labels.append(pl)
            self.pattern_panels.append(pp)

        layout.addLayout(self.grid, stretch=1)

        self.compute_btn.clicked.connect(self._on_compute)

    def _on_compute(self) -> None:
        vals = sorted(set(s.value() for s in self.tol_spins))
        self.compute_requested.emit(vals)

    def show_results(self, results: list[tuple[
        float, Image.Image, Image.Image, int
    ]]) -> None:
        """Display comparison results."""
        for i in range(3):
            if i < len(results):
                tol, color_img, pattern_img, n_colors = results[i]
                self.color_labels[i].setText(
                    f"t={tol:.1f}  —  {n_colors} 色")
                self.pattern_labels[i].setText(
                    f"t={tol:.1f}  —  {n_colors} 色")
                self.color_panels[i].set_image(color_img)
                self.pattern_panels[i].set_image(pattern_img)
            else:
                self.color_labels[i].setText("")
                self.pattern_labels[i].setText("")
                self.color_panels[i].clear_image()
                self.pattern_panels[i].clear_image()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("拼豆图纸生成器")
        self.resize(1400, 900)

        self.engine = BeadEngine()
        self._thread: QThread | None = None
        self._worker: ComputeWorker | None = None

        # Layout
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.settings = SettingsPanel()
        self.view_stack = QStackedWidget()
        self.default_view = DefaultView()
        self.tolerance_view = ToleranceView()
        self.view_stack.addWidget(self.default_view)
        self.view_stack.addWidget(self.tolerance_view)

        main_layout.addWidget(self.settings)
        main_layout.addWidget(self.view_stack, stretch=1)
        self.setCentralWidget(central)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Connect signals
        self.settings.file_btn.clicked.connect(self._open_file)
        self.settings.settings_changed.connect(self._on_setting_changed)
        self.settings.view_combo.currentIndexChanged.connect(
            self.view_stack.setCurrentIndex)
        self.settings.export_btn.clicked.connect(self._export)
        self.tolerance_view.compute_requested.connect(
            self._on_tolerance_compare)

        # Populate brand dropdown
        self.settings.brand_combo.blockSignals(True)
        for name, _ in self.engine.brands:
            self.settings.brand_combo.addItem(name)
        self.settings.brand_combo.blockSignals(False)

        # Init bead colors
        self.engine.compute(from_stage=1)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if not path:
            return
        self.settings.file_label.setText(Path(path).name)
        self.engine._image_path = path
        self._run_compute(from_stage=2)

    def _on_setting_changed(self, name: str) -> None:
        if name == "brand":
            idx = self.settings.brand_combo.currentIndex()
            if 0 <= idx < len(self.engine.brands):
                self.engine.json_path = self.engine.brands[idx][1]
                self.engine.compute(from_stage=1)
                if self.engine._image_path:
                    self._run_compute(from_stage=2)
            return

        if not self.engine._image_path:
            return

        if name == "extended":
            self.engine._extended = self.settings.extended_check.isChecked()
            self._run_compute(from_stage=1)
        elif name == "sample_ratio":
            self.engine._sample_ratio = self.settings.sample_ratio_spin.value()
            self._run_compute(from_stage=2)
        elif name == "metric":
            self.engine._metric = self.settings.metric_code
            # Adjust default tolerance for the metric's distance scale
            default_tol = {"cie76": 15.0, "ciede2000": 10.0, "hybrid": 12.0}.get(
                self.engine._metric, 12.0)
            self.settings.tolerance_spin.blockSignals(True)
            self.settings.tolerance_spin.setValue(default_tol)
            self.settings.tolerance_spin.blockSignals(False)
            self.engine._tolerance = default_tol
            self._run_compute(from_stage=3)
        elif name == "tolerance":
            self.engine._tolerance = self.settings.tolerance_spin.value()
            self._run_compute(from_stage=4)
        elif name == "border":
            self.engine._border_enabled = self.settings.border_check.isChecked()
            self.engine._border_label = self.settings.border_label
            self._run_compute(from_stage=5)
        elif name == "connection":
            self.engine._connect_enabled = self.settings.connect_check.isChecked()
            self.engine._connect_width = self.settings.connect_width_spin.value()
            self._run_compute(from_stage=6)
        elif name in ("cell_size", "origin", "guide_lines"):
            self.engine._cell_size = self.settings.cell_size_spin.value()
            self.engine._origin = self.settings.origin_code
            self.engine._guide_lines = self.settings.guide_lines_check.isChecked()
            self._run_compute(from_stage=7)

    def _run_compute(self, from_stage: int) -> None:
        """Run computation in background thread."""
        # If already running, just queue (simplified: block)
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

        self.progress_bar.show()
        self.settings.setEnabled(False)

        self._thread = QThread()
        self._worker = ComputeWorker(self.engine, from_stage)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda msg: self.status_bar.showMessage(msg))
        self._worker.finished.connect(self._on_compute_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_compute_done(self, results: dict) -> None:
        self.progress_bar.hide()
        self.settings.setEnabled(True)

        if "error" in results:
            self.status_bar.showMessage(f"错误: {results['error']}")
            return

        self.settings.export_btn.setEnabled(True)

        # Update default view
        if self.engine.img_pil:
            self.default_view.set_original(self.engine.img_pil)
            # Grid overlay
            if self.engine.row_bounds and self.engine.col_bounds:
                overlay = self.engine.img_pil.convert("RGB").copy()
                draw = ImageDraw.Draw(overlay)
                rb = self.engine.row_bounds
                cb = self.engine.col_bounds
                W, H = overlay.size
                line_color = (255, 0, 0)
                for y in rb:
                    draw.line([(0, y), (W, y)], fill=line_color, width=1)
                for x in cb:
                    draw.line([(x, 0), (x, H)], fill=line_color, width=1)
                self.default_view.set_grid_overlay(overlay)

        imgs = results.get("images", {})
        if "color" in imgs:
            self.default_view.set_color(imgs["color"])
        if "pattern" in imgs:
            self.default_view.set_pattern(imgs["pattern"])

        n_colors = results.get("n_colors", 0)
        n_beads = results.get("n_beads", 0)
        n_cols = results.get("n_cols", 0)
        n_rows = results.get("n_rows", 0)
        self.default_view.set_stats(n_colors, n_beads, n_cols, n_rows)
        self.status_bar.showMessage(
            f"完成 — {n_colors} 色, {n_beads} 豆, 豆板 {n_cols}×{n_rows}")

    def _on_tolerance_compare(self, tolerances: list[float]) -> None:
        if not self.engine._image_path or self.engine.grid_indices_raw is None:
            return

        self.progress_bar.show()
        self.settings.setEnabled(False)
        self.status_bar.showMessage("正在计算容差对比…")

        QApplication.processEvents()

        results = []
        all_indices = []
        for tol in sorted(tolerances):
            try:
                color_img, pattern_img, indices, n_colors = \
                    self.engine.render_for_tolerance(tol)
                results.append((tol, color_img, pattern_img, n_colors))
                all_indices.append(indices)
            except Exception as e:
                self.status_bar.showMessage(f"容差 t={tol} 出错: {e}")

        # Apply merge highlighting if enabled
        highlight = self.tolerance_view.highlight_check.isChecked()
        if highlight and len(all_indices) >= 2:
            base_indices = all_indices[0]  # lowest tolerance as reference
            cell_size = self.engine._cell_size
            for i in range(1, len(results)):
                tol, color_img, pattern_img, n_colors = results[i]
                # Color image: no offset
                color_img = beadify.render_merge_highlight(
                    color_img, base_indices, all_indices[i],
                    self.engine.grid_mask, cell_size)
                # Pattern image: has coordinate margins
                n_rows, n_cols = self.engine.grid_mask.shape
                margin_lr, margin_tb = beadify.get_pattern_margins(
                    n_rows, n_cols, cell_size)
                pattern_img = beadify.render_merge_highlight(
                    pattern_img, base_indices, all_indices[i],
                    self.engine.grid_mask, cell_size,
                    ox=margin_lr, oy=margin_tb)
                results[i] = (tol, color_img, pattern_img, n_colors)

        self.tolerance_view.show_results(results)
        self.progress_bar.hide()
        self.settings.setEnabled(True)
        self.status_bar.showMessage("容差对比完成")

    def _export(self) -> None:
        if self.engine.final_indices is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图纸", "", "PNG (*.png)")
        if not path:
            return
        imgs = self.engine.render_all()
        if "pattern" in imgs:
            imgs["pattern"].save(path)
            self.status_bar.showMessage(f"已导出: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
