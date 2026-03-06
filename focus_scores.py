#!/usr/bin/env python3
"""
focus_scores.py

Evaluate focus-chart sharpness and lighting/exposure consistency across manually
recorded imaging conditions such as f-number and flash power.

Notes:
- Image timing is parsed from the filename epoch token.
- Aperture, flash power, and related test conditions come from manual metadata.
- No EXIF metadata is required or used.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set, Callable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class ProcessingConfig:
    """Configuration for image processing."""
    csv_path: Path
    polygons_path: Path
    metric: str
    tz: str = "America/Chicago"
    preview_max: int = 1600
    draw: bool = False
    recompute: bool = False
    target_polygons: int = 4
    save_plots: bool = False
    plot_dir: Path = Path("focus_scores_plots")
    yolo_model_path: Optional[Path] = None
    ignore_downscaled_images: bool = False

@dataclass
class ImageMetadata:
    """Metadata derived from filename only."""
    file_name: str
    epoch_raw: Optional[int] = None
    datetime_utc: str = ""


@dataclass
class RegionLightingMetrics:
    """Lighting / tonal metrics for a masked region."""
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    p1_intensity: float
    p99_intensity: float
    dynamic_range: float
    shadow_clip_pct: float
    highlight_clip_pct: float
    rms_contrast: float


@dataclass
class FocusResult:
    """Result of focus computation on one image."""
    metadata: ImageMetadata
    metric: str
    full_image_score: float
    polygon_scores: List[float]
    avg_score: float
    polygon_score_min: float
    polygon_score_max: float
    polygon_score_std: float
    polygon_score_range: float
    full_image_lighting: RegionLightingMetrics
    polygon_lighting: List[RegionLightingMetrics]
    polygons_file: str
    preview_max: int
    run_time_local: str
    comments: str = ""


# ----------------------------
# Logging setup
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------
# Focus Metrics Registry
# ----------------------------

FocusMetricFunc = Callable[[np.ndarray, np.ndarray], float]

class FocusMetrics:
    """Registry for focus metrics."""
    
    _metrics: Dict[str, FocusMetricFunc] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a focus metric."""
        def decorator(func: FocusMetricFunc):
            cls._metrics[name.lower()] = func
            return func
        return decorator
    
    @classmethod
    def compute(cls, gray: np.ndarray, mask: np.ndarray, metric: str) -> float:
        """Compute a focus metric."""
        metric_lower = metric.lower().strip()
        if metric_lower not in cls._metrics:
            available = ', '.join(cls._metrics.keys())
            raise ValueError(f"Unknown metric: {metric!r}. Available: {available}")
        return cls._metrics[metric_lower](gray, mask)
    
    @classmethod
    def available_metrics(cls) -> List[str]:
        """Get list of available metrics."""
        return list(cls._metrics.keys())


@FocusMetrics.register("laplacian_var")
def focus_laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    """Variance of Laplacian within masked region (higher ~ sharper)."""
    vals = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0]
    if vals.size < 10:
        return float("nan")
    return float(vals.var())


@FocusMetrics.register("tenengrad")
def focus_tenengrad(gray: np.ndarray, mask: np.ndarray) -> float:
    """Tenengrad (mean gradient magnitude squared) within mask (higher ~ sharper)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    vals = mag2[mask > 0]
    if vals.size < 10:
        return float("nan")
    return float(vals.mean())


@FocusMetrics.register("brenner")
def focus_brenner(gray: np.ndarray, mask: np.ndarray) -> float:
    """Brenner gradient (sum of squared differences with offset)."""
    # Compute horizontal differences with offset of 2
    shifted = np.roll(gray, -2, axis=1)
    diff = (gray.astype(float) - shifted.astype(float)) ** 2
    vals = diff[mask > 0]
    if vals.size < 10:
        return float("nan")
    return float(vals.sum())


@FocusMetrics.register("normalized_variance")
def focus_normalized_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    """Normalized variance of pixel intensities."""
    vals = gray[mask > 0].astype(float)
    if vals.size < 10:
        return float("nan")
    mean = vals.mean()
    if mean == 0:
        return float("nan")
    return float(vals.var() / mean)

# ----------------------------
# YOLO OBB Detection
# ----------------------------

def detect_polygons_with_yolo(
    image_bgr: np.ndarray,
    model_path: Path,
    conf: float = 0.40,
    iou: float = 0.05
) -> List[List[Tuple[int, int]]]:
    """Detect focus charts using YOLO OBB model."""
    model = YOLO(str(model_path))
    
    # Run inference (save=False since we just need coordinates)
    results = model.predict(image_bgr, save=False, imgsz=960, conf=conf, iou=iou, verbose=False)
    
    if len(results) == 0 or results[0].obb is None:
        raise RuntimeError("No focus charts detected by YOLO model")
    
    # Extract oriented bounding boxes (4 corner points each)
    obb_coords = results[0].obb.xyxyxyxy.cpu().numpy()  # Shape: (N, 4, 2)
    
    # Convert to polygon format
    polygons = []
    for obb in obb_coords:
        poly = [(int(x), int(y)) for x, y in obb]
        polygons.append(poly)
    
    logger.info(f"YOLO detected {len(polygons)} focus chart(s)")
    return polygons

# ----------------------------
# Lighting metrics
# ----------------------------

def compute_region_lighting_metrics(
    gray: np.ndarray,
    mask: np.ndarray,
    shadow_threshold: int = 5,
    highlight_threshold: int = 250,
) -> RegionLightingMetrics:
    """Compute brightness / clipping metrics within a masked region."""
    vals = gray[mask > 0].astype(np.float64)
    if vals.size < 10:
        nan = float("nan")
        return RegionLightingMetrics(
            mean_intensity=nan,
            median_intensity=nan,
            std_intensity=nan,
            min_intensity=nan,
            max_intensity=nan,
            p1_intensity=nan,
            p99_intensity=nan,
            dynamic_range=nan,
            shadow_clip_pct=nan,
            highlight_clip_pct=nan,
            rms_contrast=nan,
        )

    p1 = float(np.percentile(vals, 1))
    p99 = float(np.percentile(vals, 99))
    mean_val = float(vals.mean())
    return RegionLightingMetrics(
        mean_intensity=mean_val,
        median_intensity=float(np.median(vals)),
        std_intensity=float(vals.std()),
        min_intensity=float(vals.min()),
        max_intensity=float(vals.max()),
        p1_intensity=p1,
        p99_intensity=p99,
        dynamic_range=float(p99 - p1),
        shadow_clip_pct=float(((vals <= shadow_threshold).mean()) * 100.0),
        highlight_clip_pct=float(((vals >= highlight_threshold).mean()) * 100.0),
        rms_contrast=float(vals.std()),
    )


# ----------------------------
# Metadata Matching
# ----------------------------

def load_manual_metadata(csv_path: Path) -> List[Dict[str, Any]]:
    """Load manual metadata table with time windows."""
    import csv
    from datetime import datetime, timezone
    
    metadata = []
    with csv_path.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse start/end times (UTC) with AM/PM
            date_str = row['Date'].strip()
            start_str = row['Start time (UTC)'].strip()
            end_str = row['End Time (UTC)'].strip()
            
            start_dt = datetime.strptime(f"{date_str} {start_str}", "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(f"{date_str} {end_str}", "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=timezone.utc)
            
            metadata.append({
                'start_epoch': int(start_dt.timestamp()),
                'end_epoch': int(end_dt.timestamp()),
                'manual_focus': row.get('Focus', '').strip(),
                'manual_aperture': row.get('Aperture', '').strip(),
                'manual_flash_power': row.get('Flash power', '').strip(),
                'camera_height_cm': row.get('Camera height (cm)', '').strip(),
                'focus_card_height_agl': row.get('Focus Card Height (AGL)', '').strip(),
                'comment': row.get('Comment', '').strip(),
            })
    
    logger.info(f"Loaded {len(metadata)} manual metadata entries")
    return metadata


def match_manual_metadata(epoch_val: Optional[int], manual_data: List[Dict]) -> Dict[str, str]:
    """Match image epoch to manual metadata time window."""
    if epoch_val is None:
        return {}
    
    # Handle milliseconds
    if epoch_val > 10_000_000_000:
        epoch_val = epoch_val // 1000
    
    logger.debug(f"Matching epoch {epoch_val} against manual metadata")
    for entry in manual_data:
        if entry['start_epoch'] <= epoch_val <= entry['end_epoch']:
            return {
                'manual_focus': entry['manual_focus'],
                'manual_aperture': entry['manual_aperture'],
                'manual_flash_power': entry['manual_flash_power'],
                'camera_height_cm': entry['camera_height_cm'],
                'focus_card_height_agl': entry['focus_card_height_agl'],
                'comment': entry['comment'],
            }
        else:
            logger.debug(f"Epoch {epoch_val} not in range {entry['start_epoch']} - {entry['end_epoch']}")
    
    return {}

# ----------------------------
# Filename parsing
# ----------------------------

def parse_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract epoch timestamp from filename pattern: prefix_EPOCH_suffix."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])  # second token is the epoch
    except ValueError:
        return None


def epoch_to_datetime_str(epoch_val: Optional[int]) -> str:
    """Convert epoch to UTC datetime string in 12-hour format."""
    if epoch_val is None:
        return ""
    if epoch_val > 10_000_000_000:  # milliseconds
        ts = epoch_val / 1000.0
    else:
        ts = float(epoch_val)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %I:%M:%S %p UTC")


def extract_image_metadata(image_path: Path, tz: str) -> ImageMetadata:
    """Extract metadata from the filename epoch only."""
    epoch_val = parse_epoch_from_filename(image_path.name)
    dt_str = epoch_to_datetime_str(epoch_val)

    return ImageMetadata(
        file_name=str(image_path),
        epoch_raw=epoch_val,
        datetime_utc=dt_str,
    )


# ----------------------------
# Image scaling utilities
# ----------------------------

def make_preview(image_bgr: np.ndarray, preview_max: int) -> Tuple[np.ndarray, float]:
    """
    Create a downscaled preview that fits within preview_max (max of width/height).
    Returns (preview_image, scale_factor) where preview = full * scale_factor.
    """
    h, w = image_bgr.shape[:2]
    m = max(h, w)
    if m <= preview_max:
        return image_bgr.copy(), 1.0
    scale = preview_max / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    preview = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return preview, scale


def scale_polygon(poly: List[Tuple[int, int]], sx: float, sy: float) -> List[Tuple[int, int]]:
    """Scale polygon coordinates."""
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in poly]


def clamp_polygon(poly: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[int, int]]:
    """Clamp polygon points into [0,w-1]x[0,h-1]."""
    return [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for x, y in poly]


def polygons_to_masks(
    image_shape_hw: Tuple[int, int], 
    polygons: List[List[Tuple[int, int]]]
) -> List[np.ndarray]:
    """Convert polygons to binary masks."""
    h, w = image_shape_hw
    masks = []
    for poly in polygons:
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        masks.append(mask)
    return masks


# ----------------------------
# Polygon selection UI
# ----------------------------

@dataclass
class PolygonUIState:
    """State for interactive polygon drawing."""
    image: np.ndarray
    window_name: str
    polygons: List[List[Tuple[int, int]]]
    current: List[Tuple[int, int]]
    target_polygons: int
    done: bool


def _draw_overlay(state: PolygonUIState) -> np.ndarray:
    """Draw current polygon state on image."""
    vis = state.image.copy()

    # Draw completed polygons
    for i, poly in enumerate(state.polygons):
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = poly[0]
            cv2.putText(vis, f"P{i+1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw current polygon in progress
    if len(state.current) >= 1:
        for p in state.current:
            cv2.circle(vis, p, 4, (0, 200, 255), -1)
        if len(state.current) >= 2:
            pts = np.array(state.current, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(0, 200, 255), thickness=2)

    # Draw instructions
    lines = [
        f"Define {state.target_polygons} polygons | Completed: {len(state.polygons)}/{state.target_polygons}",
        "Left-click: add vertex | Right-click: finish polygon (>=3 points)",
        "Backspace: remove last vertex | r: reset ALL | q/ESC: quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26

    return vis


def _mouse_callback(event, x, y, flags, param):
    """Mouse callback for polygon drawing."""
    state: PolygonUIState = param
    if state.done:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        state.current.append((int(x), int(y)))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(state.current) >= 3:
            state.polygons.append(state.current.copy())
            state.current.clear()
            if len(state.polygons) >= state.target_polygons:
                state.done = True


def collect_polygons_interactive(
    preview_bgr: np.ndarray, 
    target_polygons: int = 4
) -> List[List[Tuple[int, int]]]:
    """Interactive UI to draw polygons on preview image."""
    window = "Draw Polygons (Preview)"
    state = PolygonUIState(
        image=preview_bgr,
        window_name=window,
        polygons=[],
        current=[],
        target_polygons=target_polygons,
        done=False,
    )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _mouse_callback, state)

    while True:
        vis = _draw_overlay(state)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):  # ESC or q
            break
        if key == ord("r"):  # Reset
            state.polygons.clear()
            state.current.clear()
            state.done = False
        if key == 8:  # Backspace
            if state.current:
                state.current.pop()
        if state.done:
            break

    cv2.destroyWindow(window)

    if len(state.polygons) != target_polygons:
        raise RuntimeError(
            f"Polygon definition cancelled or incomplete. "
            f"Needed {target_polygons}, got {len(state.polygons)}."
        )
    
    return state.polygons


# ----------------------------
# Polygon persistence
# ----------------------------

def save_polygons(path: Path, payload: Dict[str, Any]) -> None:
    """Save polygon data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_polygons_payload(path: Path) -> Dict[str, Any]:
    """Load polygon data from JSON file."""
    return json.loads(path.read_text())


def validate_polygons_payload(payload: Dict[str, Any], required_count: int = 4) -> None:
    """Validate polygon payload structure."""
    if "polygons_preview" not in payload:
        raise ValueError("polygons file missing 'polygons_preview'")
    
    polys = payload["polygons_preview"]
    if not isinstance(polys, list) or len(polys) != required_count:
        raise ValueError(f"Expected 'polygons_preview' to be a list of {required_count} polygons")
    
    for poly in polys:
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError(f"Invalid polygon: {poly}")

    for k in ("preview_shape_hw", "full_shape_hw", "scale_preview_from_full"):
        if k not in payload:
            raise ValueError(f"polygons file missing '{k}'")




def summarize_polygon_scores(scores: List[float]) -> Dict[str, float]:
    """Summarize polygon focus scores for consistency analysis."""
    arr = np.array(scores, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        nan = float("nan")
        return {
            "avg": nan,
            "min": nan,
            "max": nan,
            "std": nan,
            "range": nan,
        }

    return {
        "avg": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "std": float(finite.std()),
        "range": float(finite.max() - finite.min()),
    }

# ----------------------------
# Visualization
# ----------------------------

def create_focus_plot(
    preview_bgr: np.ndarray,
    polygons_preview: List[List[Tuple[int, int]]],
    result: FocusResult,
    output_path: Path,
    manual_meta: Dict[str, str] = None  # NEW parameter
) -> None:
    """Create a visualization plot showing the image with polygons and focus scores."""
    
    if manual_meta is None:
        manual_meta = {}

    # Convert BGR to RGB for matplotlib
    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    
    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Display image
    ax.imshow(preview_rgb)
    ax.axis('off')
    
    # Color palette for polygons (distinct colors)
    colors = ['#00FF00', '#FF00FF', '#00FFFF', '#FFFF00', 
              '#FF6600', '#FF0066', '#6600FF', '#00FF66']
    
    # Draw polygons with scores
    for i, (poly, score) in enumerate(zip(polygons_preview, result.polygon_scores)):
        color = colors[i % len(colors)]
        
        # Draw polygon outline
        poly_array = np.array(poly + [poly[0]], dtype=np.int32)  # Close the polygon
        polygon_patch = mpatches.Polygon(
            poly_array,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            linestyle='-',
            alpha=0.9
        )
        ax.add_patch(polygon_patch)
        
        # Calculate centroid for label placement
        poly_arr = np.array(poly)
        cx = int(poly_arr[:, 0].mean())
        cy = int(poly_arr[:, 1].mean())
        
        # Draw score label with background
        score_text = f"P{i+1}: {score:.1f}" if not np.isnan(score) else f"P{i+1}: N/A"
        ax.text(
            cx, cy,
            score_text,
            fontsize=14,
            fontweight='bold',
            color=color,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='black',
                edgecolor=color,
                alpha=0.7,
                linewidth=2
            )
        )
        
        # Draw polygon number at first vertex
        x0, y0 = poly[0]
        ax.plot(x0, y0, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
    
    # Prepare metadata text
    metadata_lines = [
        f"File: {Path(result.metadata.file_name).name}",
        f"Metric: {result.metric}",
        f"Avg Score: {result.avg_score:.2f}" if not np.isnan(result.avg_score) else "Avg Score: N/A",
        f"Full Image Score: {result.full_image_score:.2f}" if not np.isnan(result.full_image_score) else "Full Image Score: N/A",
        f"Min/Max Poly: {result.polygon_score_min:.2f} / {result.polygon_score_max:.2f}" if not np.isnan(result.polygon_score_min) else "Min/Max Poly: N/A",
        f"Poly Std: {result.polygon_score_std:.2f}" if not np.isnan(result.polygon_score_std) else "Poly Std: N/A",
        f"Poly Range: {result.polygon_score_range:.2f}" if not np.isnan(result.polygon_score_range) else "Poly Range: N/A",
        f"Full Mean: {result.full_image_lighting.mean_intensity:.1f}" if not np.isnan(result.full_image_lighting.mean_intensity) else "Full Mean: N/A",
        f"Full Clip Low/High: {result.full_image_lighting.shadow_clip_pct:.2f}% / {result.full_image_lighting.highlight_clip_pct:.2f}%" if not np.isnan(result.full_image_lighting.shadow_clip_pct) else "Full Clip Low/High: N/A",
    ]

    if manual_meta.get('manual_focus'):
        metadata_lines.append(f"Focus (manual): {manual_meta['manual_focus']}")
    if manual_meta.get('manual_aperture'):
        metadata_lines.append(f"Aperture (manual): f/{manual_meta['manual_aperture']}")
    if manual_meta.get('manual_flash_power'):
        metadata_lines.append(f"Flash (manual): {manual_meta['manual_flash_power']}")
    if manual_meta.get('camera_height_cm'):
        metadata_lines.append(f"Camera Height (cm): {manual_meta['camera_height_cm']}")
    if manual_meta.get('focus_card_height_agl'):
        metadata_lines.append(f"Card Height AGL: {manual_meta['focus_card_height_agl']}")
    if result.metadata.datetime_utc:
        metadata_lines.append(f"DateTime: {result.metadata.datetime_utc}")
    
    metadata_text = '\n'.join(metadata_lines)
    
    # Add metadata text box in upper left
    ax.text(
        0.02, 0.98,
        metadata_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='black',
            edgecolor='white',
            alpha=0.85,
            linewidth=2
        ),
        color='white',
        family='monospace'
    )
    
    # Add title
    title = f"Focus Analysis: {result.metadata.file_name}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)
    
    logger.debug(f"Saved plot to {output_path}")


# ----------------------------
# CSV Management
# ----------------------------

CSV_HEADERS = [
    "file_name",
    "epoch_raw",
    "datetime_utc",
    "manual_focus",
    "manual_aperture",
    "manual_flash_power",
    "camera_height_cm",
    "focus_card_height_agl",
    "comment",
    "metric",
    "full_image_score",
    "avg_score",
    "polygon_score_min",
    "polygon_score_max",
    "polygon_score_std",
    "polygon_score_range",
    "poly1_score",
    "poly2_score",
    "poly3_score",
    "poly4_score",
    "full_mean_intensity",
    "full_median_intensity",
    "full_std_intensity",
    "full_min_intensity",
    "full_max_intensity",
    "full_p1_intensity",
    "full_p99_intensity",
    "full_dynamic_range",
    "full_shadow_clip_pct",
    "full_highlight_clip_pct",
    "full_rms_contrast",
    "poly1_mean_intensity",
    "poly1_std_intensity",
    "poly1_shadow_clip_pct",
    "poly1_highlight_clip_pct",
    "poly1_dynamic_range",
    "poly2_mean_intensity",
    "poly2_std_intensity",
    "poly2_shadow_clip_pct",
    "poly2_highlight_clip_pct",
    "poly2_dynamic_range",
    "poly3_mean_intensity",
    "poly3_std_intensity",
    "poly3_shadow_clip_pct",
    "poly3_highlight_clip_pct",
    "poly3_dynamic_range",
    "poly4_mean_intensity",
    "poly4_std_intensity",
    "poly4_shadow_clip_pct",
    "poly4_highlight_clip_pct",
    "poly4_dynamic_range",
    "polygons_file",
    "preview_max",
    "run_time_local",
    "comments",
]


def result_to_csv_row(result: FocusResult, manual_meta: Dict[str, str] = None) -> Dict[str, Any]:
    """Convert FocusResult to CSV row."""
    if manual_meta is None:
        manual_meta = {}

    row = {
        "file_name": result.metadata.file_name,
        "epoch_raw": "" if result.metadata.epoch_raw is None else str(result.metadata.epoch_raw),
        "datetime_utc": result.metadata.datetime_utc,
        "manual_focus": manual_meta.get('manual_focus', ''),
        "manual_aperture": manual_meta.get('manual_aperture', ''),
        "manual_flash_power": manual_meta.get('manual_flash_power', ''),
        "camera_height_cm": manual_meta.get('camera_height_cm', ''),
        "focus_card_height_agl": manual_meta.get('focus_card_height_agl', ''),
        "comment": manual_meta.get('comment', ''),
        "metric": result.metric,
        "full_image_score": result.full_image_score,
        "avg_score": result.avg_score,
        "polygon_score_min": result.polygon_score_min,
        "polygon_score_max": result.polygon_score_max,
        "polygon_score_std": result.polygon_score_std,
        "polygon_score_range": result.polygon_score_range,
        "poly1_score": result.polygon_scores[0] if len(result.polygon_scores) > 0 else "",
        "poly2_score": result.polygon_scores[1] if len(result.polygon_scores) > 1 else "",
        "poly3_score": result.polygon_scores[2] if len(result.polygon_scores) > 2 else "",
        "poly4_score": result.polygon_scores[3] if len(result.polygon_scores) > 3 else "",
        "full_mean_intensity": result.full_image_lighting.mean_intensity,
        "full_median_intensity": result.full_image_lighting.median_intensity,
        "full_std_intensity": result.full_image_lighting.std_intensity,
        "full_min_intensity": result.full_image_lighting.min_intensity,
        "full_max_intensity": result.full_image_lighting.max_intensity,
        "full_p1_intensity": result.full_image_lighting.p1_intensity,
        "full_p99_intensity": result.full_image_lighting.p99_intensity,
        "full_dynamic_range": result.full_image_lighting.dynamic_range,
        "full_shadow_clip_pct": result.full_image_lighting.shadow_clip_pct,
        "full_highlight_clip_pct": result.full_image_lighting.highlight_clip_pct,
        "full_rms_contrast": result.full_image_lighting.rms_contrast,
        "polygons_file": result.polygons_file,
        "preview_max": result.preview_max,
        "run_time_local": result.run_time_local,
        "comments": result.comments,
    }

    for idx in range(4):
        metrics = result.polygon_lighting[idx] if idx < len(result.polygon_lighting) else None
        row[f"poly{idx + 1}_mean_intensity"] = metrics.mean_intensity if metrics else ""
        row[f"poly{idx + 1}_std_intensity"] = metrics.std_intensity if metrics else ""
        row[f"poly{idx + 1}_shadow_clip_pct"] = metrics.shadow_clip_pct if metrics else ""
        row[f"poly{idx + 1}_highlight_clip_pct"] = metrics.highlight_clip_pct if metrics else ""
        row[f"poly{idx + 1}_dynamic_range"] = metrics.dynamic_range if metrics else ""

    return row


def append_result_to_csv(csv_path: Path, result: FocusResult, manual_meta: Dict[str, str] = None) -> None:
    """Append a single result to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    row = result_to_csv_row(result, manual_meta)
    
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_keys(csv_path: Path) -> Set[Tuple[str, str, str]]:
    """
    Load existing computation keys from CSV to avoid recomputation.
    Key: (file_name, metric, polygons_file)
    """
    if not csv_path.exists():
        return set()

    keys: Set[Tuple[str, str, str]] = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = (r.get("file_name") or "").strip()
            metric = (r.get("metric") or "").strip()
            polyfile = (r.get("polygons_file") or "").strip()
            if fn and metric:
                keys.add((fn, metric, polyfile))
    
    return keys


# ----------------------------
# Core Image Processing
# ----------------------------

def compute_focus_for_image(
    image_path: Path,
    config: ProcessingConfig,
    force_draw: bool = False,
    manual_meta: Dict[str, str] = None  # NEW parameter
) -> FocusResult:
    """
    Compute focus scores for a single image.
    
    Args:
        image_path: Path to image file
        config: Processing configuration
        force_draw: Force polygon drawing UI (overrides config.draw)
    
    Returns:
        FocusResult with all computed data
    """
    # Load image
    full_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if full_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    full_h, full_w = full_bgr.shape[:2]

    # Create preview for polygon selection
    preview_bgr, scale_preview_from_full = make_preview(full_bgr, config.preview_max)
    prev_h, prev_w = preview_bgr.shape[:2]

    # Load or create polygons
    if config.yolo_model_path:
        # Use YOLO detection
        polygons_preview = detect_polygons_with_yolo(preview_bgr, config.yolo_model_path)
        if len(polygons_preview) != config.target_polygons:
            logger.warning(f"YOLO detected {len(polygons_preview)} charts, expected {config.target_polygons}")
    else:
        # Manual polygon drawing
        should_draw = force_draw or config.draw or (not config.polygons_path.exists())
    
        if should_draw:
            polygons_preview = collect_polygons_interactive(preview_bgr, target_polygons=config.target_polygons)
            payload = {
                "polygons_preview": polygons_preview,
                "preview_shape_hw": [prev_h, prev_w],
                "full_shape_hw": [full_h, full_w],
                "scale_preview_from_full": scale_preview_from_full,
                "created_time_local": datetime.now(ZoneInfo(config.tz)).isoformat(),
            }
            save_polygons(config.polygons_path, payload)
        else:
            payload = load_polygons_payload(config.polygons_path)
            validate_polygons_payload(payload, required_count=config.target_polygons)
            polygons_preview = [[(int(x), int(y)) for x, y in poly] for poly in payload["polygons_preview"]]

    # Scale polygons from preview to full resolution
    sx = full_w / float(prev_w)
    sy = full_h / float(prev_h)

    polygons_full = []
    for poly_prev in polygons_preview:
        poly_full = scale_polygon(poly_prev, sx=sx, sy=sy)
        poly_full = clamp_polygon(poly_full, w=full_w, h=full_h)
        polygons_full.append(poly_full)

    # Compute focus scores
    gray_full = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    masks_full = polygons_to_masks((full_h, full_w), polygons_full)

    scores = [FocusMetrics.compute(gray_full, m, config.metric) for m in masks_full]
    score_summary = summarize_polygon_scores(scores)

    # Compute full image score and lighting metrics
    full_mask = np.ones((full_h, full_w), dtype=np.uint8) * 255
    full_image_score = FocusMetrics.compute(gray_full, full_mask, config.metric)
    full_image_lighting = compute_region_lighting_metrics(gray_full, full_mask)
    polygon_lighting = [compute_region_lighting_metrics(gray_full, m) for m in masks_full]

    # Extract metadata
    metadata = extract_image_metadata(image_path, config.tz)
    
    # Create result
    result = FocusResult(
        metadata=metadata,
        metric=config.metric,
        polygon_scores=scores,
        avg_score=score_summary["avg"],
        polygon_score_min=score_summary["min"],
        polygon_score_max=score_summary["max"],
        polygon_score_std=score_summary["std"],
        polygon_score_range=score_summary["range"],
        full_image_score=full_image_score,
        full_image_lighting=full_image_lighting,
        polygon_lighting=polygon_lighting,
        polygons_file=str(config.polygons_path),
        preview_max=config.preview_max,
        run_time_local=datetime.now(ZoneInfo(config.tz)).isoformat(),
        comments="",
    )

    # Create visualization plot if requested
    if config.save_plots:
        plot_filename = f"{image_path.stem}_focus_plot.png"
        plot_path = config.plot_dir / plot_filename
        
        try:
            create_focus_plot(preview_bgr, polygons_preview, result, plot_path, manual_meta)
            logger.debug(f"Saved plot: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create plot for {image_path.name}: {e}")

    return result


# ----------------------------
# Batch Processing
# ----------------------------

def find_images(images_dir: Path) -> List[Path]:
    """Find all image files in directory recursively."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.name)
    return paths


def process_images_batch(
    image_paths: List[Path],
    config: ProcessingConfig,
    manual_metadata_csv: Optional[Path] = None  # NEW
) -> Tuple[int, int, int]:
    """
    Process multiple images in batch.
    
    Returns:
        (processed_count, skipped_count, failed_count)
    """
    # Load manual metadata if provided
    manual_data = []
    if manual_metadata_csv and manual_metadata_csv.exists():
        manual_data = load_manual_metadata(manual_metadata_csv)
    

    existing_keys = load_existing_keys(config.csv_path)

    def make_key(image_name: str) -> Tuple[str, str, str]:
        return (image_name, config.metric, str(config.polygons_path))

    processed = 0
    skipped = 0
    failed = 0

    # If polygons don't exist, force draw on first image
    need_draw_once = config.draw or (not config.polygons_path.exists())

    for i, img_path in enumerate(image_paths):

        if config.ignore_downscaled_images and "downscaled" in img_path.name.lower():
            logger.info(f"Skipping {img_path.name} (marked as downscaled)")
            skipped += 1
            continue

        key = make_key(str(img_path))
        
        # Skip if already computed (unless recompute flag set)
        if (not config.recompute) and (key in existing_keys):
            logger.info(f"Skipping {img_path.name} (already computed)")
            skipped += 1
            continue

        try:
            logger.info(f"Processing {img_path.name} ({i+1}/{len(image_paths)})")
            
            # Force draw only on first image if needed
            force_draw = need_draw_once and i == 0

            # Match manual metadata BEFORE computing (so we can pass to plot)
            manual_meta = {}
            if manual_data:
                # Need to get epoch first
                epoch_val = parse_epoch_from_filename(img_path.name)
                manual_meta = match_manual_metadata(epoch_val, manual_data)
            
            result = compute_focus_for_image(img_path, config, force_draw=force_draw, manual_meta=manual_meta)
            append_result_to_csv(config.csv_path, result, manual_meta)
            
            processed += 1
            existing_keys.add(key)  # Prevent duplicate processing in same run
            
            logger.info(f"  ✓ {img_path.name}: avg_score={result.avg_score:.2f}")
            
        except Exception as e:
            failed += 1
            logger.error(f"  ✗ {img_path.name}: {type(e).__name__}: {e}")

    return processed, skipped, failed


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute focus scores on polygonal regions of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available focus metrics:
  {', '.join(FocusMetrics.available_metrics())}

Examples:
  # Interactive: draw polygons on one image
  %(prog)s --image photo.jpg --csv results.csv --draw
  
  # Batch: process all images in directory (reuse polygons)
  %(prog)s --images-dir ./photos --csv results.csv
  
  # Generate visualization plots for each image
  %(prog)s --images-dir ./photos --csv results.csv --save-plots
  
  # Custom plot directory
  %(prog)s --images-dir ./photos --csv results.csv --save-plots --plot-dir ./my_plots
  
  # Use different metric
  %(prog)s --images-dir ./photos --csv results.csv --metric tenengrad
  
  # Force recomputation with plots
  %(prog)s --images-dir ./photos --csv results.csv --recompute --save-plots
        """
    )
    
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single image")
    group.add_argument("--images-dir", type=str, help="Directory of images (recursive)")
    

    ap.add_argument("--csv", required=True, type=str, help="CSV path for results")
    ap.add_argument("--polygons", type=str, default="polygons.json",
                    help="Polygon JSON file (default: polygons.json)")
    ap.add_argument("--draw", action="store_true",
                    help="Launch UI to draw polygons")
    ap.add_argument("--metric", default="laplacian_var",
                    choices=FocusMetrics.available_metrics(),
                    help="Focus metric to compute")
    ap.add_argument("--tz", default="America/Chicago",
                    help="Timezone for datetime conversion")
    ap.add_argument("--preview-max", type=int, default=1600,
                    help="Max preview dimension in pixels")
    ap.add_argument("--recompute", action="store_true",
                    help="Recompute existing entries")
    ap.add_argument("--target-polygons", type=int, default=4,
                    help="Number of polygons to draw (default: 4)")
    ap.add_argument("--save-plots", action="store_true",
                    help="Generate and save visualization plots for each image")
    ap.add_argument("--plot-dir", type=str, default="focus_scores_plots",
                    help="Directory to save plots (default: focus_scores_plots)")
    
    ap.add_argument("--yolo-model", type=str,
                    help="Path to trained YOLO OBB model weights for automatic detection")
    
    ap.add_argument("--manual-metadata", type=str,
                    help="CSV file with manual metadata (Date, Start time, End time, etc.)")

    ap.add_argument("--ignore-downscaled", action="store_true",
                    help="Ignore downscaled images")
    
    args = ap.parse_args()

    # Build configuration
    config = ProcessingConfig(
        csv_path=Path(args.csv),
        polygons_path=Path(args.polygons),
        metric=args.metric,
        tz=args.tz,
        preview_max=args.preview_max,
        draw=args.draw,
        recompute=args.recompute,
        target_polygons=args.target_polygons,
        save_plots=args.save_plots,
        plot_dir=Path(args.plot_dir),
        yolo_model_path=Path(args.yolo_model) if args.yolo_model else None,
        ignore_downscaled_images=args.ignore_downscaled,
    )

    # Get image list
    if args.image:
        image_paths = [Path(args.image)]
    else:
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            logger.error(f"Directory not found: {images_dir}")
            return 1
        image_paths = find_images(images_dir)

    if not image_paths:
        logger.warning("No images found")
        return 0

    logger.info(f"Found {len(image_paths)} images")
    logger.info(f"Metric: {config.metric}")
    logger.info(f"CSV output: {config.csv_path}")

    # Load manual metadata if provided
    manual_csv = Path(args.manual_metadata) if args.manual_metadata else None

    # Process images
    processed, skipped, failed = process_images_batch(image_paths, config, manual_csv)

    # Summary
    logger.info("=" * 60)
    logger.info(f"COMPLETE: processed={processed}, skipped={skipped}, failed={failed}")
    logger.info(f"Results: {config.csv_path}")
    if config.save_plots:
        logger.info(f"Plots: {config.plot_dir}/")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())