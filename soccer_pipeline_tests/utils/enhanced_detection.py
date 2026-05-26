"""Enhanced detection utilities for improved pipeline performance.

Implements:
- SAHI-style tiled inference for ultra-wide fisheye images
- Multi-scale detection fusion
- Adaptive NMS for wide-angle distortion
- Undistortion-before-detection preprocessing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# ────────────────────────────────────────────────────────────
# Model selection
# ────────────────────────────────────────────────────────────

# Default model — can be overridden by fine-tuned weights
DEFAULT_MODEL = "yolov8s.pt"


def get_model(model_name: str = None):
    """Load YOLO model. Default to yolov8s for better small-object detection."""
    from ultralytics import YOLO
    return YOLO(model_name or DEFAULT_MODEL)


# ────────────────────────────────────────────────────────────
# SAHI-style tiled inference
# ────────────────────────────────────────────────────────────

def detect_with_tiling(model, frame: np.ndarray,
                       confidence: float = 0.15,
                       tile_width: int = 1280,
                       overlap: int = 200,
                       imgsz: int = 1280,
                       classes: list = [0]) -> List[Tuple[float, float, float, float, float]]:
    """
    Run YOLO detection with SAHI-style tiled inference.

    For ultra-wide images (e.g. 6500×1000), slices into overlapping tiles,
    runs detection on each, merges results with NMS.

    Args:
        model: YOLO model
        frame: Input image (H, W, C)
        confidence: Detection confidence threshold
        tile_width: Width of each tile
        overlap: Overlap between adjacent tiles (pixels)
        imgsz: YOLO inference size per tile
        classes: Class IDs to detect

    Returns:
        List of (x1, y1, x2, y2, conf) detections in original coordinates
    """
    h, w = frame.shape[:2]

    # If image is small enough, just run normally
    if w <= tile_width * 1.5:
        return _detect_single(model, frame, confidence, imgsz, classes)

    # Generate tile positions
    stride = tile_width - overlap
    tile_starts = list(range(0, w - tile_width + 1, stride))
    if tile_starts[-1] + tile_width < w:
        tile_starts.append(w - tile_width)

    all_detections = []

    for x_start in tile_starts:
        x_end = x_start + tile_width
        tile = frame[:, x_start:x_end]

        # Run detection on tile
        results = model(tile, classes=classes, conf=confidence,
                        imgsz=imgsz, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0].cpu())
                # Shift coordinates back to original image space
                all_detections.append((
                    float(x1) + x_start,
                    float(y1),
                    float(x2) + x_start,
                    float(y2),
                    conf_val,
                ))

    # Apply NMS to merged detections
    if not all_detections:
        return []

    return _apply_nms(all_detections, iou_threshold=0.5)


def _detect_single(model, frame: np.ndarray, confidence: float,
                    imgsz: int, classes: list) -> List[Tuple[float, float, float, float, float]]:
    """Run detection on a single frame without tiling."""
    results = model(frame, classes=classes, conf=confidence,
                    imgsz=imgsz, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu())
            detections.append((float(x1), float(y1), float(x2), float(y2), conf_val))
    return detections


# ────────────────────────────────────────────────────────────
# Multi-scale detection fusion
# ────────────────────────────────────────────────────────────

def detect_multiscale(model, frame: np.ndarray,
                      confidence: float = 0.15,
                      scales: list = None,
                      classes: list = [0]) -> List[Tuple[float, float, float, float, float]]:
    """
    Run detection at multiple scales and merge results.

    Small objects are better detected at high resolution,
    large objects at lower resolution. Merging gives the best of both.

    Args:
        model: YOLO model
        frame: Input image
        confidence: Detection confidence
        scales: List of imgsz values to try (default: auto-computed)
        classes: Class IDs to detect

    Returns:
        List of (x1, y1, x2, y2, conf) detections
    """
    h, w = frame.shape[:2]

    if scales is None:
        # Auto-compute scales based on image width
        if w > 3000:
            scales = [1280, 1920, 3200]
        elif w > 1920:
            scales = [1280, 1920]
        else:
            scales = [640, 1280]

    all_detections = []

    for imgsz in scales:
        results = model(frame, classes=classes, conf=confidence,
                        imgsz=imgsz, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0].cpu())
                all_detections.append((float(x1), float(y1), float(x2), float(y2), conf_val))

    if not all_detections:
        return []

    return _apply_nms(all_detections, iou_threshold=0.5)


# ────────────────────────────────────────────────────────────
# Adaptive NMS for wide-angle images
# ────────────────────────────────────────────────────────────

def _apply_nms(detections: List[Tuple[float, float, float, float, float]],
               iou_threshold: float = 0.5) -> List[Tuple[float, float, float, float, float]]:
    """
    Apply Non-Maximum Suppression to merged detections.

    Args:
        detections: List of (x1, y1, x2, y2, confidence)
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered detections after NMS
    """
    if not detections:
        return []

    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)

    # Convert xyxy to xywh for cv2.dnn.NMSBoxes
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = x2 - x1
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = y2 - y1

    # OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    # OpenCV returns indices differently depending on version
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    else:
        indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

    return [detections[i] for i in indices]


def adaptive_nms(detections: List[Tuple[float, float, float, float, float]],
                 image_width: int,
                 base_iou: float = 0.5) -> List[Tuple[float, float, float, float, float]]:
    """
    Apply position-adaptive NMS for wide-angle/fisheye images.

    Objects near the edges of fisheye images appear more distorted and
    may overlap differently than center objects. This uses a lower IoU
    threshold near edges (more aggressive suppression) and higher at center.

    Args:
        detections: List of (x1, y1, x2, y2, confidence)
        image_width: Width of the original image
        base_iou: Base IoU threshold

    Returns:
        Filtered detections
    """
    if not detections or image_width == 0:
        return detections

    # Sort by confidence (descending)
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)

        best_cx = (best[0] + best[2]) / 2
        # Distance from center (0 = center, 1 = edge)
        edge_factor = abs(best_cx - image_width / 2) / (image_width / 2)

        remaining = []
        for d in dets:
            iou = _compute_iou_single(best, d)
            # More aggressive suppression near edges
            threshold = base_iou * (1 - 0.2 * edge_factor)
            if iou < threshold:
                remaining.append(d)
        dets = remaining

    return keep


def _compute_iou_single(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2, ...)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ────────────────────────────────────────────────────────────
# Full enhanced detection pipeline
# ────────────────────────────────────────────────────────────

def enhanced_detect(model, frame: np.ndarray,
                    confidence: float = 0.15,
                    use_tiling: bool = True,
                    use_multiscale: bool = True,
                    use_adaptive_nms: bool = True,
                    undistort_map: tuple = None,
                    classes: list = [0]) -> List[Tuple[float, float, float, float]]:
    """
    Full enhanced detection pipeline combining all improvements.

    Pipeline:
    1. (Optional) Undistort fisheye frame
    2. Run tiled inference (SAHI-style)
    3. Run multi-scale fusion
    4. Merge all detections
    5. Apply adaptive NMS

    Args:
        model: YOLO model
        frame: Input image
        confidence: Detection confidence threshold
        use_tiling: Enable SAHI tiled inference
        use_multiscale: Enable multi-scale fusion
        use_adaptive_nms: Enable position-adaptive NMS
        undistort_map: Optional (map1, map2) for fisheye undistortion
        classes: YOLO class IDs to detect

    Returns:
        List of (x1, y1, x2, y2) bounding boxes
    """
    h, w = frame.shape[:2]

    # Step 1: Undistort if calibration available
    if undistort_map is not None:
        map1, map2 = undistort_map
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    all_detections = []

    # Step 2: Tiled inference for wide images
    if use_tiling and w > 2000:
        tile_dets = detect_with_tiling(
            model, frame, confidence=confidence,
            tile_width=1500, overlap=300, imgsz=1280, classes=classes
        )
        all_detections.extend(tile_dets)

    # Step 3: Multi-scale on full image
    if use_multiscale:
        ms_dets = detect_multiscale(
            model, frame, confidence=confidence, classes=classes
        )
        all_detections.extend(ms_dets)
    elif not use_tiling or w <= 2000:
        # Fallback: single inference
        imgsz = 1280 if w <= 1920 else min(3200, w // 2)
        single_dets = _detect_single(model, frame, confidence, imgsz, classes)
        all_detections.extend(single_dets)

    if not all_detections:
        return []

    # Step 4: Apply NMS
    if use_adaptive_nms:
        final = adaptive_nms(all_detections, w)
    else:
        final = _apply_nms(all_detections, iou_threshold=0.5)

    # Return without confidence scores for backward compatibility
    return [(d[0], d[1], d[2], d[3]) for d in final]


def load_undistort_maps(calibration_path: str) -> Optional[tuple]:
    """Load fisheye undistortion maps from calibration file."""
    cal_path = Path(calibration_path)
    if not cal_path.exists():
        return None

    try:
        data = np.load(str(cal_path))
        if "map1" in data and "map2" in data:
            return (data["map1"], data["map2"])
    except Exception:
        pass

    return None
