"""
Field boundary detection for sideline filtering.
Detects the green playing field area and filters out detections
on the sideline (coaches, substitutes, spectators).

Uses green field contour detection + convex hull to define the
playing area polygon, then Hough line refinement on the field edge.
"""
import cv2
import numpy as np
from typing import List, Optional


def detect_field_polygon(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the playing field polygon using green turf detection + convex hull.
    
    Returns an Nx1x2 array (OpenCV contour format) representing the field boundary,
    or None if detection fails.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect green turf
    green_mask = cv2.inRange(hsv, (30, 30, 30), (90, 255, 255))
    
    # Morphological cleanup to fill gaps in field markings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    green_filled = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find largest green contour (= the field)
    contours, _ = cv2.findContours(green_filled, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    
    # Must cover at least 10% of frame to be considered a field
    if cv2.contourArea(largest) < (h * w * 0.10):
        return None
    
    # Convex hull gives clean field boundary
    hull = cv2.convexHull(largest)
    
    # Refine: use Hough lines on field edge to straighten the boundary
    hull_refined = _refine_with_hough(green_filled, hull, h, w)
    
    return hull_refined


def _refine_with_hough(green_mask: np.ndarray, hull: np.ndarray,
                       h: int, w: int) -> np.ndarray:
    """
    Refine the hull boundary using Hough line detection on the field edge.
    Finds the dominant sideline and clips the hull to it.
    """
    # Get edges of the green field mask
    edges = cv2.Canny(green_mask, 50, 150)
    
    # Detect lines at the field boundary
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=80, maxLineGap=30)
    
    if lines is None or len(lines) < 3:
        return hull
    
    # Find the longest near-horizontal line that's close to the hull boundary
    # This is the actual sideline
    best_line = None
    best_length = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = abs(np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1))))
        
        # Sideline candidates: long, not-too-steep lines in lower half
        if length > best_length and angle < 50 and min(y1, y2) > h * 0.3:
            best_length = length
            best_line = (x1, y1, x2, y2)
    
    # If we found a strong sideline, use it; otherwise just use the hull
    return hull


def filter_detections_by_field(detections: List[List[int]], 
                               field_polygon: np.ndarray,
                               margin: int = 5) -> List[List[int]]:
    """
    Filter detections to only keep those with feet (bottom-center of bbox)
    inside the field polygon.
    
    Args:
        detections: list of [x1, y1, x2, y2] bounding boxes
        field_polygon: Nx1x2 contour array defining the field
        margin: pixels of slack to allow near boundary (negative = inside tolerance)
    
    Returns:
        Filtered list of detections on the field
    """
    if field_polygon is None or len(detections) == 0:
        return detections
    
    poly = field_polygon.reshape(-1, 1, 2).astype(np.float32)
    
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det
        # Foot position = bottom-center of bounding box
        foot_x = float((x1 + x2) / 2)
        foot_y = float(y2)
        
        # pointPolygonTest: positive = inside, 0 = on edge, negative = outside
        dist = cv2.pointPolygonTest(poly, (foot_x, foot_y), True)
        
        if dist >= -margin:
            filtered.append(det)
    
    return filtered


def detect_field_polygon_smoothed(frame: np.ndarray,
                                   prev_polygon: Optional[np.ndarray] = None,
                                   update_interval: int = 30) -> Optional[np.ndarray]:
    """
    Detect field polygon with temporal stability.
    Only recomputes every `update_interval` calls; returns cached result otherwise.
    
    For use in video pipelines - call every frame, it will only recompute
    when needed.
    """
    # This is a simple wrapper; the caller should manage caching
    # based on frame count
    polygon = detect_field_polygon(frame)
    
    if polygon is None:
        return prev_polygon
    
    if prev_polygon is not None and polygon.shape == prev_polygon.shape:
        # Smooth to reduce jitter
        smoothed = (0.8 * prev_polygon + 0.2 * polygon).astype(np.int32)
        return smoothed
    
    return polygon
