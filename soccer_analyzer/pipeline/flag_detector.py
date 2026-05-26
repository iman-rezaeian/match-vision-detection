"""Detect neon-colored flags/cones for field boundary calibration."""

import cv2
import numpy as np
from typing import Optional


# HSV ranges for common neon flag colors
# Note: outdoor flags at distance have lower S/V than indoor — thresholds tuned for real field conditions
FLAG_COLOR_RANGES = {
    "red": ((0, 80, 80), (10, 255, 255), (170, 80, 80), (180, 255, 255)),  # wraps hue 0/180
    "orange": ((5, 80, 80), (25, 255, 255)),
    "pink": ((145, 80, 100), (175, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
    "green_neon": ((35, 100, 100), (85, 255, 255)),
    "blue_neon": ((95, 100, 100), (125, 255, 255)),
}


class FlagDetector:
    """Detect neon-colored flags or cones using HSV color thresholding."""

    def __init__(self, flag_color: str = "red", min_area: int = 50,
                 max_area: int = 50000):
        """
        Args:
            flag_color: Key into FLAG_COLOR_RANGES or custom (lo_h,lo_s,lo_v,hi_h,hi_s,hi_v)
            min_area: Minimum contour area in pixels to count as a flag
            max_area: Maximum contour area to filter large blobs (not flags)
        """
        if flag_color in FLAG_COLOR_RANGES:
            color_range = FLAG_COLOR_RANGES[flag_color]
            if len(color_range) == 4:
                # Dual-range color (e.g., red wraps hue 0/180)
                self.hsv_ranges = [
                    (color_range[0], color_range[1]),
                    (color_range[2], color_range[3]),
                ]
            else:
                self.hsv_ranges = [(color_range[0], color_range[1])]
        else:
            raise ValueError(
                f"Unknown flag color '{flag_color}'. "
                f"Available: {list(FLAG_COLOR_RANGES.keys())}"
            )
        self.flag_color = flag_color
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """
        Detect flag positions in a single frame.

        Args:
            frame: BGR image

        Returns:
            List of (x, y) centroids sorted top-left to bottom-right
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build mask (supports dual-range colors like red)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Morphological cleanup — close small gaps, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))

        # Sort: top-to-bottom, then left-to-right
        centroids.sort(key=lambda p: (p[1], p[0]))
        return centroids

    def detect_stable(self, frames: list[np.ndarray], min_agreement: float = 0.6) -> list[tuple[int, int]]:
        """
        Detect flags across multiple frames and return stable positions.

        Uses median of detections across frames to reduce noise.

        Args:
            frames: List of BGR frames (5-10 frames recommended)
            min_agreement: Fraction of frames a flag must appear in

        Returns:
            Stable flag centroids (median positions)
        """
        all_detections = []
        for frame in frames:
            flags = self.detect(frame)
            all_detections.append(flags)

        if not all_detections:
            return []

        # Find most common number of flags detected
        counts = [len(d) for d in all_detections]
        if not counts:
            return []

        expected_count = max(set(counts), key=counts.count)
        min_frames = int(len(frames) * min_agreement)

        # Filter to frames with expected count
        valid_detections = [d for d in all_detections if len(d) == expected_count]

        if len(valid_detections) < min_frames:
            # Fall back to any detection
            valid_detections = [d for d in all_detections if len(d) > 0]
            if not valid_detections:
                return []
            expected_count = min(len(d) for d in valid_detections)
            valid_detections = [d[:expected_count] for d in valid_detections]

        # Compute median position for each flag index
        stable_flags = []
        for i in range(expected_count):
            xs = [d[i][0] for d in valid_detections if i < len(d)]
            ys = [d[i][1] for d in valid_detections if i < len(d)]
            if xs and ys:
                stable_flags.append((int(np.median(xs)), int(np.median(ys))))

        return stable_flags

    def assign_corners(self, centroids: list[tuple[int, int]],
                       frame_shape: tuple) -> Optional[dict]:
        """
        Assign detected flag centroids to field corners.

        Expects exactly 4 flags forming a rectangle.

        Args:
            centroids: List of (x, y) flag positions
            frame_shape: (H, W, C) of the frame

        Returns:
            Dict with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'
            mapping to (x, y) pixel coords, or None if not exactly 4 flags
        """
        if len(centroids) != 4:
            return None

        h, w = frame_shape[:2]
        pts = np.array(centroids, dtype=np.float32)

        # Find centroid of all points
        center = pts.mean(axis=0)

        # Classify each point relative to center
        top = pts[pts[:, 1] < center[1]]
        bottom = pts[pts[:, 1] >= center[1]]

        if len(top) != 2 or len(bottom) != 2:
            # Fallback: sort by angle from center
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            order = np.argsort(angles)
            # Reorder: TL, TR, BR, BL (counter-clockwise from top-left)
            pts_sorted = pts[order]
            return {
                "top_left": tuple(pts_sorted[2].astype(int)),
                "top_right": tuple(pts_sorted[3].astype(int)),
                "bottom_right": tuple(pts_sorted[0].astype(int)),
                "bottom_left": tuple(pts_sorted[1].astype(int)),
            }

        # Sort top row left-to-right
        top = top[top[:, 0].argsort()]
        # Sort bottom row left-to-right
        bottom = bottom[bottom[:, 0].argsort()]

        return {
            "top_left": tuple(top[0].astype(int)),
            "top_right": tuple(top[1].astype(int)),
            "bottom_left": tuple(bottom[0].astype(int)),
            "bottom_right": tuple(bottom[1].astype(int)),
        }

    def visualize(self, frame: np.ndarray, centroids: list[tuple[int, int]],
                  corners: Optional[dict] = None) -> np.ndarray:
        """Draw detected flags and optional corner labels on frame."""
        vis = frame.copy()

        for i, (cx, cy) in enumerate(centroids):
            cv2.circle(vis, (cx, cy), 12, (0, 255, 0), 3)
            cv2.putText(vis, str(i), (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if corners:
            color_map = {
                "top_left": (255, 0, 0),
                "top_right": (0, 255, 0),
                "bottom_left": (0, 0, 255),
                "bottom_right": (255, 255, 0),
            }
            for label, (cx, cy) in corners.items():
                color = color_map.get(label, (255, 255, 255))
                cv2.circle(vis, (cx, cy), 16, color, 4)
                cv2.putText(vis, label.replace("_", " "), (cx + 20, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw field outline
            pts = np.array([
                corners["top_left"], corners["top_right"],
                corners["bottom_right"], corners["bottom_left"]
            ], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        return vis
