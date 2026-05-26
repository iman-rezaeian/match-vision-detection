"""Pixel to real field coordinate mapping via homography."""

import cv2
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from config import DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M


class FieldHomography:
    def __init__(self, field_length_m: float = DEFAULT_FIELD_LENGTH_M,
                 field_width_m: float = DEFAULT_FIELD_WIDTH_M):
        self.H = None
        self.field_length = field_length_m
        self.field_width = field_width_m
        self.src_points = None
        self.dst_points = None

    def calibrate_auto(self, frame_h: int, frame_w: int, detections_df=None):
        """
        Estimate homography from video dimensions.
        If detections_df is provided, uses detection density to find the actual
        field zone (data-driven). Otherwise falls back to fixed margins.
        """
        if detections_df is not None and not detections_df.empty:
            self._calibrate_from_detections(frame_h, frame_w, detections_df)
        else:
            self._calibrate_fixed_margins(frame_h, frame_w)

    def _calibrate_fixed_margins(self, frame_h: int, frame_w: int):
        """Fallback: fixed margins assuming standard camera placement."""
        h_margin = 0.08
        v_margin = 0.12

        x_left = int(frame_w * h_margin)
        x_right = int(frame_w * (1 - h_margin))
        y_top = int(frame_h * v_margin)
        y_bottom = int(frame_h * (1 - v_margin))

        self.src_points = np.array([
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom],
        ], dtype=np.float32)

        self.dst_points = np.array([
            [0, 0],
            [self.field_length, 0],
            [self.field_length, self.field_width],
            [0, self.field_width],
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(self.src_points, self.dst_points)

    def _calibrate_from_detections(self, frame_h: int, frame_w: int,
                                    detections_df: pd.DataFrame):
        """
        Data-driven calibration: use detection density to find the actual
        playing field region. Works for any camera angle/zoom.
        """
        y_vals = detections_df["y_px"].values
        x_vals = detections_df["x_px"].values

        # Find the dense vertical band where the field is.
        # Build histogram, find the main cluster using IQR.
        y_p5 = np.percentile(y_vals, 3)
        y_p95 = np.percentile(y_vals, 80)
        # Filter to this range, then find the actual field extent
        field_mask = (y_vals >= y_p5) & (y_vals <= y_p95)
        if field_mask.sum() < 100:
            # Not enough data, fallback
            self._calibrate_fixed_margins(frame_h, frame_w)
            return

        field_y = y_vals[field_mask]
        field_x = x_vals[field_mask]

        # Use percentiles of the field cluster as src boundaries
        y_top = max(0, np.percentile(field_y, 2) - 10)
        y_bottom = min(frame_h, np.percentile(field_y, 98) + 10)
        x_left = max(0, np.percentile(field_x, 2) - 10)
        x_right = min(frame_w, np.percentile(field_x, 98) + 10)

        self.src_points = np.array([
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom],
        ], dtype=np.float32)

        self.dst_points = np.array([
            [0, 0],
            [self.field_length, 0],
            [self.field_length, self.field_width],
            [0, self.field_width],
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(self.src_points, self.dst_points)
        # Store the field zone boundaries for filtering
        self.field_y_min = float(y_p5)
        self.field_y_max = float(y_p95)

    def filter_to_field_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove detections that are outside the playing field zone.
        Also removes near-camera sideline people (coaches/spectators) who fall
        within the y-range but are stationary with large bboxes — indicating
        they are close to the camera rather than on the field."""
        if not hasattr(self, 'field_y_min'):
            return df
        mask = (df["y_px"] >= self.field_y_min) & (df["y_px"] <= self.field_y_max)
        result = df[mask].copy()

        # Remove stationary large-bbox tracks (sideline adults near camera).
        # Real players move significantly; sideline people stay in one spot.
        if not result.empty and "bbox_y1" in result.columns and "track_id" in result.columns:
            bh = result["bbox_y2"] - result["bbox_y1"]
            median_bh = bh.median()
            # Only apply if there's a mix of sizes (far + near detections)
            if median_bh > 0:
                # Threshold: tracks with avg bbox > 1.5x median AND low movement
                bh_threshold = max(median_bh * 1.5, 250)  # at least 250px at 4K
                displacement_threshold = 80  # pixels total displacement

                # Compute per-track stats
                sideline_tids = set()
                for tid, grp in result.groupby("track_id"):
                    if len(grp) < 4:
                        continue
                    track_bh = (grp["bbox_y2"] - grp["bbox_y1"]).mean()
                    if track_bh < bh_threshold:
                        continue
                    # Check displacement
                    dx = grp["x_px"].max() - grp["x_px"].min()
                    dy = grp["y_px"].max() - grp["y_px"].min()
                    displacement = float(np.sqrt(dx**2 + dy**2))
                    if displacement < displacement_threshold:
                        sideline_tids.add(tid)

                if sideline_tids:
                    result = result[~result["track_id"].isin(sideline_tids)]

        return result

    def calibrate_manual(self, src_points: list, dst_points: Optional[list] = None):
        """
        Compute homography from 4 manually clicked field points.
        src_points: 4 pixel coords clicked by user [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        dst_points: corresponding real-world meter coords
                    Default for U10 7v7: corners = (0,0),(50,0),(50,35),(0,35)
        """
        self.src_points = np.array(src_points, dtype=np.float32)

        if dst_points is None:
            self.dst_points = np.array([
                [0, 0],
                [self.field_length, 0],
                [self.field_length, self.field_width],
                [0, self.field_width],
            ], dtype=np.float32)
        else:
            self.dst_points = np.array(dst_points, dtype=np.float32)

        self.H, _ = cv2.findHomography(self.src_points, self.dst_points)

    def transform(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Convert single pixel point to field meters. Clamp to field bounds."""
        if self.H is None:
            raise ValueError("Homography not calibrated. Call calibrate_auto or calibrate_manual first.")

        point = np.array([[[x_px, y_px]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H)
        x_field = float(transformed[0][0][0])
        y_field = float(transformed[0][0][1])

        # Clamp to field bounds
        x_field = max(0, min(x_field, self.field_length))
        y_field = max(0, min(y_field, self.field_width))

        return x_field, y_field

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to entire detections DataFrame. Add x_field, y_field cols."""
        if self.H is None:
            raise ValueError("Homography not calibrated.")

        if df.empty:
            df["x_field"] = []
            df["y_field"] = []
            return df

        points = df[["x_px", "y_px"]].values.astype(np.float32)
        points_reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_reshaped, self.H)
        transformed = transformed.reshape(-1, 2)

        df = df.copy()
        df["x_field"] = np.clip(transformed[:, 0], 0, self.field_length)
        df["y_field"] = np.clip(transformed[:, 1], 0, self.field_width)

        return df

    def save(self, field_name: str, db):
        """Save calibration to fields table in SQLite."""
        db.save_field(
            name=field_name,
            field_length_m=self.field_length,
            field_width_m=self.field_width,
            src_points=self.src_points.tolist() if self.src_points is not None else None,
            dst_points=self.dst_points.tolist() if self.dst_points is not None else None,
            homography_matrix=self.H.tolist() if self.H is not None else None,
        )

    def load(self, field_name: str, db) -> bool:
        """Load calibration from SQLite by field name. Returns True if loaded."""
        field = db.get_field(field_name)
        if field is None:
            return False

        self.field_length = field["field_length_m"]
        self.field_width = field["field_width_m"]
        self.src_points = np.array(field["src_points"], dtype=np.float32) if field["src_points"] else None
        self.dst_points = np.array(field["dst_points"], dtype=np.float32) if field["dst_points"] else None
        self.H = np.array(field["homography_matrix"], dtype=np.float64) if field["homography_matrix"] else None
        return True

    def validate(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Check that transformed coordinates make sense.
        Warn if >20% of points fall outside field boundaries.
        Return validation_score 0-1 and warning message.
        """
        if "x_field" not in df.columns or "y_field" not in df.columns:
            return 0.0, "No transformed coordinates found."

        total = len(df)
        if total == 0:
            return 1.0, "No data to validate."

        out_of_bounds = (
            (df["x_field"] <= 0.01) | (df["x_field"] >= self.field_length - 0.01) |
            (df["y_field"] <= 0.01) | (df["y_field"] >= self.field_width - 0.01)
        ).sum()

        oob_pct = out_of_bounds / total
        score = 1.0 - oob_pct

        if oob_pct > 0.2:
            msg = (f"Calibration may be off — {oob_pct*100:.1f}% of positions fell "
                   f"outside field boundaries. Consider recalibrating.")
        elif oob_pct > 0.1:
            msg = f"Minor calibration note: {oob_pct*100:.1f}% of positions near field edges."
        else:
            msg = "Calibration looks good."

        return score, msg

    def get_grid_overlay(self, frame_h: int, frame_w: int, grid_spacing_m: float = 10.0):
        """Generate grid lines for visualization overlay on video frame."""
        if self.H is None:
            return []

        H_inv = np.linalg.inv(self.H)
        lines = []

        # Vertical lines (along field length)
        for x in np.arange(0, self.field_length + 0.1, grid_spacing_m):
            pts_field = np.array([[[x, 0]], [[x, self.field_width]]], dtype=np.float32)
            pts_px = cv2.perspectiveTransform(pts_field, H_inv)
            lines.append(pts_px.reshape(-1, 2).tolist())

        # Horizontal lines (along field width)
        for y in np.arange(0, self.field_width + 0.1, grid_spacing_m):
            pts_field = np.array([[[0, y]], [[self.field_length, y]]], dtype=np.float32)
            pts_px = cv2.perspectiveTransform(pts_field, H_inv)
            lines.append(pts_px.reshape(-1, 2).tolist())

        return lines


class FlagHomography(FieldHomography):
    """
    Homography calibration using detected flag/cone positions as ground-truth
    reference points. Used for training fields without painted lines.
    """

    def __init__(self, field_length_m: float = 40.0, field_width_m: float = 30.0):
        super().__init__(field_length_m, field_width_m)

    def calibrate_from_flags(self, corners: dict):
        """
        Calibrate homography from detected flag corner positions.

        Args:
            corners: dict with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'
                     each mapping to (x, y) pixel coordinates
        """
        self.src_points = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ], dtype=np.float32)

        self.dst_points = np.array([
            [0, 0],
            [self.field_length, 0],
            [self.field_length, self.field_width],
            [0, self.field_width],
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(self.src_points, self.dst_points)

        # Set field zone from flag positions for filtering
        all_x = [c[0] for c in corners.values()]
        all_y = [c[1] for c in corners.values()]
        margin = 50  # px margin outside flags
        self.field_y_min = min(all_y) - margin
        self.field_y_max = max(all_y) + margin
        self.field_x_min = min(all_x) - margin
        self.field_x_max = max(all_x) + margin

    def filter_to_field_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter detections to within the flag-bounded area."""
        if not hasattr(self, 'field_x_min'):
            return super().filter_to_field_zone(df)

        mask = (
            (df["x_px"] >= self.field_x_min) & (df["x_px"] <= self.field_x_max) &
            (df["y_px"] >= self.field_y_min) & (df["y_px"] <= self.field_y_max)
        )
        return df[mask].copy()

