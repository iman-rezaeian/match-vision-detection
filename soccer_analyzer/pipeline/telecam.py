"""Virtual TeleCam — pan-and-crop a 4K wide-angle video into a broadcast-style tele view."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d


@dataclass
class CropWindow:
    """Defines the virtual camera crop rectangle."""
    width: int = 1920
    height: int = 1080


class TeleCam:
    """Compute per-frame crop positions from player detection data.

    Given a 4K wide-angle source (3840×2160) and a crop window (default
    1920×1080), this produces a smooth panning trajectory that follows
    the densest cluster of players (the "ball swarm" in youth soccer).
    """

    def __init__(self, crop: CropWindow = None, smoothing: float = 0.05,
                 vertical_bias: float = 0.5):
        """
        Args:
            crop: output crop dimensions (default 1920×1080)
            smoothing: controls pan smoothing (lower = smoother).
                       Maps to Gaussian sigma on the trajectory.
            vertical_bias: where to vertically center the crop on the field
                           (0.0 = top, 0.5 = center, 1.0 = bottom)
        """
        self.crop = crop or CropWindow()
        self.smoothing = smoothing
        self.vertical_bias = vertical_bias

    @staticmethod
    def _find_densest_cluster(x_vals, y_vals, weights=None):
        """Find the center of the densest player cluster in a single frame.

        Uses a sliding-window density approach: for each player, count how
        many other players are within a neighbourhood. The player with the
        highest nearby count is the cluster center. This naturally finds the
        "ball swarm" in youth soccer.
        """
        n = len(x_vals)
        if n == 0:
            return None, None
        if n <= 2:
            if weights is not None and len(weights) == n:
                return float(np.average(x_vals, weights=weights)), float(np.average(y_vals, weights=weights))
            return float(np.mean(x_vals)), float(np.mean(y_vals))

        # Neighbourhood radius: ~20% of typical field width in pixels
        # For 3840px wide frame, ~770px radius captures a local pack
        radius = 770.0

        best_score = -1
        best_cx, best_cy = float(np.median(x_vals)), float(np.median(y_vals))

        for i in range(n):
            dx = x_vals - x_vals[i]
            dy = y_vals - y_vals[i]
            dist = np.sqrt(dx * dx + dy * dy)
            nearby = dist < radius
            score = nearby.sum()
            if score > best_score:
                best_score = score
                # Center is the median of the nearby cluster (robust to outliers)
                best_cx = float(np.median(x_vals[nearby]))
                best_cy = float(np.median(y_vals[nearby]))

        return best_cx, best_cy

    def compute_trajectory(self, detections_df: pd.DataFrame,
                           frame_w: int, frame_h: int,
                           fps: float = 30.0,
                           start_frame: int = None,
                           end_frame: int = None,
                           ball_df: pd.DataFrame = None) -> pd.DataFrame:
        """Compute per-frame crop center (cx, cy) for every frame in range.

        Args:
            detections_df: DataFrame with columns frame, x_px, y_px,
                           bbox_x1, bbox_y1, bbox_x2, bbox_y2
            frame_w: source video width (e.g. 3840)
            frame_h: source video height (e.g. 2160)
            fps: source fps
            start_frame: first frame (default: min in data)
            end_frame: last frame (default: max in data)
            ball_df: optional DataFrame with columns [frame, x_px, y_px]
                     from ball detection. If provided, ball position is
                     strongly weighted.

        Returns:
            DataFrame with columns [frame, cx, cy]
        """
        if start_frame is None:
            start_frame = int(detections_df["frame"].min())
        if end_frame is None:
            end_frame = int(detections_df["frame"].max())

        df = detections_df.copy()

        # --- Step 1: filter to on-field players only -----------------------
        # Remove sideline spectators/coaches using y_px distribution
        y_p5 = df["y_px"].quantile(0.05)
        y_p85 = df["y_px"].quantile(0.85)
        bbox_h = df["bbox_y2"] - df["bbox_y1"]
        # On-field players: in the field zone AND not huge (coaches near camera)
        median_h = bbox_h.median()
        field_mask = (
            (df["y_px"] >= y_p5) &
            (df["y_px"] <= y_p85) &
            (bbox_h < median_h * 3)  # exclude very large near-camera detections
        )
        df_field = df[field_mask]
        if len(df_field) < len(df) * 0.3:
            df_field = df  # fallback: filter too aggressive

        # --- Step 2: compute cluster center per sampled frame ---------------
        raw_points = []
        for frame_num, group in df_field.groupby("frame"):
            x_vals = group["x_px"].values
            y_vals = group["y_px"].values
            cx, cy = self._find_densest_cluster(x_vals, y_vals)
            if cx is not None:
                raw_points.append({"frame": frame_num, "raw_cx": cx, "raw_cy": cy})

        if not raw_points:
            # Fallback: center of frame
            half_w = self.crop.width // 2
            half_h = self.crop.height // 2
            return pd.DataFrame({
                "frame": range(start_frame, end_frame + 1),
                "cx": frame_w // 2,
                "cy": frame_h // 2,
            })

        grouped = pd.DataFrame(raw_points)

        # --- Step 3: incorporate ball position if available -----------------
        if ball_df is not None and not ball_df.empty:
            ball = ball_df[["frame", "x_px", "y_px"]].rename(
                columns={"x_px": "ball_cx", "y_px": "ball_cy"}
            )
            grouped = grouped.merge(ball, on="frame", how="left")
            has_ball = grouped["ball_cx"].notna()
            # When ball is detected, blend 70% ball + 30% cluster
            grouped.loc[has_ball, "raw_cx"] = (
                0.7 * grouped.loc[has_ball, "ball_cx"] +
                0.3 * grouped.loc[has_ball, "raw_cx"]
            )
            grouped.loc[has_ball, "raw_cy"] = (
                0.7 * grouped.loc[has_ball, "ball_cy"] +
                0.3 * grouped.loc[has_ball, "raw_cy"]
            )

        # --- Step 4: interpolate to every frame ----------------------------
        all_frames = pd.DataFrame({"frame": range(start_frame, end_frame + 1)})
        merged = all_frames.merge(grouped[["frame", "raw_cx", "raw_cy"]],
                                  on="frame", how="left")
        merged["raw_cx"] = merged["raw_cx"].interpolate(method="linear").ffill().bfill()
        merged["raw_cy"] = merged["raw_cy"].interpolate(method="linear").ffill().bfill()

        # --- Step 5: smooth with Gaussian filter ---------------------------
        # sigma = fps / smoothing_factor. Lower smoothing → higher sigma → smoother
        # Default smoothing=0.05 → sigma=600 (~20s window at 30fps) = very smooth
        # smoothing=0.20 → sigma=150 (~5s) = responsive
        sigma = fps / max(self.smoothing, 0.001)
        sigma = min(sigma, len(merged) / 2)  # cap at half the clip length
        cx_smooth = gaussian_filter1d(merged["raw_cx"].values.astype(float), sigma=sigma)

        # --- Step 6: mostly-fixed vertical, slight tracking ----------------
        y_min_field = df_field["y_px"].quantile(0.05)
        y_max_field = df_field["y_px"].quantile(0.95)
        y_center = y_min_field + (y_max_field - y_min_field) * self.vertical_bias
        # 90% locked vertical, 10% tracking for slight tilt
        cy_raw = merged["raw_cy"].values.astype(float)
        cy_smooth = 0.9 * y_center + 0.1 * gaussian_filter1d(cy_raw, sigma=sigma)

        # --- Step 7: clamp to valid crop bounds ----------------------------
        half_w = self.crop.width / 2
        half_h = self.crop.height / 2
        cx_smooth = np.clip(cx_smooth, half_w, frame_w - half_w)
        cy_smooth = np.clip(cy_smooth, half_h, frame_h - half_h)

        return pd.DataFrame({
            "frame": merged["frame"].values,
            "cx": cx_smooth.astype(int),
            "cy": cy_smooth.astype(int),
        })

    def get_crop_box(self, cx: int, cy: int) -> tuple:
        """Convert center (cx, cy) to (x1, y1, x2, y2) crop coordinates."""
        x1 = cx - self.crop.width // 2
        y1 = cy - self.crop.height // 2
        return (x1, y1, x1 + self.crop.width, y1 + self.crop.height)

    def preview_frame(self, frame: np.ndarray, cx: int, cy: int) -> tuple:
        """Return (cropped_frame, annotated_wide_frame) for UI preview.

        The annotated frame has the crop rectangle drawn on the wide source.
        """
        import cv2

        x1, y1, x2, y2 = self.get_crop_box(cx, cy)
        cropped = frame[y1:y2, x1:x2].copy()

        # Draw crop rectangle on a downscaled copy of the wide frame
        wide_preview = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        scale = 0.5
        cv2.rectangle(
            wide_preview,
            (int(x1 * scale), int(y1 * scale)),
            (int(x2 * scale), int(y2 * scale)),
            (0, 255, 0), 2,
        )

        return cropped, wide_preview
