"""Per-drill and per-player metrics for training sessions."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pipeline.drill_segmenter import DrillSegment
from config import SPRINT_THRESHOLD, MAX_SPEED_CAP


@dataclass
class PlayerDrillMetrics:
    """Metrics for a single player within a single drill segment."""
    player_id: str
    track_id: int
    segment_index: int
    drill_type: str
    # Movement
    distance_m: float = 0.0
    avg_speed_ms: float = 0.0
    max_speed_ms: float = 0.0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    # Intensity
    time_active_pct: float = 0.0      # % of drill time in motion (>1 m/s)
    high_intensity_pct: float = 0.0   # % time above sprint threshold
    # Spatial
    area_covered_m2: float = 0.0      # convex hull area
    avg_x: float = 0.0
    avg_y: float = 0.0


class DrillMetricsCalculator:
    """Compute per-player metrics for each drill segment."""

    def __init__(self, sprint_threshold: float = SPRINT_THRESHOLD,
                 max_speed_cap: float = MAX_SPEED_CAP):
        self.sprint_threshold = sprint_threshold
        self.max_speed_cap = max_speed_cap

    def compute_all(self, df: pd.DataFrame, segments: list[DrillSegment],
                    fps: float) -> pd.DataFrame:
        """
        Compute metrics for all players across all drill segments.

        Args:
            df: Full detection DataFrame with x_field, y_field, track_id, frame
            segments: List of DrillSegment from DrillSegmenter
            fps: Video FPS

        Returns:
            DataFrame with one row per (player, drill_segment) combination
        """
        all_metrics = []

        for seg in segments:
            seg_df = df[(df["frame"] >= seg.start_frame) &
                        (df["frame"] <= seg.end_frame)]

            if seg_df.empty:
                continue

            for tid, track_df in seg_df.groupby("track_id"):
                if len(track_df) < 3:
                    continue

                metrics = self._compute_player_segment(
                    track_df, tid, seg, fps
                )
                all_metrics.append(metrics)

        if not all_metrics:
            return pd.DataFrame()

        # Convert to DataFrame
        rows = []
        for m in all_metrics:
            rows.append({
                "track_id": m.track_id,
                "player_id": m.player_id,
                "segment_index": m.segment_index,
                "drill_type": m.drill_type,
                "distance_m": m.distance_m,
                "avg_speed_ms": m.avg_speed_ms,
                "max_speed_ms": m.max_speed_ms,
                "sprint_count": m.sprint_count,
                "sprint_distance_m": m.sprint_distance_m,
                "time_active_pct": m.time_active_pct,
                "high_intensity_pct": m.high_intensity_pct,
                "area_covered_m2": m.area_covered_m2,
                "avg_x": m.avg_x,
                "avg_y": m.avg_y,
            })

        return pd.DataFrame(rows)

    def _compute_player_segment(self, track_df: pd.DataFrame, track_id: int,
                                segment: DrillSegment, fps: float) -> PlayerDrillMetrics:
        """Compute metrics for one player in one drill segment."""
        track_df = track_df.sort_values("frame")

        x = track_df["x_field"].values
        y = track_df["y_field"].values
        frames = track_df["frame"].values

        # Player ID (if assigned)
        player_id = ""
        if "player_name" in track_df.columns:
            names = track_df["player_name"].dropna().unique()
            if len(names) > 0:
                player_id = names[0]

        # Speed computation
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(frames) / fps
        valid = dt > 0
        speeds = np.zeros(len(dx))
        speeds[valid] = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
        speeds = np.clip(speeds, 0, self.max_speed_cap)

        # Distance
        distances = np.sqrt(dx**2 + dy**2)
        # Filter unrealistic jumps (track errors)
        max_step = self.max_speed_cap * dt
        valid_steps = np.ones(len(distances), dtype=bool)
        valid_steps[valid] = distances[valid] <= max_step[valid]
        total_distance = float(np.sum(distances[valid_steps]))

        # Speeds
        avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        max_speed = float(np.percentile(speeds, 95)) if len(speeds) > 5 else float(np.max(speeds)) if len(speeds) > 0 else 0.0

        # Sprint detection
        is_sprinting = speeds > self.sprint_threshold
        sprint_count = 0
        sprint_distance = 0.0
        in_sprint = False
        for i, sprinting in enumerate(is_sprinting):
            if sprinting and not in_sprint:
                sprint_count += 1
                in_sprint = True
            elif not sprinting:
                in_sprint = False
            if sprinting and i < len(distances):
                sprint_distance += distances[i]

        # Time active (moving > 1 m/s)
        duration_s = segment.duration_s
        active_frames = np.sum(speeds > 1.0)
        time_active_pct = float(active_frames / len(speeds) * 100) if len(speeds) > 0 else 0.0

        # High intensity time
        hi_frames = np.sum(speeds > self.sprint_threshold)
        high_intensity_pct = float(hi_frames / len(speeds) * 100) if len(speeds) > 0 else 0.0

        # Area covered (convex hull)
        area_covered = 0.0
        if len(x) >= 3:
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack([x, y])
                hull = ConvexHull(points)
                area_covered = float(hull.volume)  # 2D hull → .volume is area
            except Exception:
                area_covered = 0.0

        return PlayerDrillMetrics(
            player_id=player_id,
            track_id=track_id,
            segment_index=segment.index,
            drill_type=segment.drill_type,
            distance_m=round(total_distance, 1),
            avg_speed_ms=round(avg_speed, 2),
            max_speed_ms=round(max_speed, 2),
            sprint_count=sprint_count,
            sprint_distance_m=round(sprint_distance, 1),
            time_active_pct=round(time_active_pct, 1),
            high_intensity_pct=round(high_intensity_pct, 1),
            area_covered_m2=round(area_covered, 1),
            avg_x=round(float(np.mean(x)), 1),
            avg_y=round(float(np.mean(y)), 1),
        )

    def summarize_session(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate drill metrics into per-player session totals.

        Returns DataFrame with one row per player.
        """
        if metrics_df.empty:
            return pd.DataFrame()

        # Group by player (use player_id if available, else track_id)
        id_col = "player_id" if metrics_df["player_id"].notna().any() and (metrics_df["player_id"] != "").any() else "track_id"

        summary = metrics_df.groupby(id_col).agg(
            total_distance_m=("distance_m", "sum"),
            avg_speed_ms=("avg_speed_ms", "mean"),
            max_speed_ms=("max_speed_ms", "max"),
            total_sprints=("sprint_count", "sum"),
            total_sprint_distance_m=("sprint_distance_m", "sum"),
            avg_time_active_pct=("time_active_pct", "mean"),
            avg_high_intensity_pct=("high_intensity_pct", "mean"),
            drills_participated=("segment_index", "nunique"),
        ).round(1).reset_index()

        return summary
