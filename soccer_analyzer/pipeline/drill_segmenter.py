"""Drill segmentation — detect activity/rest boundaries in training sessions."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DrillSegment:
    """A single detected drill period within a training session."""
    index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    avg_intensity: float      # average speed (m/s) during this drill
    max_intensity: float      # peak speed during this drill
    player_count: int         # number of unique tracks active
    drill_type: str = "unknown"
    label: str = ""           # user-provided or auto-generated label


class DrillSegmenter:
    """
    Segment a training session video into individual drill periods
    based on team-wide activity patterns.

    Algorithm:
    1. Compute per-second average speed across all players
    2. Mark periods where avg speed < idle_threshold for > min_transition_s as "transitions"
    3. Periods between transitions are "drills"
    4. Merge drills shorter than min_drill_s with neighbors
    5. Classify drill type based on movement patterns
    """

    def __init__(self, idle_threshold: float = 1.0, min_transition_s: float = 10.0,
                 min_drill_s: float = 30.0):
        """
        Args:
            idle_threshold: Speed below which players are considered idle (m/s)
            min_transition_s: Minimum duration of low-activity to count as a transition
            min_drill_s: Minimum drill duration (shorter segments get merged)
        """
        self.idle_threshold = idle_threshold
        self.min_transition_s = min_transition_s
        self.min_drill_s = min_drill_s

    def segment(self, df: pd.DataFrame, fps: float) -> list[DrillSegment]:
        """
        Detect drill boundaries from tracking data.

        Args:
            df: Detection DataFrame with columns: frame, track_id, x_field, y_field
            fps: Video frames per second

        Returns:
            List of DrillSegment objects ordered by time
        """
        if df.empty or "x_field" not in df.columns:
            return []

        # Compute per-frame average speed
        speed_per_second = self._compute_activity_signal(df, fps)

        if len(speed_per_second) == 0:
            return []

        # Find transition (idle) periods
        is_idle = speed_per_second < self.idle_threshold
        transitions = self._find_runs(is_idle, min_length=int(self.min_transition_s))

        # Everything between transitions is a drill
        segments = self._extract_drill_periods(speed_per_second, transitions, df, fps)

        # Merge short drills
        segments = self._merge_short_segments(segments)

        # Classify drill types
        for seg in segments:
            seg.drill_type = self._classify_drill(df, seg, fps)

        return segments

    def _compute_activity_signal(self, df: pd.DataFrame, fps: float) -> np.ndarray:
        """
        Compute a 1-Hz activity signal (average speed in m/s per second).
        """
        frames = sorted(df["frame"].unique())
        if len(frames) < 2:
            return np.array([])

        min_frame = frames[0]
        max_frame = frames[-1]
        total_seconds = int((max_frame - min_frame) / fps) + 1

        speed_per_second = np.zeros(total_seconds)
        count_per_second = np.zeros(total_seconds)

        # Sort by track and frame for speed computation
        df_sorted = df.sort_values(["track_id", "frame"])

        for tid, grp in df_sorted.groupby("track_id"):
            if len(grp) < 2:
                continue

            x = grp["x_field"].values
            y = grp["y_field"].values
            t = grp["frame"].values

            # Compute instantaneous speed between consecutive detections
            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(t) / fps  # seconds between detections

            # Avoid division by zero
            valid = dt > 0
            speeds = np.zeros(len(dx))
            speeds[valid] = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]

            # Cap unrealistic speeds
            speeds = np.clip(speeds, 0, 10.0)

            # Assign to second bins
            for i, speed in enumerate(speeds):
                sec_idx = int((t[i] - min_frame) / fps)
                if 0 <= sec_idx < total_seconds:
                    speed_per_second[sec_idx] += speed
                    count_per_second[sec_idx] += 1

        # Average speed per second (avoid div by zero)
        valid_mask = count_per_second > 0
        speed_per_second[valid_mask] /= count_per_second[valid_mask]

        return speed_per_second

    def _find_runs(self, signal: np.ndarray, min_length: int) -> list[tuple[int, int]]:
        """Find consecutive runs of True in boolean signal, filtering by min length."""
        runs = []
        in_run = False
        start = 0

        for i, val in enumerate(signal):
            if val and not in_run:
                start = i
                in_run = True
            elif not val and in_run:
                if i - start >= min_length:
                    runs.append((start, i))
                in_run = False

        # Handle run at end
        if in_run and len(signal) - start >= min_length:
            runs.append((start, len(signal)))

        return runs

    def _extract_drill_periods(self, speed_signal: np.ndarray,
                               transitions: list[tuple[int, int]],
                               df: pd.DataFrame, fps: float) -> list[DrillSegment]:
        """Extract drill segments from the gaps between transitions."""
        total_seconds = len(speed_signal)
        min_frame = df["frame"].min()

        # Add boundaries at start and end
        boundaries = [(0, 0)] + transitions + [(total_seconds, total_seconds)]

        segments = []
        for i in range(len(boundaries) - 1):
            drill_start_s = boundaries[i][1]  # end of previous transition
            drill_end_s = boundaries[i + 1][0]  # start of next transition

            duration = drill_end_s - drill_start_s
            if duration < 5:  # skip tiny gaps
                continue

            start_frame = int(min_frame + drill_start_s * fps)
            end_frame = int(min_frame + drill_end_s * fps)

            # Compute stats for this segment
            seg_speed = speed_signal[drill_start_s:drill_end_s]
            avg_intensity = float(np.mean(seg_speed)) if len(seg_speed) > 0 else 0
            max_intensity = float(np.max(seg_speed)) if len(seg_speed) > 0 else 0

            # Count unique tracks in this period
            seg_df = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
            player_count = seg_df["track_id"].nunique() if not seg_df.empty else 0

            segments.append(DrillSegment(
                index=len(segments),
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_s=float(drill_start_s),
                end_time_s=float(drill_end_s),
                duration_s=float(duration),
                avg_intensity=avg_intensity,
                max_intensity=max_intensity,
                player_count=player_count,
            ))

        return segments

    def _merge_short_segments(self, segments: list[DrillSegment]) -> list[DrillSegment]:
        """Merge segments shorter than min_drill_s with their neighbors."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.duration_s < self.min_drill_s:
                # Merge with previous
                prev = merged[-1]
                prev.end_frame = seg.end_frame
                prev.end_time_s = seg.end_time_s
                prev.duration_s = prev.end_time_s - prev.start_time_s
                prev.avg_intensity = (prev.avg_intensity + seg.avg_intensity) / 2
                prev.max_intensity = max(prev.max_intensity, seg.max_intensity)
                prev.player_count = max(prev.player_count, seg.player_count)
            else:
                merged.append(seg)

        # Re-index
        for i, seg in enumerate(merged):
            seg.index = i

        return merged

    def _classify_drill(self, df: pd.DataFrame, segment: DrillSegment,
                        fps: float) -> str:
        """
        Classify drill type based on movement patterns.

        Rules:
        - High speed, short duration, linear movement → "sprint"
        - Clustered positions, moderate speed → "possession"
        - Spread out, frequent direction changes → "passing"
        - High speed bursts with rest → "agility"
        - Moderate speed, spread out → "tactical"
        """
        seg_df = df[(df["frame"] >= segment.start_frame) &
                    (df["frame"] <= segment.end_frame)].copy()

        if seg_df.empty or "x_field" not in seg_df.columns:
            return "unknown"

        # Compute spatial spread
        x_spread = seg_df["x_field"].std() if len(seg_df) > 1 else 0
        y_spread = seg_df["y_field"].std() if len(seg_df) > 1 else 0
        total_spread = np.sqrt(x_spread**2 + y_spread**2)

        avg_speed = segment.avg_intensity
        max_speed = segment.max_intensity
        duration = segment.duration_s

        # Classification rules
        if max_speed > 6.0 and duration < 60 and total_spread > 15:
            return "sprint"
        elif total_spread < 8 and avg_speed > 1.5:
            return "possession"
        elif avg_speed > 2.5 and total_spread > 10:
            return "passing"
        elif max_speed > 5.0 and avg_speed < 2.0:
            return "agility"
        elif total_spread > 12 and avg_speed > 1.0:
            return "tactical"
        else:
            return "general"
