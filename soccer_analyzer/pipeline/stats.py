"""Per-player stat calculations from identified tracking data."""

import numpy as np
import pandas as pd
from config import SPRINT_THRESHOLD, MAX_SPEED_CAP


class StatsCalculator:
    def __init__(self, field_length: float, field_width: float, fps: float):
        self.field_length = field_length
        self.field_width = field_width
        self.fps = fps

    def calculate_all_stats(self, detections_df: pd.DataFrame,
                            passes: list = None) -> pd.DataFrame:
        """
        Calculate per-player statistics from identified tracking data.

        Input: detections_df with player_name, jersey_number, x_field, y_field, time_s
        Returns: DataFrame with one row per player
        """
        if detections_df.empty or "player_name" not in detections_df.columns:
            return pd.DataFrame()

        # Get unique players (exclude unknowns)
        player_groups = detections_df.groupby("player_name")
        stats_list = []

        for player_name, group in player_groups:
            if player_name.startswith("Unknown_"):
                continue

            group = group.sort_values("time_s").reset_index(drop=True)

            stats = self._calculate_player_stats(player_name, group, passes)
            stats_list.append(stats)

        return pd.DataFrame(stats_list) if stats_list else pd.DataFrame()

    def _calculate_player_stats(self, player_name: str, group: pd.DataFrame,
                                passes: list = None) -> dict:
        """Calculate stats for a single player."""
        # Basic info
        jersey_number = int(group["jersey_number"].iloc[0]) if "jersey_number" in group.columns else 0
        team = group["team"].iloc[0] if "team" in group.columns else "Unknown"
        id_confidence = group["id_confidence"].mean() if "id_confidence" in group.columns else 0.0

        # Minutes played (from timestamps)
        time_range = group["time_s"].max() - group["time_s"].min()
        minutes_played = time_range / 60.0

        # Number of stints (breaks in tracking indicate substitution)
        time_diffs = group["time_s"].diff()
        # If segment_id column exists, don't count segment transitions as stints
        if "segment_id" in group.columns:
            segment_change = group["segment_id"].diff().fillna(0) != 0
            stint_breaks = ((time_diffs > 30) & ~segment_change).sum()
        else:
            stint_breaks = (time_diffs > 30).sum()  # Gap > 30s = new stint
        stints = stint_breaks + 1

        # Distance and speed
        dx = group["x_field"].diff().fillna(0)
        dy = group["y_field"].diff().fillna(0)
        dt = group["time_s"].diff().fillna(1.0 / self.fps)

        # Zero out deltas across segment boundaries (no real movement there)
        if "segment_id" in group.columns:
            seg_change = group["segment_id"].diff().fillna(0) != 0
            dx[seg_change] = 0
            dy[seg_change] = 0
            dt[seg_change] = 1.0 / self.fps  # avoid division by zero

        # Zero out deltas across track_id changes (jumps from re-identification)
        if "track_id" in group.columns:
            tid_change = group["track_id"].diff().fillna(0) != 0
            dx[tid_change] = 0
            dy[tid_change] = 0

        # Distance between consecutive points
        distances = np.sqrt(dx ** 2 + dy ** 2)

        # Cap per-sample displacement based on max speed and sample interval
        max_displacement = MAX_SPEED_CAP * dt.clip(lower=0.01)
        distances = distances.clip(upper=max_displacement)

        # Speed calculation with rolling median filter
        speeds = distances / dt.clip(lower=0.01)
        speeds = speeds.clip(upper=MAX_SPEED_CAP)
        speeds_smoothed = speeds.rolling(5, min_periods=1, center=True).median()

        # Use smoothed speeds * dt for consistent distance
        distances_smoothed = speeds_smoothed * dt.clip(lower=0.01)
        total_distance = distances_smoothed.sum()

        top_speed = float(np.percentile(speeds_smoothed[speeds_smoothed > 0], 95)) if (speeds_smoothed > 0).any() else 0.0
        avg_speed = float(speeds_smoothed.mean())

        # Sprints (speed > threshold)
        sprint_mask = speeds_smoothed > SPRINT_THRESHOLD
        sprint_count = int(sprint_mask.sum())
        sprint_distance = float(distances_smoothed[sprint_mask].sum())

        # Zone percentages (thirds of the field)
        third_length = self.field_length / 3.0
        att_third = (group["x_field"] > 2 * third_length).mean() * 100
        mid_third = ((group["x_field"] >= third_length) &
                     (group["x_field"] <= 2 * third_length)).mean() * 100
        def_third = (group["x_field"] < third_length).mean() * 100

        # Average position
        avg_x = float(group["x_field"].mean())
        avg_y = float(group["y_field"].mean())

        # Positional spread (std dev)
        positional_spread = float(np.sqrt(
            group["x_field"].std() ** 2 + group["y_field"].std() ** 2
        ))

        # Pass stats
        passes_made = 0
        passes_received = 0
        if passes:
            passes_made = sum(1 for p in passes if p["passer_name"] == player_name)
            passes_received = sum(1 for p in passes if p["receiver_name"] == player_name)

        return {
            "jersey_number": jersey_number,
            "name": player_name,
            "team": team,
            "minutes_played": round(minutes_played, 1),
            "distance_m": round(total_distance, 0),
            "top_speed_ms": round(top_speed, 2),
            "avg_speed_ms": round(avg_speed, 2),
            "sprint_count": sprint_count,
            "sprint_distance_m": round(sprint_distance, 0),
            "pct_att_third": round(att_third, 1),
            "pct_mid_third": round(mid_third, 1),
            "pct_def_third": round(def_third, 1),
            "avg_x": round(avg_x, 1),
            "avg_y": round(avg_y, 1),
            "positional_spread_m": round(positional_spread, 1),
            "passes_made": passes_made,
            "passes_received": passes_received,
            "id_confidence": round(id_confidence, 3),
            "stints": stints,
            "player_id": int(group["player_id"].iloc[0]) if "player_id" in group.columns and group["player_id"].notna().any() and str(group["player_id"].iloc[0]).strip() != "" else None,
        }
