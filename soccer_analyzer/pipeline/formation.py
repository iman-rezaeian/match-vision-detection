"""Formation detection and labeling for youth soccer."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple


class FormationDetector:
    # U10 7v7 common formations (normalized positions 0-1) — 6 outfield
    FORMATIONS_7V7 = {
        "2-3-1": {
            "positions": [(0.15, 0.35), (0.15, 0.65),  # 2 defenders
                         (0.45, 0.2), (0.45, 0.5), (0.45, 0.8),  # 3 midfielders
                         (0.75, 0.5)],  # 1 forward
            "lines": [2, 3, 1],
        },
        "3-2-1": {
            "positions": [(0.15, 0.25), (0.15, 0.5), (0.15, 0.75),  # 3 defenders
                         (0.45, 0.35), (0.45, 0.65),  # 2 midfielders
                         (0.75, 0.5)],  # 1 forward
            "lines": [3, 2, 1],
        },
        "2-2-2": {
            "positions": [(0.15, 0.35), (0.15, 0.65),  # 2 defenders
                         (0.45, 0.35), (0.45, 0.65),  # 2 midfielders
                         (0.75, 0.35), (0.75, 0.65)],  # 2 forwards
            "lines": [2, 2, 2],
        },
        "3-1-2": {
            "positions": [(0.15, 0.25), (0.15, 0.5), (0.15, 0.75),  # 3 defenders
                         (0.45, 0.5),  # 1 midfielder
                         (0.75, 0.35), (0.75, 0.65)],  # 2 forwards
            "lines": [3, 1, 2],
        },
        "1-3-2": {
            "positions": [(0.15, 0.5),  # 1 defender
                         (0.4, 0.25), (0.4, 0.5), (0.4, 0.75),  # 3 midfielders
                         (0.7, 0.35), (0.7, 0.65)],  # 2 forwards
            "lines": [1, 3, 2],
        },
    }

    # 9v9 common formations (normalized positions 0-1) — 8 outfield
    FORMATIONS_9V9 = {
        "3-3-2": {
            "positions": [(0.15, 0.25), (0.15, 0.5), (0.15, 0.75),
                         (0.45, 0.25), (0.45, 0.5), (0.45, 0.75),
                         (0.75, 0.35), (0.75, 0.65)],
            "lines": [3, 3, 2],
        },
        "3-2-3": {
            "positions": [(0.15, 0.25), (0.15, 0.5), (0.15, 0.75),
                         (0.45, 0.35), (0.45, 0.65),
                         (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)],
            "lines": [3, 2, 3],
        },
        "2-4-2": {
            "positions": [(0.15, 0.35), (0.15, 0.65),
                         (0.4, 0.15), (0.4, 0.38), (0.4, 0.62), (0.4, 0.85),
                         (0.75, 0.35), (0.75, 0.65)],
            "lines": [2, 4, 2],
        },
        "3-4-1": {
            "positions": [(0.15, 0.25), (0.15, 0.5), (0.15, 0.75),
                         (0.4, 0.15), (0.4, 0.38), (0.4, 0.62), (0.4, 0.85),
                         (0.75, 0.5)],
            "lines": [3, 4, 1],
        },
        "4-3-1": {
            "positions": [(0.15, 0.15), (0.15, 0.38), (0.15, 0.62), (0.15, 0.85),
                         (0.45, 0.25), (0.45, 0.5), (0.45, 0.75),
                         (0.75, 0.5)],
            "lines": [4, 3, 1],
        },
    }

    def __init__(self, players_per_team: int = 7):
        self.players_per_team = players_per_team
        self._templates = self.FORMATIONS_9V9 if players_per_team >= 9 else self.FORMATIONS_7V7

    def detect_formation(self, team_positions: np.ndarray) -> Tuple[str, float]:
        """
        Input: array of (x, y) positions for outfield players of one team (6 players, excluding GK)
        1. Sort players by x position (defensive → attacking)
        2. Use KMeans to cluster into defensive/mid/attacking lines
        3. Count players per line
        4. Match to closest known formation template
        5. Return formation string e.g. "2-3-1" + confidence
        """
        if len(team_positions) < 4:
            return "Unknown", 0.0

        # Normalize positions to 0-1
        positions = np.array(team_positions, dtype=np.float32)
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)

        norm_positions = np.column_stack([
            (positions[:, 0] - x_min) / x_range,
            (positions[:, 1] - y_min) / y_range,
        ])

        # Remove goalkeeper (deepest player) if we have enough players
        if len(norm_positions) >= self.players_per_team:
            # GK is the player with lowest x (deepest)
            gk_idx = np.argmin(norm_positions[:, 0])
            norm_positions = np.delete(norm_positions, gk_idx, axis=0)

        # Cluster into lines using x-coordinate
        n_clusters = min(3, len(norm_positions))
        if n_clusters < 2:
            return "Unknown", 0.0

        x_values = norm_positions[:, 0].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_values)

        # Sort clusters by x position (defensive to attacking)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_clusters = np.argsort(cluster_centers)

        # Count players per line
        lines = []
        for cluster_idx in sorted_clusters:
            count = (labels == cluster_idx).sum()
            lines.append(int(count))

        # Match to known formations
        formation_str = "-".join(map(str, lines))
        confidence = self._match_to_template(norm_positions, lines)

        return formation_str, confidence

    def _match_to_template(self, positions: np.ndarray, detected_lines: list) -> float:
        """Match detected formation to closest template. Return confidence 0-1."""
        best_score = 0.0

        for name, template in self._templates.items():
            if template["lines"] == detected_lines:
                # Line structure matches exactly
                best_score = max(best_score, 0.9)
            elif len(template["lines"]) == len(detected_lines):
                # Compare line counts
                diff = sum(abs(a - b) for a, b in zip(template["lines"], detected_lines))
                score = max(0, 1.0 - diff * 0.2)
                best_score = max(best_score, score)

        return best_score

    def formation_over_time(self, detections_df: pd.DataFrame,
                            team: str = "Home",
                            window_seconds: float = 60.0) -> list:
        """
        Slide a 60-second window across the game.
        Detect formation in each window.
        Return timeline of formation changes.
        """
        if detections_df.empty:
            return []

        team_df = detections_df[detections_df["team"] == team]
        if team_df.empty:
            return []

        max_time = team_df["time_s"].max()
        timeline = []

        for t_start in np.arange(0, max_time - window_seconds, window_seconds / 2):
            t_end = t_start + window_seconds
            window = team_df[
                (team_df["time_s"] >= t_start) &
                (team_df["time_s"] < t_end)
            ]

            if window.empty:
                continue

            # Get average positions per player in this window
            player_avg = window.groupby("player_name")[["x_field", "y_field"]].mean()
            positions = player_avg.values

            if len(positions) >= 4:
                formation, confidence = self.detect_formation(positions)
                timeline.append({
                    "time_start_s": t_start,
                    "time_end_s": t_end,
                    "formation": formation,
                    "confidence": confidence,
                    "n_players": len(positions),
                })

        return timeline

    def compactness_over_time(self, detections_df: pd.DataFrame,
                              team: str = "Home",
                              window_seconds: float = 30.0) -> list:
        """
        Calculate team compactness (average inter-player distance) over time.
        Return time series of compactness values.
        """
        if detections_df.empty:
            return []

        team_df = detections_df[detections_df["team"] == team]
        if team_df.empty:
            return []

        max_time = team_df["time_s"].max()
        compactness_series = []

        for t_start in np.arange(0, max_time - window_seconds, window_seconds / 2):
            t_end = t_start + window_seconds
            window = team_df[
                (team_df["time_s"] >= t_start) &
                (team_df["time_s"] < t_end)
            ]

            if window.empty:
                continue

            # Average position per player
            player_avg = window.groupby("player_name")[["x_field", "y_field"]].mean()
            positions = player_avg.values

            if len(positions) < 3:
                continue

            # Calculate average inter-player distance
            from scipy.spatial.distance import pdist
            distances = pdist(positions)
            avg_distance = float(distances.mean()) if len(distances) > 0 else 0.0

            compactness_series.append({
                "time_s": (t_start + t_end) / 2,
                "compactness_m": avg_distance,
                "n_players": len(positions),
            })

        return compactness_series

    def get_formation_snapshot(self, detections_df: pd.DataFrame,
                               team: str, time_s: float,
                               window: float = 30.0) -> dict:
        """Get formation at a specific time point."""
        team_df = detections_df[detections_df["team"] == team]
        snapshot = team_df[
            (team_df["time_s"] >= time_s - window / 2) &
            (team_df["time_s"] < time_s + window / 2)
        ]

        if snapshot.empty:
            return {"formation": "Unknown", "positions": [], "confidence": 0.0}

        player_avg = snapshot.groupby("player_name")[["x_field", "y_field"]].mean()
        positions = player_avg.values.tolist()
        player_names = player_avg.index.tolist()

        formation, confidence = self.detect_formation(np.array(positions))

        return {
            "formation": formation,
            "positions": positions,
            "player_names": player_names,
            "confidence": confidence,
        }
