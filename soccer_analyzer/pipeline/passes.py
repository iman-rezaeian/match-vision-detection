"""Pass inference from tracking data."""

import numpy as np
import pandas as pd
from typing import List


class PassDetector:
    def detect_passes(self, detections_df: pd.DataFrame, fps: float,
                      field_length: float, field_width: float) -> List[dict]:
        """
        Infer passes between identified players using movement patterns.

        Algorithm:
        1. Smooth player positions with rolling window
        2. Estimate ball position as centroid of closest player cluster
        3. Detect possession change events based on velocity patterns
        4. Filter: minimum 3m distance

        Returns: list of pass events
        """
        if detections_df.empty or "player_name" not in detections_df.columns:
            return []

        # Only process identified players (have player_name assigned, not Unknown)
        identified = detections_df[
            detections_df["player_name"].notna() &
            ~detections_df["player_name"].str.startswith("Unknown_", na=True)
        ].copy()
        if identified.empty:
            return []

        # Sort by time
        identified = identified.sort_values("time_s").reset_index(drop=True)

        # Smooth positions per player (rolling window of 5)
        passes = []
        players = identified["player_name"].unique()

        # Build position timeseries per player
        player_positions = {}
        for player in players:
            pdata = identified[identified["player_name"] == player].copy()
            if len(pdata) < 10:
                continue
            pdata = pdata.sort_values("time_s")
            # Rolling smooth
            pdata["x_smooth"] = pdata["x_field"].rolling(5, min_periods=1, center=True).mean()
            pdata["y_smooth"] = pdata["y_field"].rolling(5, min_periods=1, center=True).mean()
            player_positions[player] = pdata

        # Detect possession changes
        # Compute velocity for each player
        for player, pdata in player_positions.items():
            if len(pdata) < 3:
                continue
            dt = pdata["time_s"].diff().fillna(1.0 / fps)
            dx = pdata["x_smooth"].diff().fillna(0)
            dy = pdata["y_smooth"].diff().fillna(0)
            speed = np.sqrt(dx ** 2 + dy ** 2) / dt.clip(lower=0.01)
            player_positions[player] = pdata.assign(speed=speed)

        # Simple pass detection: find when ball-holder changes
        # Approximate ball position as position of player with highest "ball proximity score"
        # Use acceleration bursts as indicators of ball possession

        time_steps = sorted(identified["time_s"].unique())
        if len(time_steps) < 10:
            return passes

        # Sample at 1-second intervals
        sample_times = np.arange(time_steps[0], time_steps[-1], 1.0)
        possession_timeline = []

        for t in sample_times:
            # Find closest players at this time
            window = identified[
                (identified["time_s"] >= t - 0.5) &
                (identified["time_s"] < t + 0.5)
            ]
            if window.empty:
                continue

            # Estimate possession: player closest to group centroid with high activity
            team_players = window[window["team"] == "Home"]
            if team_players.empty:
                continue

            centroid_x = team_players["x_field"].mean()
            centroid_y = team_players["y_field"].mean()

            # Find player closest to centroid (crude ball-possession estimate)
            distances = np.sqrt(
                (team_players["x_field"] - centroid_x) ** 2 +
                (team_players["y_field"] - centroid_y) ** 2
            )

            # Player with minimum distance to group center (likely near ball)
            closest_idx = distances.idxmin()
            possessor = team_players.loc[closest_idx, "player_name"]
            pos_x = team_players.loc[closest_idx, "x_field"]
            pos_y = team_players.loc[closest_idx, "y_field"]

            possession_timeline.append({
                "time_s": t,
                "player": possessor,
                "x": pos_x,
                "y": pos_y,
                "team": "Home",
                "segment_id": int(window["segment_id"].mode().iloc[0]) if "segment_id" in window.columns else 0,
            })

        # Detect possession changes as passes
        for i in range(1, len(possession_timeline)):
            prev = possession_timeline[i - 1]
            curr = possession_timeline[i]

            # Skip pass inference across segment boundaries
            if prev.get("segment_id", 0) != curr.get("segment_id", 0):
                continue

            if prev["player"] != curr["player"] and prev["team"] == curr["team"]:
                # Calculate pass distance
                dist = np.sqrt(
                    (curr["x"] - prev["x"]) ** 2 +
                    (curr["y"] - prev["y"]) ** 2
                )

                # Filter: minimum 3m, maximum 25m
                if 3.0 <= dist <= 25.0:
                    passes.append({
                        "timestamp_s": curr["time_s"],
                        "passer_name": prev["player"],
                        "receiver_name": curr["player"],
                        "team": prev["team"],
                        "passer_pos": (prev["x"], prev["y"]),
                        "receiver_pos": (curr["x"], curr["y"]),
                        "pass_distance_m": round(dist, 1),
                    })

        # Also detect away team passes
        away_identified = identified[identified["team"] == "Away"]
        if not away_identified.empty:
            away_passes = self._detect_team_passes(away_identified, fps, "Away")
            passes.extend(away_passes)

        return passes

    def _detect_team_passes(self, team_df: pd.DataFrame, fps: float, team: str) -> list:
        """Detect passes for a specific team."""
        passes = []
        time_steps = sorted(team_df["time_s"].unique())
        if len(time_steps) < 10:
            return passes

        sample_times = np.arange(time_steps[0], time_steps[-1], 1.0)
        possession_timeline = []

        for t in sample_times:
            window = team_df[
                (team_df["time_s"] >= t - 0.5) &
                (team_df["time_s"] < t + 0.5)
            ]
            if window.empty:
                continue

            centroid_x = window["x_field"].mean()
            centroid_y = window["y_field"].mean()

            distances = np.sqrt(
                (window["x_field"] - centroid_x) ** 2 +
                (window["y_field"] - centroid_y) ** 2
            )

            closest_idx = distances.idxmin()
            possessor = window.loc[closest_idx, "player_name"]
            pos_x = window.loc[closest_idx, "x_field"]
            pos_y = window.loc[closest_idx, "y_field"]

            possession_timeline.append({
                "time_s": t,
                "player": possessor,
                "x": pos_x,
                "y": pos_y,
                "segment_id": int(window["segment_id"].mode().iloc[0]) if "segment_id" in window.columns else 0,
            })

        for i in range(1, len(possession_timeline)):
            prev = possession_timeline[i - 1]
            curr = possession_timeline[i]

            # Skip pass inference across segment boundaries
            if prev.get("segment_id", 0) != curr.get("segment_id", 0):
                continue

            if prev["player"] != curr["player"]:
                dist = np.sqrt(
                    (curr["x"] - prev["x"]) ** 2 +
                    (curr["y"] - prev["y"]) ** 2
                )

                if 3.0 <= dist <= 25.0:
                    passes.append({
                        "timestamp_s": curr["time_s"],
                        "passer_name": prev["player"],
                        "receiver_name": curr["player"],
                        "team": team,
                        "passer_pos": (prev["x"], prev["y"]),
                        "receiver_pos": (curr["x"], curr["y"]),
                        "pass_distance_m": round(dist, 1),
                    })

        return passes

    def build_pass_matrix(self, passes: list, players: list) -> pd.DataFrame:
        """
        NxN matrix where N = number of players on one team.
        Value = number of passes between player i and player j.
        """
        if not passes or not players:
            return pd.DataFrame()

        player_names = [p["name"] if isinstance(p, dict) else p for p in players]
        matrix = pd.DataFrame(0, index=player_names, columns=player_names)

        for p in passes:
            passer = p["passer_name"]
            receiver = p["receiver_name"]
            if passer in matrix.index and receiver in matrix.columns:
                matrix.loc[passer, receiver] += 1

        return matrix
