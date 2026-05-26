"""Per-player time-in-zone timeline visualization."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import PITCH_GRASS, ACCENT_GREEN, TEAM_A_BLUE, TEAM_B_RED


def plot_zone_timeline(detections_df, player_name: str,
                       field_length: float,
                       window_seconds: float = 60.0) -> plt.Figure:
    """
    Plot a timeline showing which third of the field a player occupied over time.
    Color bands: Green = Attacking, Yellow = Midfield, Blue = Defensive
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    player_data = detections_df[detections_df["player_name"] == player_name]
    if player_data.empty:
        ax.text(0.5, 0.5, f"No data for {player_name}",
                ha="center", va="center", color="white", transform=ax.transAxes)
        return fig

    player_data = player_data.sort_values("time_s")
    max_time = player_data["time_s"].max()
    third_length = field_length / 3.0

    # Define zone colors
    colors = {"Attacking": "#ff5722", "Midfield": "#ffc107", "Defensive": "#2196f3"}

    # Calculate zone percentages per time window
    times = []
    att_pcts = []
    mid_pcts = []
    def_pcts = []

    for t_start in np.arange(0, max_time, window_seconds):
        t_end = t_start + window_seconds
        window = player_data[
            (player_data["time_s"] >= t_start) &
            (player_data["time_s"] < t_end)
        ]

        if window.empty:
            continue

        total = len(window)
        att = (window["x_field"] > 2 * third_length).sum() / total
        mid = ((window["x_field"] >= third_length) &
               (window["x_field"] <= 2 * third_length)).sum() / total
        def_ = (window["x_field"] < third_length).sum() / total

        times.append((t_start + t_end) / 2 / 60.0)  # Convert to minutes
        att_pcts.append(att)
        mid_pcts.append(mid)
        def_pcts.append(def_)

    if not times:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="white", transform=ax.transAxes)
        return fig

    # Stacked area chart
    ax.stackplot(times,
                 def_pcts, mid_pcts, att_pcts,
                 colors=[colors["Defensive"], colors["Midfield"], colors["Attacking"]],
                 alpha=0.8,
                 labels=["Defensive Third", "Midfield", "Attacking Third"])

    ax.set_xlabel("Time (minutes)", color="white", fontsize=10)
    ax.set_ylabel("Zone %", color="white", fontsize=10)
    ax.set_title(f"{player_name} — Zone Timeline", color="white",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(times) if times else 1)

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    return fig


def plot_team_zone_summary(stats_df, team: str) -> plt.Figure:
    """Plot zone distribution for all players on a team as horizontal bars."""
    team_stats = stats_df[stats_df["team"] == team].sort_values("jersey_number")

    if team_stats.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#161b22")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white")
        return fig

    fig, ax = plt.subplots(figsize=(10, max(4, len(team_stats) * 0.5)))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    players = team_stats["name"].values
    y_pos = np.arange(len(players))

    att = team_stats["pct_att_third"].values
    mid = team_stats["pct_mid_third"].values
    def_ = team_stats["pct_def_third"].values

    ax.barh(y_pos, def_, color="#2196f3", label="Defensive", alpha=0.8)
    ax.barh(y_pos, mid, left=def_, color="#ffc107", label="Midfield", alpha=0.8)
    ax.barh(y_pos, att, left=def_ + mid, color="#ff5722", label="Attacking", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(players, color="white", fontsize=9)
    ax.set_xlabel("Percentage (%)", color="white")
    ax.set_title(f"{team} — Zone Distribution", color="white",
                 fontsize=14, fontweight="bold")

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    return fig
