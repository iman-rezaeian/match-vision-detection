"""Passing network visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from config import PITCH_GRASS, PITCH_LINES, TEAM_A_BLUE, TEAM_B_RED


def plot_passing_network(pass_matrix: pd.DataFrame, player_positions: dict,
                         team: str, field_length: float,
                         field_width: float) -> plt.Figure:
    """
    Draw passing network on pitch.

    pass_matrix: NxN DataFrame of pass counts.
    player_positions: dict {player_name: (avg_x, avg_y)}
    team: "Home" or "Away"
    """
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=field_length,
        pitch_width=field_width,
        pitch_color=PITCH_GRASS,
        line_color=PITCH_LINES,
        linewidth=1,
    )

    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor(PITCH_GRASS)

    if pass_matrix.empty:
        ax.set_title(f"{team} Passing Network - No Data", color="white", fontsize=14)
        return fig

    color = TEAM_A_BLUE if team == "Home" else TEAM_B_RED

    players_in_matrix = list(pass_matrix.index)

    # Draw connections (passes)
    max_passes = pass_matrix.values.max() if pass_matrix.values.max() > 0 else 1

    for i, passer in enumerate(players_in_matrix):
        for j, receiver in enumerate(players_in_matrix):
            if i == j:
                continue
            count = pass_matrix.loc[passer, receiver]
            if count < 2:  # Minimum 2 passes to show connection
                continue

            if passer in player_positions and receiver in player_positions:
                x1, y1 = player_positions[passer]
                x2, y2 = player_positions[receiver]

                # Line width proportional to pass count
                width = 1 + (count / max_passes) * 5
                alpha = 0.3 + (count / max_passes) * 0.5

                ax.plot([x1, x2], [y1, y2], color=color,
                        linewidth=width, alpha=alpha, zorder=2)

    # Draw player nodes
    for player_name in players_in_matrix:
        if player_name not in player_positions:
            continue

        x, y = player_positions[player_name]
        total_passes = pass_matrix.loc[player_name].sum() + pass_matrix[player_name].sum()

        # Node size proportional to involvement
        size = 100 + total_passes * 10

        ax.scatter([x], [y], s=size, c=color, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)

        # Player label
        short_name = player_name.split(" ")[-1][:8]
        ax.text(x, y - 1.5, short_name, color="white", fontsize=7,
                ha="center", va="top", fontweight="bold", zorder=6)

    ax.set_title(f"{team} Passing Network", color="white",
                 fontsize=16, fontweight="bold", pad=10)

    return fig


def plot_pass_matrix_table(pass_matrix: pd.DataFrame, team: str) -> plt.Figure:
    """Render pass matrix as a styled table figure."""
    if pass_matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No passing data", ha="center", va="center", color="white")
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")
        ax.axis("off")
        return fig

    # Shorten names for display
    short_names = [n.split(" ")[-1][:6] for n in pass_matrix.index]
    display_matrix = pass_matrix.copy()
    display_matrix.index = short_names
    display_matrix.columns = short_names

    fig, ax = plt.subplots(figsize=(max(8, len(short_names) * 1.2),
                                     max(4, len(short_names) * 0.8)))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    # Create table
    table = ax.table(
        cellText=display_matrix.values.astype(int),
        rowLabels=display_matrix.index,
        colLabels=display_matrix.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style cells
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        cell.set_facecolor("#161b22")
        cell.set_text_props(color="white")

    ax.axis("off")
    ax.set_title(f"{team} Pass Matrix", color="white", fontsize=14,
                 fontweight="bold", pad=10)

    plt.tight_layout()
    return fig
