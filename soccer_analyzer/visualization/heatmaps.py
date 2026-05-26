"""Per-player heatmap visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from config import PITCH_GRASS, PITCH_LINES, TEAM_A_BLUE, TEAM_B_RED


def plot_player_heatmap(player_data, player_name: str, team: str,
                        field_length: float, field_width: float) -> plt.Figure:
    """
    Generate a heatmap for a single player.
    player_data: DataFrame rows for this player with x_field, y_field columns.
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

    if player_data.empty:
        ax.set_title(f"{player_name} - No Data", color="white", fontsize=14)
        return fig

    x = player_data["x_field"].values
    y = player_data["y_field"].values

    # Choose colormap based on team
    if team == "Home":
        cmap = LinearSegmentedColormap.from_list("blue_heat",
            ["#1a2a1a", "#1565c0", "#42a5f5", "#90caf9", "#ffffff"])
    else:
        cmap = LinearSegmentedColormap.from_list("red_heat",
            ["#1a2a1a", "#c62828", "#ef5350", "#ef9a9a", "#ffffff"])

    # KDE heatmap
    try:
        pitch.kdeplot(x, y, ax=ax, cmap=cmap, fill=True, levels=50, alpha=0.7)
    except Exception:
        # Fallback to scatter if KDE fails
        pitch.scatter(x, y, ax=ax, c=TEAM_A_BLUE if team == "Home" else TEAM_B_RED,
                      alpha=0.3, s=5)

    # Mark average position
    avg_x, avg_y = x.mean(), y.mean()
    pitch.scatter([avg_x], [avg_y], ax=ax, c="white", s=200,
                  edgecolors="black", linewidth=2, zorder=10, marker="o")

    ax.set_title(f"{player_name}", color="white", fontsize=16,
                 fontweight="bold", pad=10)

    return fig


def plot_all_heatmaps_grid(detections_df, team: str,
                           field_length: float, field_width: float,
                           cols: int = 4) -> plt.Figure:
    """
    Generate a grid of mini heatmaps for all players on a team.
    """
    team_df = detections_df[detections_df["team"] == team]
    players = sorted(team_df["player_name"].unique())

    if not players:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No players found", ha="center", va="center", color="white")
        fig.patch.set_facecolor(PITCH_GRASS)
        return fig

    rows = (len(players) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    fig.patch.set_facecolor(PITCH_GRASS)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, player_name in enumerate(players):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx, col_idx]

        pitch = Pitch(
            pitch_type="custom",
            pitch_length=field_length,
            pitch_width=field_width,
            pitch_color=PITCH_GRASS,
            line_color=PITCH_LINES,
            linewidth=0.5,
        )
        pitch.draw(ax=ax)

        pdata = team_df[team_df["player_name"] == player_name]
        if not pdata.empty:
            x = pdata["x_field"].values
            y = pdata["y_field"].values

            color = TEAM_A_BLUE if team == "Home" else TEAM_B_RED
            ax.scatter(x, y, c=color, alpha=0.2, s=2)
            ax.scatter([x.mean()], [y.mean()], c="white", s=50,
                       edgecolors="black", linewidth=1, zorder=10)

        ax.set_title(player_name.split(" ")[-1], color="white",
                     fontsize=9, fontweight="bold", pad=2)
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)

    # Hide unused axes
    for idx in range(len(players), rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    return fig
