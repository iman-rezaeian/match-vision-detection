"""All players average positions pitch overview."""

import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from config import PITCH_GRASS, PITCH_LINES, TEAM_A_BLUE, TEAM_B_RED


def plot_pitch_overview(detections_df, field_length: float,
                        field_width: float) -> plt.Figure:
    """
    Plot all players' average positions on a single pitch.
    Both teams shown with different colors.
    """
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=field_length,
        pitch_width=field_width,
        pitch_color=PITCH_GRASS,
        line_color=PITCH_LINES,
        linewidth=1,
    )

    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor(PITCH_GRASS)

    if detections_df.empty or "player_name" not in detections_df.columns:
        ax.set_title("Average Positions - No Data", color="white", fontsize=14)
        return fig

    # Calculate average positions per player
    player_avg = detections_df.groupby(["player_name", "team"]).agg({
        "x_field": "mean",
        "y_field": "mean",
        "jersey_number": "first",
    }).reset_index()

    # Plot Home team
    home = player_avg[player_avg["team"] == "Home"]
    if not home.empty:
        ax.scatter(home["x_field"], home["y_field"],
                   s=400, c=TEAM_A_BLUE, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)
        for _, row in home.iterrows():
            name_parts = row["player_name"].split(" ")
            short_name = name_parts[-1][:8] if len(name_parts) > 1 else name_parts[0][:8]
            jersey = int(row["jersey_number"]) if row["jersey_number"] else ""

            # Jersey number inside circle
            ax.text(row["x_field"], row["y_field"], str(jersey),
                    color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold", zorder=6)
            # Name below
            ax.text(row["x_field"], row["y_field"] - 2,
                    short_name, color="white", fontsize=7,
                    ha="center", va="top", zorder=6)

    # Plot Away team
    away = player_avg[player_avg["team"] == "Away"]
    if not away.empty:
        ax.scatter(away["x_field"], away["y_field"],
                   s=400, c=TEAM_B_RED, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)
        for _, row in away.iterrows():
            name_parts = row["player_name"].split(" ")
            short_name = name_parts[-1][:8] if len(name_parts) > 1 else name_parts[0][:8]

            ax.text(row["x_field"], row["y_field"], "?",
                    color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold", zorder=6)
            ax.text(row["x_field"], row["y_field"] - 2,
                    short_name, color="white", fontsize=7,
                    ha="center", va="top", zorder=6)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TEAM_A_BLUE, edgecolor="white", label="Home"),
        Patch(facecolor=TEAM_B_RED, edgecolor="white", label="Away"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=10)

    ax.set_title("Average Positions", color="white",
                 fontsize=16, fontweight="bold", pad=10)

    return fig
