"""Formation shape visualization."""

import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from config import PITCH_GRASS, PITCH_LINES, TEAM_A_BLUE, TEAM_B_RED


def plot_formation(snapshot: dict, team: str,
                   field_length: float, field_width: float,
                   title: str = None) -> plt.Figure:
    """
    Plot a formation snapshot on a pitch.
    snapshot: {formation, positions, player_names, confidence}
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

    color = TEAM_A_BLUE if team == "Home" else TEAM_B_RED

    positions = snapshot.get("positions", [])
    player_names = snapshot.get("player_names", [])
    formation = snapshot.get("formation", "Unknown")
    confidence = snapshot.get("confidence", 0.0)

    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        ax.scatter(xs, ys, s=500, c=color, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)

        for i, (x, y) in enumerate(positions):
            if i < len(player_names):
                name = player_names[i].split(" ")[-1][:6]
            else:
                name = ""
            ax.text(x, y - 1.8, name, color="white", fontsize=8,
                    ha="center", va="top", fontweight="bold", zorder=6)

    title_text = title or f"{team} Formation: {formation}"
    if confidence > 0:
        title_text += f" ({confidence:.0%})"

    ax.set_title(title_text, color="white", fontsize=16,
                 fontweight="bold", pad=10)

    return fig


def plot_formation_timeline(timeline: list, team: str) -> plt.Figure:
    """
    Plot formation changes over time as a horizontal bar chart.
    timeline: list of {time_start_s, time_end_s, formation, confidence}
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    if not timeline:
        ax.text(0.5, 0.5, "No formation data", ha="center", va="center",
                color="white", transform=ax.transAxes)
        return fig

    # Assign colors to formations
    unique_formations = list(set(t["formation"] for t in timeline))
    formation_colors = {}
    color_palette = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#f44336",
                     "#00bcd4", "#ffeb3b", "#795548"]

    for i, f in enumerate(unique_formations):
        formation_colors[f] = color_palette[i % len(color_palette)]

    # Draw bars
    for entry in timeline:
        start_min = entry["time_start_s"] / 60.0
        duration_min = (entry["time_end_s"] - entry["time_start_s"]) / 60.0
        color = formation_colors[entry["formation"]]

        ax.barh(0.5, duration_min, left=start_min, height=0.6,
                color=color, edgecolor="#30363d", linewidth=0.5, alpha=0.85)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f) for f, c in formation_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9, ncol=len(unique_formations))

    ax.set_xlabel("Time (minutes)", color="white", fontsize=10)
    ax.set_title(f"{team} Formation Over Time", color="white",
                 fontsize=14, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(0, 1)

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_compactness_chart(compactness_data: list, team: str) -> plt.Figure:
    """Plot team compactness (avg inter-player distance) over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    if not compactness_data:
        ax.text(0.5, 0.5, "No compactness data", ha="center", va="center",
                color="white", transform=ax.transAxes)
        return fig

    times = [d["time_s"] / 60.0 for d in compactness_data]
    values = [d["compactness_m"] for d in compactness_data]

    color = TEAM_A_BLUE if team == "Home" else TEAM_B_RED

    ax.plot(times, values, color=color, linewidth=2, alpha=0.9)
    ax.fill_between(times, values, alpha=0.2, color=color)

    # Add reference line for "bunched" threshold
    avg_compact = np.mean(values)
    ax.axhline(y=avg_compact, color="#8b949e", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Avg: {avg_compact:.1f}m")

    ax.set_xlabel("Time (minutes)", color="white", fontsize=10)
    ax.set_ylabel("Avg Inter-player Distance (m)", color="white", fontsize=10)
    ax.set_title(f"{team} Compactness Over Time", color="white",
                 fontsize=14, fontweight="bold")

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    return fig
