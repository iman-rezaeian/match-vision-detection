"""Test output visualization utilities."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Optional
import pandas as pd


def save_undistortion_comparison(raw_frame: np.ndarray, undistorted_frame: np.ndarray,
                                  output_path: str, title: str = "Fisheye Undistortion"):
    """Save side-by-side comparison of raw vs undistorted frame."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("#0d1117")

    # Raw frame
    raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Raw (Distorted)", color="white", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Undistorted frame
    undist_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
    axes[1].imshow(undist_rgb)
    axes[1].set_title("Undistorted", color="white", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(title, color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_detection_overlay(frame: np.ndarray, detections: list,
                            ground_truth: list, output_path: str,
                            title: str = "Detection Results"):
    """
    Save frame with detection bounding boxes overlaid.
    Green = matched detections, Red = false positives, Blue = missed (ground truth only)
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.patch.set_facecolor("#0d1117")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)

    # Draw ground truth (blue dashed)
    for bbox in ground_truth:
        x1, y1, x2, y2 = bbox[:4]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor="cyan",
                              facecolor="none", linestyle="--")
        ax.add_patch(rect)

    # Draw detections (green solid)
    for bbox in detections:
        x1, y1, x2, y2 = bbox[:4]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor="lime",
                              facecolor="none")
        ax.add_patch(rect)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(edgecolor="lime", facecolor="none", linewidth=2, label="Detections"),
        Patch(edgecolor="cyan", facecolor="none", linewidth=2, linestyle="--", label="Ground Truth"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    ax.set_title(f"{title} | Detected: {len(detections)} | GT: {len(ground_truth)}",
                 color="white", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_tracking_trajectories(detections_df: pd.DataFrame, output_path: str,
                                field_length: float = 105.0, field_width: float = 68.0,
                                title: str = "Tracking Trajectories"):
    """
    Save all player tracking trajectories on a pitch plot.
    Each track_id gets a unique color.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#1a2a1a")

    # Draw pitch outline
    ax.plot([0, field_length], [0, 0], color="#4a7a4a", linewidth=1)
    ax.plot([0, field_length], [field_width, field_width], color="#4a7a4a", linewidth=1)
    ax.plot([0, 0], [0, field_width], color="#4a7a4a", linewidth=1)
    ax.plot([field_length, field_length], [0, field_width], color="#4a7a4a", linewidth=1)
    # Center line
    ax.plot([field_length / 2, field_length / 2], [0, field_width],
            color="#4a7a4a", linewidth=1, linestyle="--")

    if "x_field" not in detections_df.columns:
        ax.text(field_length / 2, field_width / 2, "No field coordinates",
                color="white", ha="center", va="center", fontsize=14)
    else:
        track_ids = detections_df["track_id"].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(track_ids))))

        for i, tid in enumerate(track_ids):
            track_data = detections_df[detections_df["track_id"] == tid].sort_values("frame")
            color = colors[i % len(colors)]

            ax.plot(track_data["x_field"], track_data["y_field"],
                    color=color, alpha=0.6, linewidth=1)
            # Start point
            if not track_data.empty:
                ax.scatter(track_data["x_field"].iloc[0], track_data["y_field"].iloc[0],
                           color=color, s=30, zorder=5, edgecolors="white", linewidth=0.5)

    ax.set_xlim(-2, field_length + 2)
    ax.set_ylim(-2, field_width + 2)
    ax.set_xlabel("Field Length (m)", color="white")
    ax.set_ylabel("Field Width (m)", color="white")
    ax.set_title(title, color="white", fontsize=16, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")

    n_tracks = detections_df["track_id"].nunique() if "track_id" in detections_df.columns else 0
    ax.text(0.02, 0.98, f"Tracks: {n_tracks}", transform=ax.transAxes,
            color="#00c853", fontsize=12, va="top", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_position_error_heatmap(errors_by_position: list, output_path: str,
                                 field_length: float = 105.0, field_width: float = 68.0):
    """
    Visualize where on the field position errors are highest.
    errors_by_position: list of (x, y, error_m)
    """
    if not errors_by_position:
        return

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#1a2a1a")

    # Draw pitch outline
    ax.plot([0, field_length], [0, 0], color="#4a7a4a", linewidth=1)
    ax.plot([0, field_length], [field_width, field_width], color="#4a7a4a", linewidth=1)
    ax.plot([0, 0], [0, field_width], color="#4a7a4a", linewidth=1)
    ax.plot([field_length, field_length], [0, field_width], color="#4a7a4a", linewidth=1)

    xs = [e[0] for e in errors_by_position]
    ys = [e[1] for e in errors_by_position]
    errs = [e[2] for e in errors_by_position]

    scatter = ax.scatter(xs, ys, c=errs, cmap="RdYlGn_r", s=20, alpha=0.7,
                          vmin=0, vmax=3.0)
    plt.colorbar(scatter, ax=ax, label="Position Error (m)")

    ax.set_xlim(-2, field_length + 2)
    ax.set_ylim(-2, field_width + 2)
    ax.set_title("Position Error by Field Location", color="white",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Field Length (m)", color="white")
    ax.set_ylabel("Field Width (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_metrics_summary(metrics: dict, output_path: str):
    """Save a visual summary card of all metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "Pipeline Accuracy Report", ha="center", va="top",
            fontsize=20, fontweight="bold", color="white", transform=ax.transAxes)

    # Metrics as text
    y_pos = 0.85
    line_height = 0.06

    sections = [
        ("DETECTION", metrics.get("player_detection", {})),
        ("HOMOGRAPHY", metrics.get("homography", {})),
        ("TRACKING", metrics.get("tracking", {})),
    ]

    for section_name, section_data in sections:
        ax.text(0.05, y_pos, section_name, fontsize=14, fontweight="bold",
                color="#00c853", transform=ax.transAxes)
        y_pos -= line_height

        for key, value in section_data.items():
            if isinstance(value, float):
                text = f"  {key}: {value:.3f}"
            else:
                text = f"  {key}: {value}"
            ax.text(0.05, y_pos, text, fontsize=11, color="white",
                    transform=ax.transAxes, family="monospace")
            y_pos -= line_height * 0.8

        y_pos -= line_height * 0.5

    # Overall verdict
    verdict = metrics.get("overall_verdict", "UNKNOWN")
    color = "#00c853" if verdict == "PASS" else "#f44336" if verdict == "FAIL" else "#ff9800"
    ax.text(0.5, 0.05, f"VERDICT: {verdict}", ha="center", va="bottom",
            fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {output_path}")
