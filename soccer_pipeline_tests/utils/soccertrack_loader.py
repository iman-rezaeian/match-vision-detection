"""SoccerTrack dataset loader utilities.

Handles the specific SoccerTrack annotation format:
- Multi-level CSV headers: TeamID, PlayerID, Attributes (bb_height, bb_left, bb_top, bb_width)
- Frame index starting at 1
- Videos in videos/ subdirectory, annotations in annotations/
- Keypoints JSON for homography calibration
- GNSS CSV for ground truth positions
"""

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_soccertrack_annotations(annotation_path: str) -> Dict[int, List[tuple]]:
    """
    Load SoccerTrack bounding box annotations from multi-level CSV.

    Returns:
        Dict mapping frame_id -> list of (x1, y1, x2, y2) bounding boxes
    """
    ann_path = Path(annotation_path)
    df = pd.read_csv(str(ann_path), header=[0, 1, 2], index_col=0)

    annotations = {}
    for frame_idx in df.index:
        frame_data = df.loc[frame_idx]
        bboxes = []

        # Iterate over all players (team 0 and team 1)
        teams = set(df.columns.get_level_values(0)) - {"BALL"}
        for team_id in sorted(teams):
            team_cols = df.columns[df.columns.get_level_values(0) == team_id]
            player_ids = sorted(set(team_cols.get_level_values(1)))

            for player_id in player_ids:
                try:
                    bb_left = frame_data[(team_id, player_id, "bb_left")]
                    bb_top = frame_data[(team_id, player_id, "bb_top")]
                    bb_width = frame_data[(team_id, player_id, "bb_width")]
                    bb_height = frame_data[(team_id, player_id, "bb_height")]

                    # Skip NaN or zero-area boxes
                    if pd.isna(bb_left) or pd.isna(bb_top) or bb_width <= 0 or bb_height <= 0:
                        continue

                    x1 = float(bb_left)
                    y1 = float(bb_top)
                    x2 = x1 + float(bb_width)
                    y2 = y1 + float(bb_height)
                    bboxes.append((x1, y1, x2, y2))
                except (KeyError, TypeError):
                    continue

        if bboxes:
            annotations[int(frame_idx)] = bboxes

    return annotations


def load_soccertrack_tracking_gt(annotation_path: str) -> pd.DataFrame:
    """
    Load SoccerTrack annotations as a tracking ground truth DataFrame.

    Returns:
        DataFrame with columns: frame, gt_id, x, y, w, h, team_id
    """
    ann_path = Path(annotation_path)
    df = pd.read_csv(str(ann_path), header=[0, 1, 2], index_col=0)

    records = []
    player_counter = 0
    player_id_map = {}  # (team, player) -> unique_id

    teams = sorted(set(df.columns.get_level_values(0)) - {"BALL"})

    for team_id in teams:
        team_cols = df.columns[df.columns.get_level_values(0) == team_id]
        player_ids = sorted(set(team_cols.get_level_values(1)))

        for player_id in player_ids:
            key = (team_id, player_id)
            if key not in player_id_map:
                player_id_map[key] = player_counter
                player_counter += 1

            unique_id = player_id_map[key]

            for frame_idx in df.index:
                try:
                    bb_left = df.loc[frame_idx, (team_id, player_id, "bb_left")]
                    bb_top = df.loc[frame_idx, (team_id, player_id, "bb_top")]
                    bb_width = df.loc[frame_idx, (team_id, player_id, "bb_width")]
                    bb_height = df.loc[frame_idx, (team_id, player_id, "bb_height")]

                    if pd.isna(bb_left) or pd.isna(bb_top) or bb_width <= 0 or bb_height <= 0:
                        continue

                    cx = float(bb_left) + float(bb_width) / 2
                    cy = float(bb_top) + float(bb_height) / 2

                    records.append({
                        "frame": int(frame_idx),
                        "gt_id": unique_id,
                        "x": cx,
                        "y": cy,
                        "w": float(bb_width),
                        "h": float(bb_height),
                        "team_id": int(team_id),
                    })
                except (KeyError, TypeError):
                    continue

    return pd.DataFrame(records)


def load_fisheye_keypoints(keypoints_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load fisheye_keypoints.json — pixel/field point correspondences.

    Returns:
        (pixel_points, field_points): Nx2 arrays
        pixel_points: pixel coordinates in image
        field_points: real-world field coordinates in meters
    """
    with open(keypoints_path, "r") as f:
        data = json.load(f)

    pixel_points = []
    field_points = []

    for field_coord_str, pixel_coord in data.items():
        # Parse field coordinate from string like "(52.5, 34.0)"
        field_coord_str = field_coord_str.strip("()")
        parts = field_coord_str.split(",")
        fx = float(parts[0].strip())
        fy = float(parts[1].strip())

        px = float(pixel_coord[0])
        py = float(pixel_coord[1])

        field_points.append([fx, fy])
        pixel_points.append([px, py])

    return np.array(pixel_points, dtype=np.float32), np.array(field_points, dtype=np.float32)


def find_video_file(dataset_dir: Path) -> Optional[Path]:
    """Find a raw video file (not viz_results) in SoccerTrack directory."""
    # Prefer videos/ subdirectory over viz_results/
    videos_dir = dataset_dir / "videos"
    if videos_dir.exists():
        videos = sorted(videos_dir.glob("*.mp4"))
        if videos:
            return videos[0]

    # Fallback: any video not in viz_results
    for v in sorted(dataset_dir.rglob("*.mp4")):
        if "viz_results" not in str(v):
            return v

    # Last resort: viz_results
    for v in sorted(dataset_dir.rglob("*.mp4")):
        return v

    return None


def find_matching_annotation(video_path: Path, dataset_dir: Path) -> Optional[Path]:
    """Find the annotation CSV matching a video file."""
    video_stem = video_path.stem  # e.g. "F_20200220_1_0000_0030"
    ann_dir = dataset_dir / "annotations"

    if ann_dir.exists():
        ann_file = ann_dir / f"{video_stem}.csv"
        if ann_file.exists():
            return ann_file
        # Try any annotation
        anns = sorted(ann_dir.glob("*.csv"))
        if anns:
            return anns[0]

    return None


def find_keypoints_file(dataset_dir: Path) -> Optional[Path]:
    """Find fisheye_keypoints.json (may be one level up from wide_view/)."""
    # Check current dir
    kp = dataset_dir / "fisheye_keypoints.json"
    if kp.exists():
        return kp

    # Check parent (common if dataset_dir is wide_view/)
    kp = dataset_dir.parent / "fisheye_keypoints.json"
    if kp.exists():
        return kp

    # Search recursively
    for p in dataset_dir.parents:
        kp = p / "fisheye_keypoints.json"
        if kp.exists():
            return kp
        # Don't go above the data directory
        if p.name == "data" or p == dataset_dir.parent.parent:
            break

    return None


def get_video_info(video_path: Path) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info
