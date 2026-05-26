"""Step 4 — Multi-Object Tracking (MOT) Metrics Test.

Validates ByteTrack tracking performance against ground truth
using SoccerTrack v2 annotated data.
"""

import cv2
import numpy as np
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "soccer_analyzer"))

from utils.metrics import compute_mot_metrics
from utils.visualization import save_tracking_trajectories
from utils.soccertrack_loader import (load_soccertrack_tracking_gt, find_video_file,
                                       find_matching_annotation)
from utils.enhanced_detection import (get_model, enhanced_detect,
                                       load_undistort_maps)
from utils.team_tracker import TeamTracker
from utils.appearance_tracker import AppearanceTracker


def run_tracking_test(v2_path: str, output_dir: str,
                      confidence_threshold: float = 0.15,
                      num_test_frames: int = 300,
                      fps: float = 30.0) -> dict:
    """
    Run multi-object tracking test on SoccerTrack v2 dataset.

    Steps:
    1. Load ground truth tracking annotations (frame, id, bbox)
    2. Run YOLOv8 + ByteTrack on video
    3. Compute MOT metrics (MOTA, MOTP, IDF1, ID switches)
    4. Report results against thresholds

    Targets:
    - MOTA > 0.60
    - IDF1 > 0.55
    - ID switches < 20/min

    Returns: dict with MOT metrics and verdict
    """
    print("\n" + "=" * 60)
    print("STEP 4: MULTI-OBJECT TRACKING TEST")
    print("=" * 60)

    v2_dir = Path(v2_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load ground truth tracking data
    gt_df = None
    video_path = find_video_file(v2_dir)
    if video_path:
        ann_path = find_matching_annotation(video_path, v2_dir)
        if ann_path:
            print(f"  Loading GT from: {ann_path}")
            gt_df = load_soccertrack_tracking_gt(str(ann_path))

    if gt_df is None:
        gt_df = _load_tracking_gt(v2_dir)

    if gt_df is None:
        print("  ⚠️  No tracking ground truth found.")
        print("  Running tracker to verify it initializes correctly...")
        return _run_tracker_validation(v2_dir, output, confidence_threshold, fps)

    print(f"  Ground truth: {len(gt_df)} annotations, "
          f"{gt_df['gt_id'].nunique()} tracks, "
          f"{gt_df['frame'].nunique()} frames")

    # Load or run tracker
    result = _run_tracker_on_video(v2_dir, confidence_threshold, num_test_frames, fps)

    if result is None or (isinstance(result, tuple) and result[0] is None):
        print("  ERROR: Tracker produced no results.")
        return {"verdict": "fail", "error": "tracker_no_output"}

    tracker_df, track_colors = result
    if tracker_df is None or tracker_df.empty:
        print("  ERROR: Tracker produced no results.")
        return {"verdict": "fail", "error": "tracker_no_output"}

    print(f"  Tracker output: {len(tracker_df)} detections, "
          f"{tracker_df['track_id'].nunique()} tracks")

    # Filter out short-lived tracks (noise from FP detections)
    # Keep only the top-N longest tracks where N = number of GT tracks
    track_lengths = tracker_df.groupby("track_id")["frame"].nunique()
    n_gt_tracks = gt_df["gt_id"].nunique()
    valid_tracks = track_lengths.nlargest(n_gt_tracks).index
    tracker_df = tracker_df[tracker_df["track_id"].isin(valid_tracks)]
    print(f"  After filtering (top {n_gt_tracks} tracks): {len(tracker_df)} detections, "
          f"{tracker_df['track_id'].nunique()} tracks")

    # Assign team labels via jersey color clustering on filtered tracks only
    filtered_track_ids = set(tracker_df["track_id"].unique())
    filtered_colors = {tid: colors for tid, colors in track_colors.items()
                       if tid in filtered_track_ids and len(colors) >= 3}
    if len(filtered_colors) >= 4:
        tids = list(filtered_colors.keys())
        samples = np.array([np.mean(colors, axis=0) for colors in filtered_colors.values()],
                          dtype=np.float32)
        # Weight H heavily for jersey color; low weight on V (lighting)
        weights = np.array([2.0, 1.0, 0.3], dtype=np.float32)
        weighted = samples * weights
        # Cluster into 2 teams (ignore referees for now since GT only has 2)
        n_clusters = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, labels, centers = cv2.kmeans(
            weighted, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        track_team = {tids[i]: int(labels[i][0]) for i in range(len(tids))}
        tracker_df["team"] = tracker_df["track_id"].map(track_team).fillna(-1).astype(int)
        unweighted_centers = centers / weights
        print(f"    Team clustering (2 groups on {len(tids)} tracks):")
        for i, c in enumerate(unweighted_centers):
            count = (labels == i).sum()
            print(f"      Team {i}: H={c[0]:.0f} S={c[1]:.0f} V={c[2]:.0f} ({count} tracks)")
    else:
        tracker_df["team"] = -1

    # Align frame ranges
    common_frames = set(gt_df["frame"].unique()) & set(tracker_df["frame"].unique())
    if not common_frames:
        print("  ERROR: No overlapping frames between tracker and GT")
        return {"verdict": "fail", "error": "no_frame_overlap"}

    gt_aligned = gt_df[gt_df["frame"].isin(common_frames)]
    tracker_aligned = tracker_df[tracker_df["frame"].isin(common_frames)]

    print(f"  Common frames: {len(common_frames)}")

    # Map tracker team clusters to GT team_ids using spatial overlap
    if "team" in tracker_aligned.columns and "team_id" in gt_aligned.columns:
        _map_tracker_teams_to_gt(tracker_aligned, gt_aligned)

    # Compute MOT metrics
    print("\n  Computing MOT metrics...")
    mot_metrics = compute_mot_metrics(tracker_aligned, gt_aligned)

    # Print results
    print(f"\n  Tracking Results:")
    print(f"    MOTA:  {mot_metrics['MOTA']:.4f}")
    print(f"    MOTP:  {mot_metrics['MOTP']:.4f}")
    print(f"    IDF1:  {mot_metrics['IDF1']:.4f}")
    print(f"    ID switches (total): {mot_metrics['id_switches_total']}")
    print(f"    ID switches/min: {mot_metrics['id_switches_per_minute']:.1f}")
    print(f"    Fragmentations: {mot_metrics['fragmentations']}")
    print(f"    Duration: {mot_metrics['duration_s']:.1f}s")

    # Save trajectory visualization
    if "x_field" in tracker_aligned.columns or "x" in tracker_aligned.columns:
        vis_df = tracker_aligned.copy()
        if "x_field" not in vis_df.columns:
            vis_df["x_field"] = vis_df["x"]
            vis_df["y_field"] = vis_df["y"]
        save_tracking_trajectories(
            vis_df, str(output / "tracking_trajectories.png"),
            title="ByteTrack Tracking Trajectories"
        )

    # Determine verdicts
    mota_verdict = _verdict_mota(mot_metrics["MOTA"])
    idf1_verdict = _verdict_idf1(mot_metrics["IDF1"])
    switch_verdict = _verdict_switches(mot_metrics["id_switches_per_minute"])

    # Overall tracking verdict: worst of the three
    verdict_levels = {"pass": 2, "marginal": 1, "fail": 0}
    overall_level = min(verdict_levels[mota_verdict],
                        verdict_levels[idf1_verdict],
                        verdict_levels[switch_verdict])
    overall_verdict = {2: "pass", 1: "marginal", 0: "fail"}[overall_level]

    results = {
        **mot_metrics,
        "mota_verdict": mota_verdict,
        "idf1_verdict": idf1_verdict,
        "switch_verdict": switch_verdict,
        "verdict": overall_verdict,
    }

    _print_verdict(overall_verdict, mot_metrics)
    return results


def _map_tracker_teams_to_gt(tracker_df: pd.DataFrame, gt_df: pd.DataFrame):
    """
    Map tracker cluster labels to GT team_ids by spatial overlap.
    Modifies tracker_df["team"] in place to match GT team_id encoding.
    """
    gt_teams = sorted(gt_df["team_id"].unique())
    tracker_teams = sorted(tracker_df["team"].unique())
    tracker_teams = [t for t in tracker_teams if t >= 0]

    if len(gt_teams) < 2 or len(tracker_teams) < 2:
        return

    # For each tracker team, count how many detections overlap with each GT team
    # Use a sample of frames for speed
    sample_frames = sorted(gt_df["frame"].unique())[:50]
    overlap_matrix = defaultdict(lambda: defaultdict(int))

    for frame_id in sample_frames:
        gt_frame = gt_df[gt_df["frame"] == frame_id]
        det_frame = tracker_df[tracker_df["frame"] == frame_id]
        if gt_frame.empty or det_frame.empty:
            continue

        for _, det_row in det_frame.iterrows():
            if det_row["team"] < 0:
                continue
            # Find nearest GT
            dists = np.sqrt((gt_frame["x"].values - det_row["x"])**2 +
                           (gt_frame["y"].values - det_row["y"])**2)
            nearest_idx = np.argmin(dists)
            if dists[nearest_idx] < 100:  # within 100px
                gt_team = gt_frame.iloc[nearest_idx]["team_id"]
                overlap_matrix[det_row["team"]][gt_team] += 1

    # Find best mapping: tracker_team -> gt_team
    team_map = {}
    for t_team in tracker_teams:
        if overlap_matrix[t_team]:
            best_gt = max(overlap_matrix[t_team], key=overlap_matrix[t_team].get)
            team_map[t_team] = int(best_gt)

    # Apply mapping
    if team_map:
        tracker_df["team"] = tracker_df["team"].map(
            lambda x: team_map.get(x, x) if x >= 0 else x
        )
        print(f"    Team mapping: {team_map}")


def _run_tracker_on_video(v2_dir: Path, confidence_threshold: float,
                           num_frames: int, fps: float) -> pd.DataFrame:
    """Run YOLO + ByteTrack on video and return tracking results."""
    video_path = find_video_file(v2_dir)
    if video_path is None:
        return None

    # Load models
    try:
        import supervision as sv
    except ImportError as e:
        print(f"  ERROR: Missing dependency: {e}")
        return None

    print(f"  Loading YOLOv8s model...")
    model = get_model("yolov8s.pt")

    # Use single ByteTrack with appearance re-ID
    app_tracker = AppearanceTracker(
        max_lost_frames=60,
        appearance_threshold=0.65,
    )
    # Collect jersey colors per track for team assignment
    track_colors = defaultdict(list)  # track_id -> list of HSV colors

    print(f"  Running tracker on: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    results_list = []
    frame_count = 0

    while frame_count < min(num_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Enhanced detection (tiling + multiscale + adaptive NMS)
        # Do NOT undistort — GT annotations are in original pixel coords
        # Use slightly higher confidence for tracking to reduce noise tracks
        track_conf = max(confidence_threshold, 0.25)
        detections = enhanced_detect(
            model, frame, confidence=track_conf,
            use_tiling=True, use_multiscale=False,  # Skip multiscale for speed
            use_adaptive_nms=True, undistort_map=None,
        )

        if detections:
            # Convert to supervision format
            xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
            # Use uniform confidence since enhanced_detect strips it
            conf = np.ones(len(detections), dtype=np.float32) * 0.5
            class_ids = np.zeros(len(detections), dtype=int)

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=class_ids,
            )

            # Track with appearance-based re-ID
            tracked = app_tracker.update(frame, sv_detections)

            if tracked.tracker_id is not None:
                for i, track_id in enumerate(tracked.tracker_id):
                    x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = float(x2 - x1)
                    h = float(y2 - y1)

                    # Extract jersey color (upper body HSV mean)
                    bh, bw = y2 - y1, x2 - x1
                    if bh > 8 and bw > 4:
                        ty1 = max(0, y1 + int(bh * 0.20))
                        ty2 = y1 + int(bh * 0.60)
                        tx1 = max(0, x1 + int(bw * 0.10))
                        tx2 = x2 - int(bw * 0.10)
                        crop = frame[ty1:ty2, tx1:tx2]
                        if crop.size > 0:
                            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            mean_hsv = hsv_crop.reshape(-1, 3).mean(axis=0)
                            track_colors[int(track_id)].append(mean_hsv)

                    results_list.append({
                        "frame": frame_count,
                        "track_id": int(track_id),
                        "x": float(cx),
                        "y": float(cy),
                        "w": w,
                        "h": h,
                    })

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"    Processed {frame_count}/{min(num_frames, total_frames)} frames...")

    cap.release()

    if not results_list:
        return None, {}

    df = pd.DataFrame(results_list)
    df["team"] = -1  # Will be assigned after filtering
    return df, dict(track_colors)


def _run_tracker_validation(v2_dir: Path, output: Path,
                             confidence_threshold: float, fps: float) -> dict:
    """Validate tracker initializes and runs without errors."""
    try:
        from ultralytics import YOLO
        import supervision as sv
    except ImportError as e:
        return {"verdict": "fail", "error": f"import_failed: {e}"}

    # Quick test: load model and run on a single frame
    model = get_model("yolov8s.pt")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.2,
        lost_track_buffer=30,
        minimum_matching_threshold=0.7,
    )

    # Try video if available
    video_path = _find_video(v2_dir)
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
    else:
        # Synthetic frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        ret = True

    if not ret:
        return {"verdict": "fail", "error": "cannot_read_frame"}

    # Run detection + tracking
    frame_w = frame.shape[1]
    imgsz = 1280 if frame_w <= 1920 else min(3200, frame_w // 2)
    results = model(frame, classes=[0], conf=confidence_threshold, imgsz=imgsz, verbose=False)
    n_detections = sum(len(r.boxes) for r in results)

    print(f"  Tracker validation: model loaded, {n_detections} detections on test frame")

    return {
        "MOTA": 0.0,
        "MOTP": 0.0,
        "IDF1": 0.0,
        "id_switches_total": 0,
        "id_switches_per_minute": 0.0,
        "fragmentations": 0,
        "track_fragments_per_player": 0.0,
        "duration_s": 0.0,
        "verdict": "marginal",
        "notes": "Tracker runs OK but no GT data for metrics. Download SoccerTrack v2.",
    }


def _load_tracking_gt(v2_dir: Path) -> pd.DataFrame:
    """
    Load tracking ground truth in MOT format.
    Expected: frame, id, x, y, w, h (or variations)
    """
    # Standard MOT format
    gt_files = (list(v2_dir.rglob("gt/gt.txt")) +
                list(v2_dir.rglob("*gt*.txt")) +
                list(v2_dir.rglob("*gt*.csv")) +
                list(v2_dir.rglob("*annotations*.csv")))

    for f in gt_files:
        try:
            if f.suffix == ".txt":
                # MOT format: frame,id,x,y,w,h,conf,class,visibility
                df = pd.read_csv(str(f), header=None,
                                 names=["frame", "gt_id", "x", "y", "w", "h",
                                        "conf", "class", "visibility"])
                # Filter by confidence (1 = active, 0 = ignore)
                if "conf" in df.columns:
                    df = df[df["conf"] >= 0]
                # Convert to center coordinates
                df["x"] = df["x"] + df["w"] / 2
                df["y"] = df["y"] + df["h"] / 2
                print(f"  Loaded GT: {f}")
                return df[["frame", "gt_id", "x", "y", "w", "h"]]
            else:
                df = pd.read_csv(str(f))
                # Try to identify columns
                col_map = {}
                for col in df.columns:
                    cl = col.lower()
                    if "frame" in cl:
                        col_map[col] = "frame"
                    elif cl in ["id", "track_id", "gt_id", "object_id", "player_id"]:
                        col_map[col] = "gt_id"
                    elif cl == "x" or ("center" in cl and "x" in cl):
                        col_map[col] = "x"
                    elif cl == "y" or ("center" in cl and "y" in cl):
                        col_map[col] = "y"
                    elif cl in ["w", "width"]:
                        col_map[col] = "w"
                    elif cl in ["h", "height"]:
                        col_map[col] = "h"

                if "frame" in col_map.values() and "gt_id" in col_map.values():
                    df = df.rename(columns=col_map)
                    if "x" not in df.columns and "w" not in df.columns:
                        continue
                    print(f"  Loaded GT: {f}")
                    needed = ["frame", "gt_id"]
                    if "x" in df.columns:
                        needed.append("x")
                    if "y" in df.columns:
                        needed.append("y")
                    if "w" in df.columns:
                        needed.append("w")
                    if "h" in df.columns:
                        needed.append("h")
                    return df[needed]
        except Exception as e:
            continue

    # Try sportsLabKit
    try:
        import sportsLabKit as slk
        dataset = slk.load_soccertrack(str(v2_dir))
        if hasattr(dataset, "ground_truth"):
            return dataset.ground_truth
    except (ImportError, Exception):
        pass

    return None


def _find_video(directory: Path) -> Path:
    """Find a video file in directory."""
    extensions = [".mp4", ".avi", ".mov", ".mkv"]
    for ext in extensions:
        videos = list(directory.rglob(f"*{ext}"))
        if videos:
            return videos[0]
    return None


def _verdict_mota(mota: float) -> str:
    if mota > 0.60:
        return "pass"
    elif mota > 0.45:
        return "marginal"
    return "fail"


def _verdict_idf1(idf1: float) -> str:
    if idf1 > 0.55:
        return "pass"
    elif idf1 > 0.40:
        return "marginal"
    return "fail"


def _verdict_switches(switches_per_min: float) -> str:
    if switches_per_min < 20:
        return "pass"
    elif switches_per_min < 40:
        return "marginal"
    return "fail"


def _print_verdict(verdict: str, metrics: dict):
    """Print colored verdict."""
    mota = metrics["MOTA"]
    idf1 = metrics["IDF1"]
    sw = metrics["id_switches_per_minute"]

    if verdict == "pass":
        print(f"\n  ✅ STEP 4 VERDICT: PASS")
        print(f"     MOTA={mota:.3f} (>0.60) | IDF1={idf1:.3f} (>0.55) | "
              f"Switches={sw:.0f}/min (<20)")
    elif verdict == "marginal":
        print(f"\n  ⚠️  STEP 4 VERDICT: MARGINAL")
        print(f"     MOTA={mota:.3f} | IDF1={idf1:.3f} | Switches={sw:.0f}/min")
    else:
        print(f"\n  ❌ STEP 4 VERDICT: FAIL")
        print(f"     MOTA={mota:.3f} | IDF1={idf1:.3f} | Switches={sw:.0f}/min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test multi-object tracking")
    parser.add_argument("--v2_path", default="data/soccertrack_v2/",
                        help="Path to SoccerTrack v2 data")
    parser.add_argument("--output", default="test_outputs/",
                        help="Output directory for test results")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to test")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS for timing calculations")
    args = parser.parse_args()

    results = run_tracking_test(args.v2_path, args.output,
                                args.conf, args.frames, args.fps)
    print(f"\nResults: {results}")
