"""Accuracy computation utilities for pipeline testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def compute_detection_metrics(detections: list, ground_truth: list,
                               iou_threshold: float = 0.3) -> dict:
    """
    Compute detection precision, recall, and F1.

    Args:
        detections: List of detected bounding boxes [(x1,y1,x2,y2), ...]
        ground_truth: List of ground truth bounding boxes [(x1,y1,x2,y2), ...]
        iou_threshold: IoU threshold for a match

    Returns:
        Dict with precision, recall, f1, false_positives, false_negatives
    """
    if not ground_truth:
        return {
            "precision": 1.0 if not detections else 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(detections),
            "false_negatives": 0,
        }

    if not detections:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth),
        }

    # Compute IoU matrix
    iou_matrix = np.zeros((len(detections), len(ground_truth)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = _compute_iou(det, gt)

    # Hungarian matching to find best assignment
    cost_matrix = 1 - iou_matrix
    det_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Count matches above threshold
    true_positives = 0
    for di, gi in zip(det_indices, gt_indices):
        if iou_matrix[di, gi] >= iou_threshold:
            true_positives += 1

    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_detection_metrics_batch(all_detections: Dict[int, list],
                                     all_ground_truth: Dict[int, list],
                                     iou_threshold: float = 0.3) -> dict:
    """
    Compute detection metrics across multiple frames.

    Args:
        all_detections: {frame_id: [(x1,y1,x2,y2), ...]}
        all_ground_truth: {frame_id: [(x1,y1,x2,y2), ...]}

    Returns:
        Aggregated metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_frame_fp = []

    all_frames = set(list(all_detections.keys()) + list(all_ground_truth.keys()))

    for frame_id in all_frames:
        dets = all_detections.get(frame_id, [])
        gts = all_ground_truth.get(frame_id, [])
        metrics = compute_detection_metrics(dets, gts, iou_threshold)
        total_tp += metrics["true_positives"]
        total_fp += metrics["false_positives"]
        total_fn += metrics["false_negatives"]
        per_frame_fp.append(metrics["false_positives"])

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "mean_precision": round(precision, 4),
        "mean_recall": round(recall, 4),
        "mean_f1": round(f1, 4),
        "false_positives_per_frame": round(np.mean(per_frame_fp), 2) if per_frame_fp else 0,
        "total_frames": len(all_frames),
    }


def compute_homography_accuracy(detected_positions: np.ndarray,
                                 ground_truth_positions: np.ndarray) -> dict:
    """
    Compute position accuracy after homography transformation.

    Args:
        detected_positions: Nx2 array of (x, y) field coordinates from pipeline
        ground_truth_positions: Nx2 array of (x, y) field coordinates from ground truth

    Returns:
        Dict with mean/median/p95 error and percentage within thresholds
    """
    if len(detected_positions) == 0 or len(ground_truth_positions) == 0:
        return {
            "mean_error_m": float("inf"),
            "median_error_m": float("inf"),
            "p95_error_m": float("inf"),
            "pct_within_1m": 0.0,
            "pct_within_2m": 0.0,
            "n_matched": 0,
        }

    # Match detected to ground truth using Hungarian algorithm
    n_det = len(detected_positions)
    n_gt = len(ground_truth_positions)

    # Distance matrix
    dist_matrix = np.zeros((n_det, n_gt))
    for i in range(n_det):
        for j in range(n_gt):
            dist_matrix[i, j] = np.linalg.norm(
                detected_positions[i] - ground_truth_positions[j]
            )

    # Optimal matching
    det_indices, gt_indices = linear_sum_assignment(dist_matrix)

    # Compute errors for matched pairs (filter out matches > 10m as mismatches)
    errors = []
    for di, gi in zip(det_indices, gt_indices):
        error = dist_matrix[di, gi]
        if error < 10.0:  # Only count reasonable matches
            errors.append(error)

    if not errors:
        return {
            "mean_error_m": float("inf"),
            "median_error_m": float("inf"),
            "p95_error_m": float("inf"),
            "pct_within_1m": 0.0,
            "pct_within_2m": 0.0,
            "n_matched": 0,
        }

    errors = np.array(errors)

    return {
        "mean_error_m": round(float(errors.mean()), 3),
        "median_error_m": round(float(np.median(errors)), 3),
        "p95_error_m": round(float(np.percentile(errors, 95)), 3),
        "pct_within_1m": round(float((errors < 1.0).mean() * 100), 1),
        "pct_within_2m": round(float((errors < 2.0).mean() * 100), 1),
        "n_matched": len(errors),
    }


def compute_homography_accuracy_batch(detections_df: pd.DataFrame,
                                       ground_truth_df: pd.DataFrame,
                                       H: np.ndarray = None) -> dict:
    """
    Compute homography accuracy across multiple frames.

    Args:
        detections_df: DataFrame with frame, x_field, y_field columns
        ground_truth_df: DataFrame with frame, x_field, y_field columns
        H: Optional homography matrix (if coords not already transformed)

    Returns:
        Aggregated position accuracy metrics
    """
    all_errors = []
    frames = sorted(set(detections_df["frame"].unique()) &
                    set(ground_truth_df["frame"].unique()))

    for frame_id in frames:
        det_frame = detections_df[detections_df["frame"] == frame_id]
        gt_frame = ground_truth_df[ground_truth_df["frame"] == frame_id]

        det_pos = det_frame[["x_field", "y_field"]].values
        gt_pos = gt_frame[["x_field", "y_field"]].values

        if len(det_pos) == 0 or len(gt_pos) == 0:
            continue

        # Match and compute per-frame
        metrics = compute_homography_accuracy(det_pos, gt_pos)
        if metrics["n_matched"] > 0:
            # Recompute individual errors for this frame
            dist_matrix = np.zeros((len(det_pos), len(gt_pos)))
            for i in range(len(det_pos)):
                for j in range(len(gt_pos)):
                    dist_matrix[i, j] = np.linalg.norm(det_pos[i] - gt_pos[j])

            det_idx, gt_idx = linear_sum_assignment(dist_matrix)
            for di, gi in zip(det_idx, gt_idx):
                error = dist_matrix[di, gi]
                if error < 10.0:
                    all_errors.append(error)

    if not all_errors:
        return {
            "mean_error_m": float("inf"),
            "median_error_m": float("inf"),
            "p95_error_m": float("inf"),
            "pct_within_1m": 0.0,
            "pct_within_2m": 0.0,
            "n_frames": len(frames),
            "n_total_matches": 0,
        }

    errors = np.array(all_errors)
    return {
        "mean_error_m": round(float(errors.mean()), 3),
        "median_error_m": round(float(np.median(errors)), 3),
        "p95_error_m": round(float(np.percentile(errors, 95)), 3),
        "pct_within_1m": round(float((errors < 1.0).mean() * 100), 1),
        "pct_within_2m": round(float((errors < 2.0).mean() * 100), 1),
        "n_frames": len(frames),
        "n_total_matches": len(errors),
    }


def compute_mot_metrics(tracker_results: pd.DataFrame,
                         ground_truth: pd.DataFrame) -> dict:
    """
    Compute MOT (Multiple Object Tracking) metrics.

    Args:
        tracker_results: DataFrame with columns [frame, track_id, x, y, w, h]
        ground_truth: DataFrame with columns [frame, gt_id, x, y, w, h]

    Returns:
        Dict with MOTA, MOTP, IDF1, id_switches, fragmentation metrics
    """
    try:
        import motmetrics as mm
    except ImportError:
        print("WARNING: motmetrics not installed. Install with: pip install motmetrics")
        return _compute_mot_metrics_manual(tracker_results, ground_truth)

    # Build accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(tracker_results["frame"].unique()) |
                    set(ground_truth["frame"].unique()))

    for frame_id in frames:
        # Get objects in this frame
        gt_frame = ground_truth[ground_truth["frame"] == frame_id]
        det_frame = tracker_results[tracker_results["frame"] == frame_id]

        gt_ids = gt_frame["gt_id"].values.tolist()
        det_ids = det_frame["track_id"].values.tolist()

        if len(gt_ids) == 0 and len(det_ids) == 0:
            continue

        # Compute distance matrix (using center positions)
        gt_positions = gt_frame[["x", "y"]].values
        det_positions = det_frame[["x", "y"]].values

        if len(gt_positions) > 0 and len(det_positions) > 0:
            # Check if w/h available for IoU-based matching
            has_wh = ("w" in gt_frame.columns and "h" in gt_frame.columns and
                      "w" in det_frame.columns and "h" in det_frame.columns)
            
            if has_wh:
                # Use IoU-based distance (1 - IoU) — more robust for nearby players
                gt_boxes = gt_frame[["x", "y", "w", "h"]].values
                det_boxes = det_frame[["x", "y", "w", "h"]].values
                
                distances = np.ones((len(gt_boxes), len(det_boxes)))
                for i, gb in enumerate(gt_boxes):
                    gx1, gy1 = gb[0] - gb[2]/2, gb[1] - gb[3]/2
                    gx2, gy2 = gb[0] + gb[2]/2, gb[1] + gb[3]/2
                    for j, db in enumerate(det_boxes):
                        dx1, dy1 = db[0] - db[2]/2, db[1] - db[3]/2
                        dx2, dy2 = db[0] + db[2]/2, db[1] + db[3]/2
                        xi1, yi1 = max(gx1, dx1), max(gy1, dy1)
                        xi2, yi2 = min(gx2, dx2), min(gy2, dy2)
                        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
                        union = (gx2-gx1)*(gy2-gy1) + (dx2-dx1)*(dy2-dy1) - inter
                        iou = inter / union if union > 0 else 0
                        distances[i, j] = 1 - iou
                # Mark pairs with IoU < 0.2 as impossible matches
                distances[distances > 0.8] = np.nan
            else:
                # Fallback to center distance
                distances = mm.distances.norm2squared_matrix(gt_positions, det_positions)
                distances[distances > 200**2] = np.nan

            # Team-aware masking: prevent cross-team assignments
            # Only apply if team labels are available and reliably separated
            has_gt_team = "team_id" in gt_frame.columns
            has_det_team = "team" in det_frame.columns
            if has_gt_team and has_det_team:
                gt_teams = gt_frame["team_id"].values
                det_teams = det_frame["team"].values
                gt_team_set = set(int(t) for t in gt_teams)
                if len(gt_team_set) == 2:
                    gt_team_list = sorted(gt_team_set)
                    for i in range(len(gt_teams)):
                        for j in range(len(det_teams)):
                            dt = int(det_teams[j])
                            gt = int(gt_teams[i])
                            if dt >= 0 and dt in gt_team_set and dt != gt:
                                # Soften: don't fully block, just penalize heavily
                                # Set to a high cost (0.95) instead of nan
                                if distances[i, j] != distances[i, j]:  # already nan
                                    continue
                                distances[i, j] = max(distances[i, j], 0.95)
        else:
            distances = np.empty((len(gt_ids), len(det_ids)))

        acc.update(gt_ids, det_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "mota", "motp", "idf1", "num_switches",
        "num_fragmentations", "mostly_tracked", "mostly_lost"
    ], name="test")

    # Extract values
    mota = float(summary["mota"].iloc[0])
    motp = float(summary["motp"].iloc[0])
    idf1 = float(summary["idf1"].iloc[0])
    id_switches = int(summary["num_switches"].iloc[0])
    fragmentations = int(summary["num_fragmentations"].iloc[0])

    # Use actual video FPS (estimated from frame count)
    # SoccerTrack is 25fps; use frames/25 as more accurate
    fps_estimate = 25.0
    duration_s = len(frames) / fps_estimate
    duration_min = max(duration_s / 60.0, 0.001)

    n_gt_tracks = ground_truth["gt_id"].nunique()

    return {
        "MOTA": round(mota, 4),
        "MOTP": round(1.0 - motp / 10000.0, 4) if motp > 1 else round(motp, 4),
        "IDF1": round(idf1, 4),
        "id_switches_total": id_switches,
        "id_switches_per_minute": round(id_switches / duration_min, 1),
        "fragmentations": fragmentations,
        "track_fragments_per_player": round(fragmentations / max(n_gt_tracks, 1), 1),
        "duration_s": round(duration_s, 1),
    }


def _compute_mot_metrics_manual(tracker_results: pd.DataFrame,
                                 ground_truth: pd.DataFrame) -> dict:
    """Fallback MOT metrics computation without motmetrics library."""
    frames = sorted(set(tracker_results["frame"].unique()) &
                    set(ground_truth["frame"].unique()))

    if not frames:
        return {
            "MOTA": 0.0, "MOTP": 0.0, "IDF1": 0.0,
            "id_switches_total": 0, "id_switches_per_minute": 0.0,
            "fragmentations": 0, "track_fragments_per_player": 0.0,
            "duration_s": 0.0,
        }

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_switches = 0
    position_errors = []

    # Track assignment history: gt_id -> last matched track_id
    assignment_history = {}

    for frame_id in frames:
        gt_frame = ground_truth[ground_truth["frame"] == frame_id]
        det_frame = tracker_results[tracker_results["frame"] == frame_id]

        gt_positions = gt_frame[["x", "y"]].values
        det_positions = det_frame[["x", "y"]].values
        gt_ids = gt_frame["gt_id"].values
        det_ids = det_frame["track_id"].values

        total_gt += len(gt_ids)

        if len(gt_positions) == 0:
            total_fp += len(det_positions)
            continue
        if len(det_positions) == 0:
            total_fn += len(gt_positions)
            continue

        # Distance matrix
        dist_matrix = np.zeros((len(gt_positions), len(det_positions)))
        for i in range(len(gt_positions)):
            for j in range(len(det_positions)):
                dist_matrix[i, j] = np.linalg.norm(gt_positions[i] - det_positions[j])

        # Match with threshold
        gt_idx, det_idx = linear_sum_assignment(dist_matrix)
        matched_gt = set()
        matched_det = set()

        for gi, di in zip(gt_idx, det_idx):
            if dist_matrix[gi, di] < 200:  # pixel distance threshold (wide fisheye)
                matched_gt.add(gi)
                matched_det.add(di)
                total_tp += 1
                position_errors.append(dist_matrix[gi, di])

                # Check for ID switch
                gt_id = gt_ids[gi]
                det_id = det_ids[di]
                if gt_id in assignment_history:
                    if assignment_history[gt_id] != det_id:
                        total_switches += 1
                assignment_history[gt_id] = det_id

        total_fp += len(det_positions) - len(matched_det)
        total_fn += len(gt_positions) - len(matched_gt)

    # Compute MOTA
    mota = 1 - (total_fn + total_fp + total_switches) / max(total_gt, 1)

    # Compute MOTP (average position error for matched pairs)
    motp = float(np.mean(position_errors)) if position_errors else 0.0

    # Approximate IDF1 (simplified)
    idf1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    duration_s = len(frames) / 25.0  # SoccerTrack is 25fps
    duration_min = max(duration_s / 60.0, 0.001)
    n_gt_tracks = ground_truth["gt_id"].nunique()

    return {
        "MOTA": round(mota, 4),
        "MOTP": round(motp, 4),
        "IDF1": round(idf1, 4),
        "id_switches_total": total_switches,
        "id_switches_per_minute": round(total_switches / duration_min, 1),
        "fragmentations": 0,
        "track_fragments_per_player": 0.0,
        "duration_s": round(duration_s, 1),
    }


def compute_overall_verdict(detection_metrics: dict, homography_metrics: dict,
                             tracking_metrics: dict,
                             undistortion_verdict: str = "pass") -> dict:
    """
    Compute overall pipeline verdict based on all test results.

    Pass/Fail criteria from spec:
    - Detection recall > 85% = pass, 70-85% = marginal, < 70% = fail
    - Position error mean < 1.5m = pass, 1.5-2.5m = marginal, > 2.5m = fail
    - MOTA > 0.60 = pass, 0.45-0.60 = marginal, < 0.45 = fail
    - IDF1 > 0.55 = pass, 0.40-0.55 = marginal, < 0.40 = fail
    - ID switches/min < 20 = pass, 20-40 = marginal, > 40 = fail
    """
    verdicts = {}

    # Undistortion
    verdicts["undistortion"] = undistortion_verdict

    # Detection
    recall = detection_metrics.get("mean_recall", 0)
    if recall > 0.85:
        verdicts["detection"] = "pass"
    elif recall > 0.70:
        verdicts["detection"] = "marginal"
    else:
        verdicts["detection"] = "fail"

    # Homography
    mean_error = homography_metrics.get("mean_error_m", float("inf"))
    if mean_error < 1.5:
        verdicts["homography"] = "pass"
    elif mean_error < 2.5:
        verdicts["homography"] = "marginal"
    else:
        verdicts["homography"] = "fail"

    # Tracking - MOTA
    mota = tracking_metrics.get("MOTA", 0)
    if mota > 0.60:
        verdicts["tracking_mota"] = "pass"
    elif mota > 0.45:
        verdicts["tracking_mota"] = "marginal"
    else:
        verdicts["tracking_mota"] = "fail"

    # Tracking - IDF1
    idf1 = tracking_metrics.get("IDF1", 0)
    if idf1 > 0.55:
        verdicts["tracking_idf1"] = "pass"
    elif idf1 > 0.40:
        verdicts["tracking_idf1"] = "marginal"
    else:
        verdicts["tracking_idf1"] = "fail"

    # Tracking - ID switches
    switches_per_min = tracking_metrics.get("id_switches_per_minute", float("inf"))
    if switches_per_min < 20:
        verdicts["tracking_switches"] = "pass"
    elif switches_per_min < 40:
        verdicts["tracking_switches"] = "marginal"
    else:
        verdicts["tracking_switches"] = "fail"

    # Overall verdict
    all_verdicts = list(verdicts.values())
    if "fail" in all_verdicts:
        overall = "FAIL"
        failed_steps = [k for k, v in verdicts.items() if v == "fail"]
        recommendation = f"Fix before purchase: {', '.join(failed_steps)}"
    elif "marginal" in all_verdicts:
        overall = "MARGINAL"
        marginal_steps = [k for k, v in verdicts.items() if v == "marginal"]
        recommendation = f"Investigate before purchase: {', '.join(marginal_steps)}"
    else:
        overall = "PASS"
        recommendation = "Safe to purchase hardware"

    return {
        "per_component": verdicts,
        "overall_verdict": overall,
        "recommendation": recommendation,
    }


def _compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
