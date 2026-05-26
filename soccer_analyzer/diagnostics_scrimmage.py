#!/usr/bin/env python3
"""
Scrimmage Diagnostic Tool — validates the fisheye + flag + game pipeline for intra-team scrimmages.

Usage:
    python diagnostics_scrimmage.py "/path/to/scrimmage.mov" \
        --calibration data/calibration/neewer_fisheye.npz \
        --flag-color orange --field-size 40x30 --my-team red

Outputs to diagnostics_scrimmage/ folder:
    00_undistort/        — before/after fisheye correction samples
    01_flags/            — detected flag positions + homography overlay
    02_detections/       — annotated frames with bboxes + track IDs
    03_team_assignment/  — colored frames with team overlay
    04_field_positions/  — player positions on pitch (field coords)
    05_movement_stats/   — per-track speed, distance charts
    06_formation/        — detected formation snapshot
    07_player_id/        — identification results
    08_passes/           — pass map
    summary.json         — pipeline metrics
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd

from config import (DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SAMPLE_RATE, YOLO_IMGSZ,
                    DEFAULT_MODEL_SIZE, SPRINT_THRESHOLD, MAX_SPEED_CAP,
                    DEFAULT_PLAYERS_PER_TEAM, DEFAULT_CALIBRATION_FILE)


def parse_args():
    p = argparse.ArgumentParser(description="Scrimmage pipeline diagnostic tool")
    p.add_argument("video", help="Path to scrimmage video file")
    p.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_FILE),
                   help=f"Path to fisheye calibration .npz file (default: {DEFAULT_CALIBRATION_FILE})")
    p.add_argument("--flag-color", type=str, default="red",
                   choices=["red", "orange", "pink", "yellow", "green_neon", "blue_neon"],
                   help="Color of field marker flags (default: red)")
    p.add_argument("--field-size", type=str, default="40x30",
                   help="Field dimensions as LxW in meters (default: 40x30)")
    p.add_argument("--start", type=int, default=0,
                   help="Start frame (default: 0)")
    p.add_argument("--frames", type=int, default=60,
                   help="Number of sampled frames to process (default: 60)")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                   help=f"Process every Nth frame (default: {DEFAULT_SAMPLE_RATE})")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"YOLO confidence threshold (default: {DEFAULT_CONFIDENCE})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_SIZE,
                   help=f"YOLO model size: n/s/m/l/x (default: {DEFAULT_MODEL_SIZE})")
    p.add_argument("--tracker", type=str, default="botsort",
                   choices=["bytetrack", "botsort", "botsort_noid"],
                   help="Tracker algorithm (default: botsort)")
    p.add_argument("--yolo-version", type=str, default="8",
                   choices=["8", "11"],
                   help="YOLO version (default: 8)")
    p.add_argument("--my-team", type=str, default=None,
                   choices=["black", "white", "red", "blue", "green", "yellow", "orange", "purple"],
                   help="My team's jersey/bib color")
    p.add_argument("--opponent", type=str, default=None,
                   choices=["black", "white", "red", "blue", "green", "yellow", "orange", "purple"],
                   help="Opponent's jersey/bib color")
    p.add_argument("--outdir", type=str, default="diagnostics_output/diagnostics_scrimmage",
                   help="Output directory (default: diagnostics_output/diagnostics_scrimmage/)")
    return p.parse_args()


def setup_dirs(outdir):
    subdirs = ["00_undistort", "01_flags", "02_detections", "03_team_assignment",
               "04_field_positions", "05_movement_stats", "06_formation",
               "07_player_id", "08_passes"]
    dirs = {}
    for name in subdirs:
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        dirs[name] = d
    os.makedirs(outdir, exist_ok=True)
    return dirs


def main():
    args = parse_args()
    outdir = args.outdir
    dirs = setup_dirs(outdir)

    # Parse field size
    try:
        field_length, field_width = map(float, args.field_size.split("x"))
    except ValueError:
        print(f"Error: Invalid field-size '{args.field_size}'. Use LxW (e.g., 40x30)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SCRIMMAGE PIPELINE DIAGNOSTIC")
    print(f"{'='*60}")

    summary = {"session_type": "scrimmage", "args": vars(args)}
    t_start = time.time()

    # =====================================================================
    # Step 0: Fisheye Undistortion
    # =====================================================================
    print(f"\n[0/8] Loading fisheye calibration...")
    from pipeline.fisheye import FisheyeCalibration

    calib = FisheyeCalibration(Path(args.calibration))
    print(f"  Calibration loaded: {args.calibration}")

    # Open video and undistort sample frames
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames "
          f"({total_frames/fps/60:.1f} min)")

    # Save before/after undistortion samples
    sample_indices = [args.start, args.start + total_frames // 4,
                      args.start + total_frames // 2]
    for idx in sample_indices[:3]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            undistorted = calib.undistort(frame)
            # Save comparison
            h, w = frame.shape[:2]
            scale = 1280 / w if w > 1280 else 1.0
            if scale < 1.0:
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                undist_small = cv2.resize(undistorted, (int(w * scale), int(h * scale)))
            else:
                frame_small = frame
                undist_small = undistorted
            comparison = np.hstack([frame_small, undist_small])
            cv2.imwrite(os.path.join(dirs["00_undistort"], f"compare_{idx:06d}.jpg"),
                        comparison, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"  Saved {len(sample_indices)} undistortion comparisons to {dirs['00_undistort']}/")

    # =====================================================================
    # Step 0.5: Flag Detection + Homography
    # =====================================================================
    print(f"\n[0.5/8] Detecting field flags...")
    from pipeline.flag_detector import FlagDetector
    from pipeline.homography import FlagHomography

    flag_detector = FlagDetector(flag_color=args.flag_color)

    # Use first few frames to detect stable flag positions
    flag_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            flag_frames.append(calib.undistort(frame))

    if not flag_frames:
        print("  Error: Could not read frames for flag detection")
        sys.exit(1)

    centroids = flag_detector.detect_stable(flag_frames)
    print(f"  Detected {len(centroids)} flags: {centroids}")

    if len(centroids) < 4:
        print(f"  ⚠️ Need 4 flags for homography, got {len(centroids)}. "
              f"Falling back to density-based calibration.")
        homography = FlagHomography(field_length, field_width)
        use_flags = False
    else:
        corners = flag_detector.assign_corners(centroids, flag_frames[0].shape)
        if corners is None:
            print(f"  ⚠️ Could not assign flag corners. Falling back.")
            homography = FlagHomography(field_length, field_width)
            use_flags = False
        else:
            homography = FlagHomography(field_length, field_width)
            homography.calibrate_from_flags(corners)
            use_flags = True
            print(f"  ✓ Flag-based homography calibrated: {field_length}x{field_width}m")
            print(f"    Corners: {corners}")

    # Save flag visualization
    vis_frame = flag_detector.visualize(
        flag_frames[0], centroids,
        corners if use_flags else None
    )
    h, w = vis_frame.shape[:2]
    if w > 1920:
        scale = 1920 / w
        vis_frame = cv2.resize(vis_frame, (int(w * scale), int(h * scale)))
    cv2.imwrite(os.path.join(dirs["01_flags"], "flag_detection.jpg"),
                vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved flag visualization to {dirs['01_flags']}/")

    summary["flags_detected"] = len(centroids)
    summary["flag_homography"] = use_flags

    # =====================================================================
    # Step 1: Detection + Tracking (on undistorted frames)
    # =====================================================================
    print(f"\n[1/8] Running YOLO detection + {args.tracker}...")
    from ultralytics import YOLO
    import supervision as sv

    # Load model
    if args.yolo_version == "11":
        model = YOLO(f"yolo11{args.model}.pt")
    else:
        model = YOLO(f"yolov8{args.model}.pt")

    confidence = args.confidence
    if confidence == DEFAULT_CONFIDENCE:
        confidence = 0.25 if frame_w >= 3840 else 0.30 if frame_w >= 1920 else 0.35

    effective_fps = fps / args.sample_rate
    track_buffer = int(120 / args.sample_rate)

    # Init tracker
    if args.tracker in ("botsort", "botsort_noid"):
        from boxmot import BotSort
        tracker = BotSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt") if args.tracker == "botsort" else None,
            device="cpu",
            half=False,
            track_high_thresh=confidence,
            track_low_thresh=0.1,
            new_track_thresh=confidence,
            track_buffer=track_buffer,
            match_thresh=0.8 if args.tracker == "botsort" else 0.7,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method="sof",
            frame_rate=int(effective_fps),
            with_reid=(args.tracker == "botsort"),
        )
    else:
        tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.25,
            frame_rate=effective_fps,
        )

    model_name = f"yolo11{args.model}" if args.yolo_version == "11" else f"yolov8{args.model}"
    print(f"  Model: {model_name}, conf={confidence}, tracker={args.tracker}")

    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    all_detections = []
    frames = {}
    processed = 0
    frame_id = args.start
    MIN_BBOX_AREA = 400
    t0 = time.time()

    while processed < args.frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % args.sample_rate == 0:
            # Undistort before processing
            frame = calib.undistort(frame)
            frames[frame_id] = frame.copy()

            results = model(frame, conf=confidence, classes=[0],
                            device=DEVICE, imgsz=YOLO_IMGSZ, verbose=False)

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                valid = areas >= MIN_BBOX_AREA
                xyxy = xyxy[valid]
                confs = confs[valid]

                if len(xyxy) > 0:
                    if args.tracker in ("botsort", "botsort_noid"):
                        dets_arr = np.column_stack([xyxy, confs, np.zeros(len(xyxy))])
                        tracks = tracker.update(dets_arr, frame)
                        if tracks is not None and len(tracks) > 0:
                            for t in tracks:
                                bbox = t[:4]
                                tid = int(t[4])
                                all_detections.append({
                                    "frame": frame_id, "track_id": tid,
                                    "x_px": float((bbox[0] + bbox[2]) / 2),
                                    "y_px": float((bbox[1] + bbox[3]) / 2),
                                    "bbox_x1": float(bbox[0]), "bbox_y1": float(bbox[1]),
                                    "bbox_x2": float(bbox[2]), "bbox_y2": float(bbox[3]),
                                    "conf": float(t[5]) if len(t) > 5 else 0.0,
                                    "frame_w": frame_w, "frame_h": frame_h,
                                })
                    else:
                        detections = sv.Detections(
                            xyxy=xyxy, confidence=confs,
                            class_id=np.zeros(len(xyxy), dtype=int)
                        )
                        detections = tracker.update_with_detections(detections)
                        for i in range(len(detections)):
                            bbox = detections.xyxy[i]
                            tid = detections.tracker_id[i] if detections.tracker_id is not None else i
                            all_detections.append({
                                "frame": frame_id, "track_id": int(tid),
                                "x_px": float((bbox[0] + bbox[2]) / 2),
                                "y_px": float((bbox[1] + bbox[3]) / 2),
                                "bbox_x1": float(bbox[0]), "bbox_y1": float(bbox[1]),
                                "bbox_x2": float(bbox[2]), "bbox_y2": float(bbox[3]),
                                "conf": float(detections.confidence[i]),
                                "frame_w": frame_w, "frame_h": frame_h,
                            })

            processed += 1
            print(f"\r  Frame {frame_id} [{processed}/{args.frames}] "
                  f"({time.time()-t0:.1f}s)", end="", flush=True)

        frame_id += 1

    cap.release()
    print()

    df = pd.DataFrame(all_detections)
    t_detect = time.time() - t0
    n_tracks = df["track_id"].nunique() if not df.empty else 0
    print(f"  Detection complete: {len(df)} detections, {n_tracks} tracks in {t_detect:.1f}s")
    summary["detections"] = len(df)
    summary["tracks"] = n_tracks

    if df.empty:
        print("  ⚠️ No detections found. Check video path and confidence.")
        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Save detection frames
    from matplotlib import pyplot as plt
    sample_frames = sorted(frames.keys())[:10]
    for fid in sample_frames:
        frame = frames[fid].copy()
        frame_dets = df[df["frame"] == fid]
        for _, det in frame_dets.iterrows():
            x1, y1, x2, y2 = int(det.bbox_x1), int(det.bbox_y1), int(det.bbox_x2), int(det.bbox_y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"T{int(det.track_id)}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imwrite(os.path.join(dirs["02_detections"], f"det_{fid:06d}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  Saved {len(sample_frames)} detection frames to {dirs['02_detections']}/")

    # =====================================================================
    # Step 2: Homography + Field Filter
    # =====================================================================
    print(f"\n[2/8] Applying homography + field filter...")
    if not use_flags:
        # Fallback: calibrate from detection density
        video_meta = {"frame_h": frame_h, "frame_w": frame_w, "fps": fps}
        homography.calibrate_auto(frame_h, frame_w, df)

    pre_count = len(df)
    df = homography.filter_to_field_zone(df)
    post_count = len(df)
    if pre_count > post_count:
        print(f"  Field zone filter: {pre_count} → {post_count} detections "
              f"(removed {pre_count - post_count} off-field)")

    df = homography.transform_df(df)
    val_score, val_msg = homography.validate(df)
    print(f"  Homography validation: score={val_score:.2f} — {val_msg}")
    summary["homography_score"] = val_score

    # =====================================================================
    # Step 3: Team Classification
    # =====================================================================
    print(f"\n[3/8] Running team classification...")
    from pipeline.team_classifier import TeamClassifier

    classifier = TeamClassifier()
    df = classifier.classify_teams(
        df, frames,
        my_team_color=args.my_team,
        opponent_color=args.opponent,
    )
    team_map = df.groupby("track_id")["team"].first().to_dict()

    home_count = sum(1 for v in team_map.values() if v == "Home")
    away_count = sum(1 for v in team_map.values() if v == "Away")
    print(f"  Teams: Home={home_count} tracks, Away={away_count} tracks")
    summary["home_tracks"] = home_count
    summary["away_tracks"] = away_count

    # Save team-colored frames
    sample_frames_team = sorted(frames.keys())[:5]
    for fid in sample_frames_team:
        if fid not in frames:
            continue
        frame = frames[fid].copy()
        frame_dets = df[df["frame"] == fid]
        for _, det in frame_dets.iterrows():
            x1, y1, x2, y2 = int(det.bbox_x1), int(det.bbox_y1), int(det.bbox_x2), int(det.bbox_y2)
            team = det.get("team", "")
            color = (255, 150, 0) if team == "Home" else (0, 100, 255) if team == "Away" else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"T{int(det.track_id)} {team}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imwrite(os.path.join(dirs["03_team_assignment"], f"team_{fid:06d}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  Saved team assignment frames to {dirs['03_team_assignment']}/")

    # =====================================================================
    # Step 4: Field Positions + Movement Stats
    # =====================================================================
    print(f"\n[4/8] Computing movement stats...")
    from pipeline.stats import StatsCalculator

    stats_calc = StatsCalculator(field_length=field_length, field_width=field_width, fps=fps)
    # Compute per-track movement
    if "x_field" in df.columns and not df.empty:
        # Field position scatter
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for team_label, color in [("Home", "blue"), ("Away", "red")]:
            team_df = df[df["team"] == team_label]
            ax.scatter(team_df["x_field"], team_df["y_field"],
                       c=color, alpha=0.3, s=5, label=team_label)
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        ax.set_title("Field Positions")
        ax.legend()
        ax.set_aspect("equal")
        fig.savefig(os.path.join(dirs["04_field_positions"], "field_scatter.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved field position scatter to {dirs['04_field_positions']}/")

    # =====================================================================
    # Step 5: Formation
    # =====================================================================
    print(f"\n[5/8] Detecting formation...")
    from pipeline.formation import FormationDetector

    formation_detector = FormationDetector(players_per_team=DEFAULT_PLAYERS_PER_TEAM)

    # Get average field positions per track for each team
    home_formation = "Unknown"
    away_formation = "Unknown"
    if "x_field" in df.columns:
        for team_label, var_name in [("Home", "home_formation"), ("Away", "away_formation")]:
            team_df = df[df["team"] == team_label]
            if not team_df.empty:
                avg_pos = team_df.groupby("track_id")[["x_field", "y_field"]].mean().values
                if len(avg_pos) >= 4:
                    formation, conf = formation_detector.detect_formation(avg_pos)
                    if var_name == "home_formation":
                        home_formation = f"{formation} ({conf:.0%})"
                    else:
                        away_formation = f"{formation} ({conf:.0%})"
    print(f"  Home formation: {home_formation}")
    print(f"  Away formation: {away_formation}")
    summary["home_formation"] = home_formation
    summary["away_formation"] = away_formation

    # =====================================================================
    # Step 6: Player Identification
    # =====================================================================
    print(f"\n[6/8] Running player identification...")
    try:
        from pipeline.fingerprint import PlayerFingerprinter
        from database import RosterDB

        roster_db = RosterDB()
        # Fingerprinter requires face_reid, gait_analyzer, cleat_extractor
        # Skip if those aren't available yet
        fingerprinter = PlayerFingerprinter(
            roster_db=roster_db,
            face_reid=None,
            gait_analyzer=None,
            cleat_extractor=None,
        )
        home_df = df[df["team"] == "Home"].copy()
        if not home_df.empty:
            track_ids = home_df["track_id"].unique()
            assignments = {}
            for tid in track_ids:
                result = fingerprinter.identify_track(tid, home_df, frames)
                assignments[tid] = result
            n_identified = sum(1 for v in assignments.values()
                              if v and v.get("confidence", 0) > 0.5)
            print(f"  Identified {n_identified}/{len(assignments)} home tracks")
            summary["players_identified"] = n_identified
        else:
            print(f"  No home team detections for identification")
    except Exception as e:
        print(f"  ⚠️ Player identification skipped: {e}")
        summary["players_identified"] = 0

    # =====================================================================
    # Step 7: Pass Detection
    # =====================================================================
    print(f"\n[7/8] Running pass detection...")
    from pipeline.passes import PassDetector

    pass_detector = PassDetector()
    passes = pass_detector.detect_passes(df, fps=fps,
                                         field_length=field_length,
                                         field_width=field_width)
    print(f"  Detected {len(passes)} passes")
    summary["passes_detected"] = len(passes)

    # =====================================================================
    # Step 8: Summary
    # =====================================================================
    total_time = time.time() - t_start
    summary["total_time_s"] = round(total_time, 1)
    print(f"\n{'='*60}")
    print(f"SCRIMMAGE DIAGNOSTIC COMPLETE — {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"  Detections: {summary['detections']}, Tracks: {summary['tracks']}")
    print(f"  Homography: {'flag-based' if use_flags else 'density-based'} (score={val_score:.2f})")
    print(f"  Teams: Home={home_count}, Away={away_count}")
    print(f"  Formation: Home={home_formation}, Away={away_formation}")
    print(f"  Output: {outdir}/")

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary.json")


if __name__ == "__main__":
    main()
