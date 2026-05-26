#!/usr/bin/env python3
"""
Drill/Training Diagnostic Tool — validates the fisheye + flag + drill segmentation pipeline.

Usage:
    python diagnostics_drill.py "/path/to/training.mov" \
        --calibration data/calibration/neewer_fisheye.npz \
        --flag-color orange --field-size 40x30 --frames 120

Outputs to diagnostics_drill/ folder:
    00_undistort/        — before/after fisheye correction samples
    01_flags/            — detected flag positions + homography overlay
    02_detections/       — annotated frames with bboxes + track IDs
    03_player_id/        — player identification results (all 16 players)
    04_segmentation/     — drill boundary timeline + intensity chart
    05_drill_metrics/    — per-drill × per-player metric tables
    06_session_summary/  — aggregated per-player development metrics
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
import matplotlib.pyplot as plt

from config import (DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SAMPLE_RATE, YOLO_IMGSZ,
                    DEFAULT_MODEL_SIZE, SPRINT_THRESHOLD, MAX_SPEED_CAP)


def parse_args():
    p = argparse.ArgumentParser(description="Drill/Training pipeline diagnostic tool")
    p.add_argument("video", help="Path to training video file")
    p.add_argument("--calibration", type=str, required=True,
                   help="Path to fisheye calibration .npz file")
    p.add_argument("--flag-color", type=str, default="orange",
                   choices=["orange", "pink", "yellow", "green_neon", "blue_neon"],
                   help="Color of field marker flags (default: orange)")
    p.add_argument("--field-size", type=str, default="40x30",
                   help="Field dimensions as LxW in meters (default: 40x30)")
    p.add_argument("--start", type=int, default=0,
                   help="Start frame (default: 0)")
    p.add_argument("--frames", type=int, default=120,
                   help="Number of sampled frames to process (default: 120)")
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
    p.add_argument("--idle-threshold", type=float, default=1.0,
                   help="Speed below which players are idle, m/s (default: 1.0)")
    p.add_argument("--min-drill", type=float, default=30.0,
                   help="Minimum drill duration in seconds (default: 30)")
    p.add_argument("--outdir", type=str, default="diagnostics_output/diagnostics_drill",
                   help="Output directory (default: diagnostics_output/diagnostics_drill/)")
    return p.parse_args()


def setup_dirs(outdir):
    subdirs = ["00_undistort", "01_flags", "02_detections", "03_player_id",
               "04_segmentation", "05_drill_metrics", "06_session_summary"]
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

    try:
        field_length, field_width = map(float, args.field_size.split("x"))
    except ValueError:
        print(f"Error: Invalid field-size '{args.field_size}'. Use LxW (e.g., 40x30)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DRILL/TRAINING PIPELINE DIAGNOSTIC")
    print(f"{'='*60}")

    summary = {"session_type": "drill", "args": vars(args)}
    t_start = time.time()

    # =====================================================================
    # Step 0: Fisheye Undistortion
    # =====================================================================
    print(f"\n[0/6] Loading fisheye calibration...")
    from pipeline.fisheye import FisheyeCalibration

    calib = FisheyeCalibration(Path(args.calibration))
    print(f"  Calibration loaded: {args.calibration}")

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

    # Save undistortion samples
    for idx in [args.start, args.start + 100, args.start + 200]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            undistorted = calib.undistort(frame)
            h, w = frame.shape[:2]
            scale = 1280 / w if w > 1280 else 1.0
            if scale < 1.0:
                frame_s = cv2.resize(frame, (int(w * scale), int(h * scale)))
                undist_s = cv2.resize(undistorted, (int(w * scale), int(h * scale)))
            else:
                frame_s, undist_s = frame, undistorted
            comparison = np.hstack([frame_s, undist_s])
            cv2.imwrite(os.path.join(dirs["00_undistort"], f"compare_{idx:06d}.jpg"),
                        comparison, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  Saved undistortion comparisons to {dirs['00_undistort']}/")

    # =====================================================================
    # Step 0.5: Flag Detection + Homography
    # =====================================================================
    print(f"\n[0.5/6] Detecting field flags...")
    from pipeline.flag_detector import FlagDetector
    from pipeline.homography import FlagHomography

    flag_detector = FlagDetector(flag_color=args.flag_color)

    flag_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            flag_frames.append(calib.undistort(frame))

    centroids = flag_detector.detect_stable(flag_frames) if flag_frames else []
    print(f"  Detected {len(centroids)} flags")

    use_flags = False
    homography = FlagHomography(field_length, field_width)

    if len(centroids) >= 4:
        corners = flag_detector.assign_corners(centroids, flag_frames[0].shape)
        if corners:
            homography.calibrate_from_flags(corners)
            use_flags = True
            print(f"  ✓ Flag-based homography: {field_length}x{field_width}m")

    if not use_flags:
        print(f"  ⚠️ Flag detection incomplete, will use density-based fallback")

    # Save flag visualization
    if flag_frames:
        vis = flag_detector.visualize(flag_frames[0], centroids,
                                      corners if use_flags else None)
        h, w = vis.shape[:2]
        if w > 1920:
            vis = cv2.resize(vis, (1920, int(h * 1920 / w)))
        cv2.imwrite(os.path.join(dirs["01_flags"], "flags.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    summary["flags_detected"] = len(centroids)
    summary["flag_homography"] = use_flags

    # =====================================================================
    # Step 1: Detection + Tracking
    # =====================================================================
    print(f"\n[1/6] Running YOLO detection + {args.tracker}...")
    from ultralytics import YOLO
    import supervision as sv

    if args.yolo_version == "11":
        model = YOLO(f"yolo11{args.model}.pt")
    else:
        model = YOLO(f"yolov8{args.model}.pt")

    confidence = args.confidence
    if confidence == DEFAULT_CONFIDENCE:
        confidence = 0.25 if frame_w >= 3840 else 0.30 if frame_w >= 1920 else 0.35

    effective_fps = fps / args.sample_rate
    track_buffer = int(120 / args.sample_rate)

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

    print(f"  Model: yolo{'11' if args.yolo_version == '11' else 'v8'}{args.model}, "
          f"conf={confidence}, tracker={args.tracker}")

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
                                all_detections.append({
                                    "frame": frame_id, "track_id": int(t[4]),
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
            if processed % 10 == 0:
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
        print("  ⚠️ No detections. Exiting.")
        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Save sample detection frames
    for fid in sorted(frames.keys())[:5]:
        frame = frames[fid].copy()
        frame_dets = df[df["frame"] == fid]
        for _, det in frame_dets.iterrows():
            x1, y1 = int(det.bbox_x1), int(det.bbox_y1)
            x2, y2 = int(det.bbox_x2), int(det.bbox_y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"T{int(det.track_id)}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h, w = frame.shape[:2]
        if w > 1920:
            frame = cv2.resize(frame, (1920, int(h * 1920 / w)))
        cv2.imwrite(os.path.join(dirs["02_detections"], f"det_{fid:06d}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # =====================================================================
    # Step 2: Homography + Transform
    # =====================================================================
    print(f"\n[2/6] Applying homography...")
    if not use_flags:
        homography.calibrate_auto(frame_h, frame_w, df)

    pre = len(df)
    df = homography.filter_to_field_zone(df)
    print(f"  Field filter: {pre} → {len(df)} detections")
    df = homography.transform_df(df)

    # =====================================================================
    # Step 3: Player Identification (all players, no team split)
    # =====================================================================
    print(f"\n[3/6] Running player identification (all players)...")
    from pipeline.fingerprint import PlayerFingerprinter

    fingerprinter = PlayerFingerprinter()
    assignments = fingerprinter.identify(df, frames)
    n_identified = sum(1 for v in assignments.values() if v is not None)
    print(f"  Identified {n_identified}/{len(assignments)} tracks")
    summary["players_identified"] = n_identified

    # Apply player names to df
    if assignments:
        name_map = {tid: name for tid, name in assignments.items() if name is not None}
        df["player_name"] = df["track_id"].map(name_map)

    # =====================================================================
    # Step 4: Drill Segmentation
    # =====================================================================
    print(f"\n[4/6] Segmenting drills...")
    from pipeline.drill_segmenter import DrillSegmenter

    segmenter = DrillSegmenter(
        idle_threshold=args.idle_threshold,
        min_drill_s=args.min_drill,
    )
    segments = segmenter.segment(df, fps)
    print(f"  Found {len(segments)} drill segments:")
    for seg in segments:
        print(f"    [{seg.index}] {seg.start_time_s:.0f}s–{seg.end_time_s:.0f}s "
              f"({seg.duration_s:.0f}s) type={seg.drill_type} "
              f"intensity={seg.avg_intensity:.1f}m/s players={seg.player_count}")

    summary["drill_segments"] = len(segments)
    summary["segments"] = [
        {"index": s.index, "start_s": s.start_time_s, "end_s": s.end_time_s,
         "duration_s": s.duration_s, "type": s.drill_type,
         "intensity": s.avg_intensity, "players": s.player_count}
        for s in segments
    ]

    # Visualize segmentation timeline
    if segments:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        # Activity signal
        activity = segmenter._compute_activity_signal(df, fps)
        ax1.plot(activity, color="steelblue", linewidth=0.8)
        ax1.axhline(args.idle_threshold, color="red", linestyle="--", alpha=0.7, label="Idle threshold")
        ax1.set_ylabel("Avg Speed (m/s)")
        ax1.set_title("Training Session Activity Signal")
        ax1.legend()

        # Segment bars
        colors = {"sprint": "red", "possession": "orange", "passing": "green",
                  "agility": "purple", "tactical": "blue", "general": "gray", "unknown": "lightgray"}
        for seg in segments:
            ax2.barh(0, seg.duration_s, left=seg.start_time_s,
                     color=colors.get(seg.drill_type, "gray"), edgecolor="black", linewidth=0.5)
            ax2.text(seg.start_time_s + seg.duration_s / 2, 0,
                     f"D{seg.index}\n{seg.drill_type}",
                     ha="center", va="center", fontsize=8)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Drills")
        ax2.set_yticks([])
        ax2.set_title("Detected Drill Segments")

        fig.tight_layout()
        fig.savefig(os.path.join(dirs["04_segmentation"], "drill_timeline.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved drill timeline to {dirs['04_segmentation']}/")

    # =====================================================================
    # Step 5: Per-Drill Metrics
    # =====================================================================
    print(f"\n[5/6] Computing per-drill metrics...")
    from pipeline.drill_metrics import DrillMetricsCalculator

    metrics_calc = DrillMetricsCalculator()
    metrics_df = metrics_calc.compute_all(df, segments, fps)

    if not metrics_df.empty:
        print(f"  Computed metrics for {len(metrics_df)} player×drill combinations")

        # Save per-drill tables
        for seg in segments:
            seg_metrics = metrics_df[metrics_df["segment_index"] == seg.index]
            if seg_metrics.empty:
                continue
            seg_metrics_sorted = seg_metrics.sort_values("distance_m", ascending=False)
            seg_metrics_sorted.to_csv(
                os.path.join(dirs["05_drill_metrics"], f"drill_{seg.index}_{seg.drill_type}.csv"),
                index=False
            )

        # Summary chart: distance per player per drill
        if len(segments) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot = metrics_df.pivot_table(
                index="track_id", columns="segment_index",
                values="distance_m", aggfunc="sum"
            ).fillna(0)
            pivot.plot(kind="bar", ax=ax, width=0.8)
            ax.set_xlabel("Track ID")
            ax.set_ylabel("Distance (m)")
            ax.set_title("Distance per Player per Drill")
            ax.legend(title="Drill #", bbox_to_anchor=(1.05, 1))
            fig.tight_layout()
            fig.savefig(os.path.join(dirs["05_drill_metrics"], "distance_by_drill.png"),
                        dpi=100, bbox_inches="tight")
            plt.close(fig)

        print(f"  Saved drill metrics to {dirs['05_drill_metrics']}/")
    else:
        print(f"  No metrics computed (check data)")

    # =====================================================================
    # Step 6: Session Summary
    # =====================================================================
    print(f"\n[6/6] Computing session summary...")

    if not metrics_df.empty:
        session_summary = metrics_calc.summarize_session(metrics_df)
        session_summary.to_csv(
            os.path.join(dirs["06_session_summary"], "session_totals.csv"),
            index=False
        )

        # Per-player summary chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if "total_distance_m" in session_summary.columns:
            session_summary.sort_values("total_distance_m", ascending=True).plot.barh(
                x=session_summary.columns[0], y="total_distance_m",
                ax=axes[0, 0], color="steelblue", legend=False
            )
            axes[0, 0].set_title("Total Distance (m)")

        if "max_speed_ms" in session_summary.columns:
            session_summary.sort_values("max_speed_ms", ascending=True).plot.barh(
                x=session_summary.columns[0], y="max_speed_ms",
                ax=axes[0, 1], color="coral", legend=False
            )
            axes[0, 1].set_title("Max Speed (m/s)")

        if "total_sprints" in session_summary.columns:
            session_summary.sort_values("total_sprints", ascending=True).plot.barh(
                x=session_summary.columns[0], y="total_sprints",
                ax=axes[1, 0], color="green", legend=False
            )
            axes[1, 0].set_title("Total Sprints")

        if "avg_time_active_pct" in session_summary.columns:
            session_summary.sort_values("avg_time_active_pct", ascending=True).plot.barh(
                x=session_summary.columns[0], y="avg_time_active_pct",
                ax=axes[1, 1], color="purple", legend=False
            )
            axes[1, 1].set_title("Avg Time Active (%)")

        fig.suptitle("Training Session — Player Summary", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["06_session_summary"], "player_summary.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved session summary to {dirs['06_session_summary']}/")
        print(f"\n  Top performers:")
        top = session_summary.nlargest(3, "total_distance_m")
        for _, row in top.iterrows():
            pid = row[session_summary.columns[0]]
            print(f"    {pid}: {row['total_distance_m']:.0f}m, "
                  f"max {row['max_speed_ms']:.1f}m/s, "
                  f"{int(row['total_sprints'])} sprints")

    # Final summary
    total_time = time.time() - t_start
    summary["total_time_s"] = round(total_time, 1)
    print(f"\n{'='*60}")
    print(f"DRILL DIAGNOSTIC COMPLETE — {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"  Detections: {summary['detections']}, Tracks: {summary['tracks']}")
    print(f"  Drills found: {len(segments)}")
    print(f"  Players identified: {summary.get('players_identified', 0)}")
    print(f"  Output: {outdir}/")

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
