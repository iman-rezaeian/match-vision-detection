#!/usr/bin/env python3
"""
Game Pipeline Diagnostic Tool — fast, visual validation of detection + tracking + classification + analytics.

Usage:
    python diagnostics_game.py "/path/to/video.mov"
    python diagnostics_game.py "/path/to/video.mov" --start 300 --frames 30 --sample-rate 5
    python diagnostics_game.py "/path/to/video.mov" --my-team black --frames 60

Outputs to diagnostics/ folder:
    01_detections/       — annotated frames with bboxes + track IDs
    02_torso_crops/      — what the color extractor actually sees per track
    03_color_swatches/   — extracted BGR color per track (before clustering)
    04_team_assignment/  — annotated frames with team color overlay
    05_track_timeline.png — track lifetime chart (which tracks span which frames)
    06_color_clusters.png — 2D PCA of track colors, colored by team assignment
    07_detection_heatmap.png — detection center scatter + bbox size distribution
    08_field_positions.png — team positions on a pitch diagram (field coords)
    09_movement_stats.png — per-track distance, speed, sprint bar charts
    10_formation.png     — detected formation snapshot per team
    11_team_territory.png — team territorial dominance + compactness
    12_player_id.png     — player identification results (face/gait/cleat fusion)
    13_player_stats.png  — per-player stats table (distance, speed, sprints, passes)
    14_pass_map.png      — detected passes on a pitch diagram
    summary.txt          — text report of all metrics
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add parent so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
from collections import Counter

# Pipeline imports
from config import (DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SAMPLE_RATE, YOLO_IMGSZ,
                    DEFAULT_MODEL_SIZE, DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M,
                    DEFAULT_PLAYERS_PER_TEAM, SPRINT_THRESHOLD, MAX_SPEED_CAP,
                    PITCH_GRASS, PITCH_LINES, TEAM_A_BLUE, TEAM_B_RED)


def parse_args():
    p = argparse.ArgumentParser(description="Soccer pipeline diagnostic tool")
    p.add_argument("video", help="Path to input video file")
    p.add_argument("--start", type=int, default=300,
                   help="Start frame (default: 300 = ~10s in)")
    p.add_argument("--frames", type=int, default=30,
                   help="Number of sampled frames to process (default: 30)")
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
                   help="YOLO version: 8 or 11 (default: 8)")
    p.add_argument("--outdir", type=str, default="diagnostics_output/diagnostics",
                   help="Output directory (default: diagnostics_output/diagnostics/)")
    p.add_argument("--my-team", type=str, default=None,
                   choices=["black", "white", "red", "blue", "green", "yellow", "orange", "purple"],
                   help="My team's jersey color for auto Home/Away assignment")
    p.add_argument("--opponent", type=str, default=None,
                   choices=["black", "white", "red", "blue", "green", "yellow", "orange", "purple"],
                   help="Opponent's jersey color for better team separation")
    return p.parse_args()


def setup_dirs(outdir):
    dirs = {}
    for name in ["01_detections", "02_torso_crops", "03_color_swatches",
                  "04_team_assignment"]:
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        dirs[name] = d
    os.makedirs(outdir, exist_ok=True)
    return dirs


def run_detection(video_path, start_frame, n_frames, sample_rate, confidence, model_size,
                  tracker_type="botsort", yolo_version="8"):
    """Run YOLO + tracker on a short clip. Returns detections_df, frames dict, video_meta."""
    from ultralytics import YOLO
    import supervision as sv

    if yolo_version == "11":
        model = YOLO(f"yolo11{model_size}.pt")
    else:
        model = YOLO(f"yolov8{model_size}.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto-tune confidence based on resolution (matches app)
    if confidence == DEFAULT_CONFIDENCE:  # only override if user didn't specify
        confidence = 0.25 if frame_w >= 3840 else 0.30 if frame_w >= 1920 else 0.35

    effective_fps = 30.0 / sample_rate
    track_buffer = int(120 / sample_rate)

    if tracker_type == "botsort":
        from boxmot import BotSort
        from pathlib import Path
        tracker = BotSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device="cpu",  # ReID model runs on CPU (MPS not supported by boxmot)
            half=False,
            track_high_thresh=confidence,
            track_low_thresh=0.1,
            new_track_thresh=confidence,
            track_buffer=track_buffer,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method="sof",
            frame_rate=int(effective_fps),
            with_reid=True,
        )
    elif tracker_type == "botsort_noid":
        from boxmot import BotSort
        tracker = BotSort(
            reid_weights=None,
            device="cpu",
            half=False,
            track_high_thresh=confidence,
            track_low_thresh=0.1,
            new_track_thresh=confidence,
            track_buffer=track_buffer,
            match_thresh=0.7,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method="sof",
            frame_rate=int(effective_fps),
            with_reid=False,
        )
    else:
        tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.25,
            frame_rate=effective_fps,
        )

    video_meta = {
        "fps": fps, "total_frames": total_frames,
        "frame_w": frame_w, "frame_h": frame_h,
    }

    model_name = f"yolo11{model_size}" if yolo_version == "11" else f"yolov8{model_size}"
    print(f"  Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames "
          f"({total_frames/fps/60:.1f} min)")
    print(f"  Processing frames {start_frame}..{start_frame + n_frames * sample_rate} "
          f"(every {sample_rate}th frame, {n_frames} detections)")
    print(f"  Model: {model_name}, imgsz={YOLO_IMGSZ}, conf={confidence}, device={DEVICE}")
    print(f"  Tracker: {tracker_type}, effective_fps={effective_fps}, buffer={track_buffer}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_detections = []
    frames = {}
    processed = 0
    frame_id = start_frame
    MIN_BBOX_AREA = 400

    t0 = time.time()
    while processed < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % sample_rate == 0:
            frames[frame_id] = frame.copy()

            results = model(frame, conf=confidence, classes=[0],
                            device=DEVICE, imgsz=YOLO_IMGSZ, verbose=False)

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                # Filter tiny detections
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                valid = areas >= MIN_BBOX_AREA
                xyxy = xyxy[valid]
                confs = confs[valid]

                if len(xyxy) == 0:
                    processed += 1
                    frame_id += 1
                    continue

                if tracker_type in ("botsort", "botsort_noid"):
                    # boxmot expects [x1,y1,x2,y2,conf,cls]
                    dets_arr = np.column_stack([
                        xyxy, confs, np.zeros(len(xyxy))
                    ])
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
            elapsed = time.time() - t0
            print(f"\r  Frame {frame_id} [{processed}/{n_frames}] "
                  f"({elapsed:.1f}s, {processed/elapsed:.1f} fps)", end="", flush=True)

        frame_id += 1

    cap.release()
    del model
    print()

    df = pd.DataFrame(all_detections) if all_detections else pd.DataFrame()

    # Add time_s and segment_id columns for downstream stages
    if not df.empty:
        df["time_s"] = (df["frame"] - df["frame"].min()) / max(fps, 1.0)
        df["segment_id"] = 0

    return df, frames, video_meta


def draw_detections(df, frames, outdir, max_save=100):
    """Save annotated frames with bboxes, track IDs, and confidence."""
    print(f"  Saving annotated detection frames...")
    # Assign consistent colors per track
    track_ids = df["track_id"].unique() if not df.empty else []
    rng = np.random.RandomState(42)
    colors = {tid: tuple(rng.randint(60, 255, 3).tolist()) for tid in track_ids}

    all_frames = sorted(frames.keys())
    # Only save a sample of frames to avoid I/O bottleneck on large clips
    if len(all_frames) > max_save:
        step = len(all_frames) // max_save
        all_frames = all_frames[::step][:max_save]

    saved = 0
    for frame_num in all_frames:
        frame = frames[frame_num].copy()
        frame_dets = df[df["frame"] == frame_num] if not df.empty else pd.DataFrame()

        for _, row in frame_dets.iterrows():
            tid = int(row["track_id"])
            c = colors.get(tid, (255, 255, 255))
            x1, y1 = int(row["bbox_x1"]), int(row["bbox_y1"])
            x2, y2 = int(row["bbox_x2"]), int(row["bbox_y2"])
            conf = row["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            label = f"T{tid} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), c, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add frame number overlay
        cv2.putText(frame, f"Frame {frame_num}  |  {len(frame_dets)} detections",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Save at reduced resolution (1920 wide) for viewability
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            frame = cv2.resize(frame, (1920, int(h * scale)))

        path = os.path.join(outdir, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1

    print(f"  Saved {saved} annotated frames to {outdir}/")
    return saved


def draw_torso_crops(df, frames, outdir):
    """Save the actual torso crop region per track, showing what the color extractor sees.
    
    Mirrors the exact logic from TeamClassifier.extract_jersey_color:
    - Tight torso crop
    - Green field filter, skin filter (S<=130), green post-filter
    - Saturation-boost: top 40% most-saturated pixels for color
    - Dark-jersey fallback (median sat < 15 → use all filtered pixels)
    """
    print(f"  Saving torso crop diagnostics...")
    if df.empty:
        return 0

    track_ids = df["track_id"].unique()
    saved = 0

    for tid in track_ids:
        track = df[df["track_id"] == tid]
        # Pick the frame with the largest bbox for this track
        track = track.copy()
        track["_area"] = (track["bbox_x2"] - track["bbox_x1"]) * (track["bbox_y2"] - track["bbox_y1"])
        best = track.sort_values("_area", ascending=False).iloc[0]
        frame_num = int(best["frame"])
        if frame_num not in frames:
            continue

        frame = frames[frame_num]
        x1, y1 = int(best["bbox_x1"]), int(best["bbox_y1"])
        x2, y2 = int(best["bbox_x2"]), int(best["bbox_y2"])
        h = y2 - y1
        w = x2 - x1

        # Reproduce the exact torso crop logic from team_classifier
        if w < 30:
            ty1 = y1 + int(h * 0.22)
            ty2 = y1 + int(h * 0.52)
            tx1 = x1 + int(w * 0.15)
            tx2 = x2 - int(w * 0.15)
        else:
            ty1 = y1 + int(h * 0.25)
            ty2 = y1 + int(h * 0.50)
            tx1 = x1 + int(w * 0.25)
            tx2 = x2 - int(w * 0.25)

        ty1 = max(0, ty1)
        ty2 = min(frame.shape[0], ty2)
        tx1 = max(0, tx1)
        tx2 = min(frame.shape[1], tx2)

        if ty2 <= ty1 or tx2 <= tx1:
            continue

        # Draw on a copy: full bbox in green, torso crop in red
        vis = frame[max(0, y1-10):min(frame.shape[0], y2+10),
                     max(0, x1-10):min(frame.shape[1], x2+10)].copy()
        ox, oy = max(0, x1-10), max(0, y1-10)
        cv2.rectangle(vis, (x1-ox, y1-oy), (x2-ox, y2-oy), (0, 255, 0), 2)
        cv2.rectangle(vis, (tx1-ox, ty1-oy), (tx2-ox, ty2-oy), (0, 0, 255), 3)

        # --- Exact pipeline color extraction logic ---
        torso_crop = frame[ty1:ty2, tx1:tx2]
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        pixels_hsv = hsv.reshape(-1, 3).astype(np.float32)
        pixels_bgr = torso_crop.reshape(-1, 3).astype(np.float32)

        # Green field filter
        green_mask = (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) & (pixels_hsv[:, 1] > 40)
        # Skin filter (S capped at 130 to preserve saturated reds)
        skin_mask = (
            (pixels_hsv[:, 0] <= 25) &
            (pixels_hsv[:, 1] >= 30) & (pixels_hsv[:, 1] <= 130) &
            (pixels_hsv[:, 2] > 60) & (pixels_hsv[:, 2] < 230)
        )
        filtered = pixels_bgr[~green_mask & ~skin_mask]

        relaxed = False
        if len(filtered) < 5:
            filtered = pixels_bgr[~green_mask]
            relaxed = True

        # Saturation-boost: pick top 40% most-saturated pixels
        sat_boosted = False
        dark_jersey = False
        if len(filtered) >= 5:
            filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
            filtered_hsv_arr = cv2.cvtColor(
                filtered_uint8.reshape(1, -1, 3), cv2.COLOR_BGR2HSV
            ).reshape(-1, 3)
            sat_values = filtered_hsv_arr[:, 1].astype(np.float32)

            if np.median(sat_values) < 15:
                # Dark jersey — use all filtered pixels
                dark_jersey = True
                final_color = np.median(filtered, axis=0).astype(np.uint8)
            else:
                sat_boosted = True
                n_take = max(3, int(len(sat_values) * 0.4))
                top_idx = np.argsort(sat_values)[-n_take:]
                top_pixels = filtered[top_idx]
                final_color = np.median(top_pixels, axis=0).astype(np.uint8)
        elif len(filtered) > 0:
            final_color = np.median(filtered, axis=0).astype(np.uint8)
        else:
            final_color = np.array([128, 128, 128], dtype=np.uint8)

        # --- Create composite visualization ---
        th, tw = torso_crop.shape[:2]

        # Scale player vis to ~200px tall
        vh, vw = vis.shape[:2]
        scale = min(200 / vh, 200 / vw) if vh > 0 and vw > 0 else 1
        vis_resized = cv2.resize(vis, (max(1, int(vw*scale)), max(1, int(vh*scale))))

        # Scale torso crop to match
        torso_resized = cv2.resize(torso_crop, (max(1, int(tw*scale)), max(1, int(th*scale))))

        # Create mask visualization (green=field, pink=skin, white=jersey)
        mask_vis = np.zeros_like(torso_crop)
        mask_flat = mask_vis.reshape(-1, 3)
        mask_flat[green_mask] = [0, 255, 0]          # green = field
        mask_flat[skin_mask & ~green_mask] = [180, 105, 255]  # pink = skin
        mask_flat[~green_mask & ~skin_mask] = [255, 255, 255]  # white = jersey
        mask_vis = mask_flat.reshape(torso_crop.shape)
        mask_resized = cv2.resize(mask_vis, (max(1, int(tw*scale)), max(1, int(th*scale))))

        # Color swatch (final extracted color)
        swatch = np.full((vis_resized.shape[0], 80, 3), final_color, dtype=np.uint8)

        # Stats text on swatch
        n_green = green_mask.sum()
        n_skin = (skin_mask & ~green_mask).sum()
        n_jersey = (~green_mask & ~skin_mask).sum()
        cv2.putText(swatch, f"J:{n_jersey}", (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(swatch, f"S:{n_skin}", (2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(swatch, f"G:{n_green}", (2, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        # Show which mode was used
        mode = "SAT40%" if sat_boosted else ("DARK" if dark_jersey else ("RELAX" if relaxed else "MED"))
        cv2.putText(swatch, mode, (2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        # Show RGB value
        rgb_str = f"{final_color[2]},{final_color[1]},{final_color[0]}"
        cv2.putText(swatch, rgb_str, (2, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Pad all to same height
        target_h = vis_resized.shape[0]
        parts = []
        for img in [vis_resized, torso_resized, mask_resized, swatch]:
            ih, iw = img.shape[:2]
            if ih < target_h:
                pad = np.zeros((target_h - ih, iw, 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            elif ih > target_h:
                img = img[:target_h]
            parts.append(img)

        composite = np.hstack(parts)

        # Add track ID label
        cv2.putText(composite, f"Track {tid} (bbox {w}x{h})",
                    (5, composite.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        path = os.path.join(outdir, f"track_{tid:04d}.jpg")
        cv2.imwrite(path, composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1

    print(f"  Saved {saved} torso crop diagnostics to {outdir}/")
    return saved


def draw_team_assignment(df, frames, team_colors, team_map, outdir):
    """Save frames with team-colored bboxes after classification."""
    print(f"  Saving team-assigned frames...")
    home_bgr = team_colors.get("Home", np.array([255, 0, 0]))
    away_bgr = team_colors.get("Away", np.array([0, 0, 255]))

    saved = 0
    for frame_num in sorted(frames.keys()):
        frame = frames[frame_num].copy()
        frame_dets = df[df["frame"] == frame_num] if not df.empty else pd.DataFrame()

        for _, row in frame_dets.iterrows():
            tid = int(row["track_id"])
            team = team_map.get(tid, "Unknown")
            if team == "Home":
                c = tuple(int(x) for x in home_bgr)
            elif team == "Away":
                c = tuple(int(x) for x in away_bgr)
            else:
                c = (128, 128, 128)

            x1, y1 = int(row["bbox_x1"]), int(row["bbox_y1"])
            x2, y2 = int(row["bbox_x2"]), int(row["bbox_y2"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            label = f"T{tid} {team}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Team color legend
        cv2.putText(frame, "HOME", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    tuple(int(x) for x in home_bgr), 3)
        cv2.putText(frame, "AWAY", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    tuple(int(x) for x in away_bgr), 3)

        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            frame = cv2.resize(frame, (1920, int(h * scale)))

        path = os.path.join(outdir, f"team_{frame_num:06d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1

    print(f"  Saved {saved} team-assigned frames to {outdir}/")
    return saved


def draw_track_timeline(df, outdir):
    """Save a chart showing track lifetimes across frames."""
    print(f"  Generating track timeline...")
    if df.empty:
        print("  No detections — skipping timeline.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track_ids = sorted(df["track_id"].unique())
    fig, ax = plt.subplots(figsize=(16, max(4, len(track_ids) * 0.25)))

    for i, tid in enumerate(track_ids):
        track = df[df["track_id"] == tid]
        frames_list = sorted(track["frame"].unique())
        n_dets = len(track)
        ax.barh(i, frames_list[-1] - frames_list[0] + 1,
                left=frames_list[0], height=0.7, alpha=0.7,
                label=f"T{tid}" if i < 30 else None)
        ax.text(frames_list[-1] + 2, i, f"T{tid} ({n_dets}det)",
                fontsize=7, va="center")

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Track")
    ax.set_title(f"Track Timeline — {len(track_ids)} tracks, "
                 f"{len(df)} total detections")
    ax.set_yticks(range(len(track_ids)))
    ax.set_yticklabels([f"T{t}" for t in track_ids], fontsize=6)
    plt.tight_layout()

    path = os.path.join(outdir, "05_track_timeline.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved track timeline to {path}")


def draw_color_clusters(track_colors, team_map, team_color_centers, outdir):
    """Save a visualization of the color clustering with 3-cluster diagnostics."""
    print(f"  Generating color cluster plot...")
    if not track_colors:
        print("  No track colors — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tids = list(track_colors.keys())
    bgr_arr = np.array([track_colors[t] for t in tids])

    # Convert to RGB for display
    rgb_arr = bgr_arr[:, ::-1] / 255.0

    # Use first two principal components for scatter
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(bgr_arr)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Left: PCA scatter colored by extracted color
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], c=rgb_arr, s=80, edgecolors="black",
               linewidths=0.5)
    for i, tid in enumerate(tids):
        ax.annotate(f"T{tid}", (coords[i, 0], coords[i, 1]), fontsize=5,
                    alpha=0.6)
    ax.set_title("Track colors (PCA) — by jersey color")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Middle: same scatter but colored by team assignment
    ax = axes[1]
    team_colors_plot = []
    for tid in tids:
        team = team_map.get(tid, "Unknown")
        if team == "Home":
            team_colors_plot.append("blue")
        elif team == "Away":
            team_colors_plot.append("red")
        else:
            team_colors_plot.append("gray")
    ax.scatter(coords[:, 0], coords[:, 1], c=team_colors_plot, s=80,
               edgecolors="black", linewidths=0.5)
    for i, tid in enumerate(tids):
        ax.annotate(f"T{tid}", (coords[i, 0], coords[i, 1]), fontsize=5,
                    alpha=0.6)

    home_bgr = team_color_centers.get("Home", np.array([128, 128, 128]))
    away_bgr = team_color_centers.get("Away", np.array([128, 128, 128]))
    ax.set_title(f"Team assignment — Home: RGB({home_bgr[2]:.0f},{home_bgr[1]:.0f},{home_bgr[0]:.0f}) "
                 f"Away: RGB({away_bgr[2]:.0f},{away_bgr[1]:.0f},{away_bgr[0]:.0f})")

    for label, bgr, ypos in [("Home", home_bgr, 0.95), ("Away", away_bgr, 0.88)]:
        rgb = [bgr[2]/255, bgr[1]/255, bgr[0]/255]
        ax.annotate(f"■ {label}", xy=(0.02, ypos), xycoords="axes fraction",
                    fontsize=12, color=rgb, fontweight="bold")

    # Right: 3-cluster visualization (replicate pipeline clustering)
    ax = axes[2]
    from sklearn.cluster import KMeans as KM
    if len(bgr_arr) >= 3:
        bgr_img = bgr_arr.reshape(1, -1, 3).astype(np.uint8)
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        lab_arr = lab_img.reshape(-1, 3).astype(np.float32)
        km3 = KM(n_clusters=min(3, len(lab_arr)), random_state=42, n_init=10)
        labels3 = km3.fit_predict(lab_arr)
        c3 = km3.cluster_centers_
        # Show which clusters were selected
        l_vals = c3[:, 0]
        black_idx = int(np.argmin(l_vals))
        remaining = [i for i in range(len(c3)) if i != black_idx]
        chroma = [np.sqrt((c3[i][1]-128)**2 + (c3[i][2]-128)**2) for i in remaining]
        color_idx = remaining[int(np.argmax(chroma))]
        noise_idx = [i for i in range(len(c3)) if i != black_idx and i != color_idx]

        cluster_colors_plot = []
        for l in labels3:
            if l == black_idx:
                cluster_colors_plot.append("black")
            elif l == color_idx:
                cluster_colors_plot.append("red")
            else:
                cluster_colors_plot.append("lightgray")

        ax.scatter(coords[:, 0], coords[:, 1], c=cluster_colors_plot, s=80,
                   edgecolors="black", linewidths=0.5)

        # Show centers
        for ci, (name, color) in enumerate([(f"C{black_idx}:DARK", "black"),
                                             (f"C{color_idx}:COLOR", "red")] +
                                            [(f"C{ni}:NOISE", "gray") for ni in noise_idx]):
            center_bgr = cv2.cvtColor(c3[int(name[1])].reshape(1,1,3).astype(np.uint8),
                                       cv2.COLOR_LAB2BGR).reshape(3)
            ax.annotate(f"■ {name} RGB({center_bgr[2]},{center_bgr[1]},{center_bgr[0]})",
                       xy=(0.02, 0.95 - ci * 0.07), xycoords="axes fraction",
                       fontsize=9, color=color, fontweight="bold")
        ax.set_title("3-cluster selection (dark + most-colorful)")
    else:
        ax.text(0.5, 0.5, "Too few tracks for 3-cluster", ha="center", va="center")
        ax.set_title("3-cluster (N/A)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    path = os.path.join(outdir, "06_color_clusters.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved color cluster plot to {path}")


def run_team_classification(df, frames, my_team_color=None, opponent_color=None):
    """Run the team classifier and return team_map, track_colors, team_color_centers."""
    if df.empty:
        return {}, {}, {}
    from pipeline.team_classifier import TeamClassifier
    tc = TeamClassifier()
    df_with_teams = tc.classify_teams(df, frames, sample_per_track=5,
                                       my_team_color=my_team_color,
                                       opponent_color=opponent_color)

    # Extract per-track colors for visualization
    track_colors = {}
    for tid in df["track_id"].unique():
        track = df[df["track_id"] == tid]
        track = track.copy()
        track["_area"] = (track["bbox_x2"] - track["bbox_x1"]) * (track["bbox_y2"] - track["bbox_y1"])
        best = track.sort_values("_area", ascending=False).head(5)
        colors = []
        for _, row in best.iterrows():
            fn = int(row["frame"])
            if fn not in frames:
                continue
            frame = frames[fn]
            bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
            c = tc.extract_jersey_color(frame, bbox)
            if c.sum() > 0:
                colors.append(c)
        if colors:
            track_colors[tid] = np.mean(colors, axis=0)

    team_map = {}
    for tid in df["track_id"].unique():
        rows = df_with_teams[df_with_teams["track_id"] == tid]
        if not rows.empty:
            team_map[tid] = rows.iloc[0].get("team", "Unknown")

    return team_map, track_colors, tc.team_colors


def draw_detection_heatmap(df, frames, outdir):
    """Save a heatmap showing where detections cluster on the frame."""
    print(f"  Generating detection heatmap...")
    if df.empty:
        print("  No detections — skipping heatmap.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get a reference frame for the background
    ref_frame_num = sorted(frames.keys())[len(frames) // 2]
    ref_frame = frames[ref_frame_num]
    fh, fw = ref_frame.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: detection centers on a sample frame
    ax = axes[0]
    ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    # Downscale for display
    display_w = 960
    scale = display_w / fw
    display_h = int(fh * scale)
    ref_small = cv2.resize(ref_rgb, (display_w, display_h))
    ax.imshow(ref_small)

    # Plot all detection centers
    all_x = df["x_px"].values * scale
    all_y = df["y_px"].values * scale
    ax.scatter(all_x, all_y, s=15, c="red", alpha=0.3, edgecolors="none")
    ax.set_title(f"All detection centers ({len(df)} detections)")
    ax.set_xlim(0, display_w)
    ax.set_ylim(display_h, 0)

    # Right: bbox size distribution
    ax = axes[1]
    bw = (df["bbox_x2"] - df["bbox_x1"]).values
    bh = (df["bbox_y2"] - df["bbox_y1"]).values
    areas = bw * bh
    ax.hist2d(bw, bh, bins=30, cmap="YlOrRd")
    ax.set_xlabel("Bbox width (px)")
    ax.set_ylabel("Bbox height (px)")
    ax.set_title(f"Bbox size distribution (median {np.median(bw):.0f}x{np.median(bh):.0f})")
    ax.axvline(x=20, color="white", linestyle="--", alpha=0.5, label="20px")
    ax.axhline(y=40, color="white", linestyle="--", alpha=0.5, label="40px")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(outdir, "07_detection_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved detection heatmap to {path}")


# ---------------------------------------------------------------------------
# NEW STAGES: Homography, Movement Stats, Formation, Territory
# ---------------------------------------------------------------------------

def run_homography(df, video_meta):
    """Auto-calibrate homography and add x_field/y_field columns."""
    from pipeline.homography import FieldHomography
    fh = FieldHomography(DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M)
    fh.calibrate_auto(video_meta["frame_h"], video_meta["frame_w"], df)
    df = fh.transform_df(df)
    return df, fh


def compute_track_movement(df, fps):
    """Compute per-track distance, speed, sprints using field coordinates."""
    if df.empty or "x_field" not in df.columns:
        return {}

    track_stats = {}
    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        if len(group) < 2:
            continue

        dx = group["x_field"].diff().fillna(0).values
        dy = group["y_field"].diff().fillna(0).values
        dists = np.sqrt(dx**2 + dy**2)

        # Use time_s if available, otherwise derive dt from frame gaps + fps
        if "time_s" in group.columns:
            dt = group["time_s"].diff().fillna(1.0 / max(fps, 1.0)).values
        else:
            dt = group["frame"].diff().fillna(1.0).values / max(fps, 1.0)
        dt = np.clip(dt, 0.01, None)  # avoid division by zero

        # Cap per-sample displacement based on max speed and dt (matches StatsCalculator)
        max_displacement = MAX_SPEED_CAP * dt
        dists = np.clip(dists, 0, max_displacement)

        speeds = dists / dt
        speeds = np.clip(speeds, 0, MAX_SPEED_CAP)

        # Rolling median filter (window=5) to suppress tracking jitter
        speeds_s = pd.Series(speeds)
        speeds_smooth = speeds_s.rolling(5, min_periods=1, center=True).median().values
        # Recompute distances consistent with smoothed speeds
        dists_smooth = speeds_smooth * dt

        total_dist = float(dists_smooth.sum())
        top_speed = float(np.percentile(speeds_smooth[speeds_smooth > 0], 95)) if (speeds_smooth > 0).any() else 0.0
        avg_speed = float(speeds_smooth.mean())
        sprint_count = int((speeds_smooth > SPRINT_THRESHOLD).sum())
        sprint_dist = float(dists_smooth[speeds_smooth > SPRINT_THRESHOLD].sum())

        # Zone percentages (thirds of field)
        third = DEFAULT_FIELD_LENGTH_M / 3.0
        att_pct = float((group["x_field"] > 2 * third).mean() * 100)
        mid_pct = float(((group["x_field"] >= third) & (group["x_field"] <= 2 * third)).mean() * 100)
        def_pct = float((group["x_field"] < third).mean() * 100)

        track_stats[int(tid)] = {
            "distance_m": round(total_dist, 1),
            "top_speed_ms": round(top_speed, 2),
            "avg_speed_ms": round(avg_speed, 2),
            "sprint_count": sprint_count,
            "sprint_dist_m": round(sprint_dist, 1),
            "att_pct": round(att_pct, 1),
            "mid_pct": round(mid_pct, 1),
            "def_pct": round(def_pct, 1),
            "n_detections": len(group),
            "avg_x": round(float(group["x_field"].mean()), 1),
            "avg_y": round(float(group["y_field"].mean()), 1),
        }

    return track_stats


def draw_field_positions(df, team_map, team_colors_bgr, outdir):
    """Plot all tracks' average positions on a pitch diagram by team."""
    print(f"  Generating field position plot...")
    if df.empty or "x_field" not in df.columns:
        print("  No field coordinates — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from mplsoccer import Pitch
        has_mplsoccer = True
    except ImportError:
        has_mplsoccer = False

    fig, ax = plt.subplots(figsize=(12, 8))

    if has_mplsoccer:
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=DEFAULT_FIELD_LENGTH_M,
            pitch_width=DEFAULT_FIELD_WIDTH_M,
            pitch_color=PITCH_GRASS,
            line_color=PITCH_LINES,
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.patch.set_facecolor(PITCH_GRASS)
    else:
        ax.set_facecolor("#1a2a1a")
        ax.set_xlim(0, DEFAULT_FIELD_LENGTH_M)
        ax.set_ylim(0, DEFAULT_FIELD_WIDTH_M)
        ax.set_aspect("equal")
        fig.patch.set_facecolor("#1a2a1a")

    # Group by track, get average position
    for tid, group in df.groupby("track_id"):
        avg_x = group["x_field"].mean()
        avg_y = group["y_field"].mean()
        team = team_map.get(int(tid), "Unknown")

        if team == "Home":
            c = TEAM_A_BLUE
            marker = "o"
        elif team == "Away":
            c = TEAM_B_RED
            marker = "s"
        else:
            c = "gray"
            marker = "x"

        size = min(200, max(30, len(group) * 15))
        ax.scatter([avg_x], [avg_y], s=size, c=c, edgecolors="white",
                   linewidth=1, zorder=5, alpha=0.8, marker=marker)
        ax.text(avg_x, avg_y - 1.0, f"T{tid}", color="white", fontsize=6,
                ha="center", va="top", zorder=6)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=TEAM_A_BLUE, edgecolor="white", label="Home"),
        Patch(facecolor=TEAM_B_RED, edgecolor="white", label="Away"),
        Patch(facecolor="gray", edgecolor="white", label="Unknown"),
    ]
    ax.legend(handles=legend_elems, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)

    ax.set_title("Average Track Positions (field coordinates)",
                 color="white", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, "08_field_positions.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved field positions to {path}")


def draw_movement_stats(track_stats, team_map, outdir):
    """Bar charts of distance, speed, sprints per track (top N by distance)."""
    print(f"  Generating movement stats plots...")
    if not track_stats:
        print("  No movement stats — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort tracks by distance, take top 20 with at least 3 detections
    filtered = {k: v for k, v in track_stats.items() if v["n_detections"] >= 3}
    if not filtered:
        print("  No tracks with >= 3 detections — skipping.")
        return

    sorted_tids = sorted(filtered.keys(), key=lambda t: filtered[t]["distance_m"], reverse=True)[:20]

    labels = [f"T{t}" for t in sorted_tids]
    distances = [filtered[t]["distance_m"] for t in sorted_tids]
    top_speeds = [filtered[t]["top_speed_ms"] for t in sorted_tids]
    sprint_counts = [filtered[t]["sprint_count"] for t in sorted_tids]
    colors = [TEAM_A_BLUE if team_map.get(t) == "Home"
              else TEAM_B_RED if team_map.get(t) == "Away"
              else "gray" for t in sorted_tids]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#161b22")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # Distance
    ax = axes[0]
    ax.barh(range(len(labels)), distances, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color="white")
    ax.set_xlabel("Distance (m)", color="white")
    ax.set_title("Total Distance per Track", color="white", fontweight="bold")
    ax.invert_yaxis()

    # Top speed
    ax = axes[1]
    ax.barh(range(len(labels)), top_speeds, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color="white")
    ax.set_xlabel("Top Speed (m/s)", color="white")
    ax.set_title("Top Speed per Track (95th pct)", color="white", fontweight="bold")
    ax.axvline(x=SPRINT_THRESHOLD, color="yellow", linestyle="--", alpha=0.5, label=f"Sprint ({SPRINT_THRESHOLD} m/s)")
    ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.invert_yaxis()

    # Sprint count
    ax = axes[2]
    ax.barh(range(len(labels)), sprint_counts, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color="white")
    ax.set_xlabel("Sprint Count", color="white")
    ax.set_title("Sprint Detections per Track", color="white", fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(outdir, "09_movement_stats.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved movement stats to {path}")


def draw_formation_snapshot(df, team_map, outdir):
    """Detect and plot formation for each team using average track positions."""
    print(f"  Generating formation snapshot...")
    if df.empty or "x_field" not in df.columns:
        print("  No field coordinates — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pipeline.formation import FormationDetector

    fd = FormationDetector(players_per_team=DEFAULT_PLAYERS_PER_TEAM)

    try:
        from mplsoccer import Pitch
        has_mplsoccer = True
    except ImportError:
        has_mplsoccer = False

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax_idx, team in enumerate(["Home", "Away"]):
        ax = axes[ax_idx]
        team_tids = [t for t, tm in team_map.items() if tm == team]
        if not team_tids:
            ax.text(0.5, 0.5, f"{team}: No tracks", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.set_facecolor(PITCH_GRASS)
            continue

        # Get average position per track (only tracks with >= 3 detections)
        positions = []
        tid_labels = []
        for tid in team_tids:
            grp = df[df["track_id"] == tid]
            if len(grp) >= 3:
                positions.append([grp["x_field"].mean(), grp["y_field"].mean()])
                tid_labels.append(f"T{tid}")

        if len(positions) < 4:
            ax.text(0.5, 0.5, f"{team}: Too few tracks ({len(positions)})",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            ax.set_facecolor(PITCH_GRASS)
            continue

        positions_arr = np.array(positions)
        formation, confidence = fd.detect_formation(positions_arr)

        color = TEAM_A_BLUE if team == "Home" else TEAM_B_RED

        if has_mplsoccer:
            pitch = Pitch(
                pitch_type="custom",
                pitch_length=DEFAULT_FIELD_LENGTH_M,
                pitch_width=DEFAULT_FIELD_WIDTH_M,
                pitch_color=PITCH_GRASS,
                line_color=PITCH_LINES,
                linewidth=1,
            )
            pitch.draw(ax=ax)
        else:
            ax.set_facecolor(PITCH_GRASS)
            ax.set_xlim(0, DEFAULT_FIELD_LENGTH_M)
            ax.set_ylim(0, DEFAULT_FIELD_WIDTH_M)
            ax.set_aspect("equal")

        ax.scatter(positions_arr[:, 0], positions_arr[:, 1],
                   s=300, c=color, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)

        for i, (x, y) in enumerate(positions):
            ax.text(x, y - 1.3, tid_labels[i], color="white", fontsize=7,
                    ha="center", va="top", fontweight="bold", zorder=6)

        conf_str = f" ({confidence:.0%})" if confidence > 0 else ""
        ax.set_title(f"{team}: {formation}{conf_str} ({len(positions)} players)",
                     color="white", fontsize=14, fontweight="bold")

    fig.patch.set_facecolor(PITCH_GRASS)
    plt.tight_layout()
    path = os.path.join(outdir, "10_formation.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved formation snapshot to {path}")


def draw_team_territory(df, team_map, outdir):
    """Visualize team territorial coverage and compactness."""
    print(f"  Generating team territory plot...")
    if df.empty or "x_field" not in df.columns:
        print("  No field coordinates — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(PITCH_GRASS)

    try:
        from mplsoccer import Pitch
        has_mplsoccer = True
    except ImportError:
        has_mplsoccer = False

    # Left: convex hull territory per team
    ax = axes[0]
    if has_mplsoccer:
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=DEFAULT_FIELD_LENGTH_M,
            pitch_width=DEFAULT_FIELD_WIDTH_M,
            pitch_color=PITCH_GRASS,
            line_color=PITCH_LINES,
            linewidth=1,
        )
        pitch.draw(ax=ax)
    else:
        ax.set_facecolor(PITCH_GRASS)
        ax.set_xlim(0, DEFAULT_FIELD_LENGTH_M)
        ax.set_ylim(0, DEFAULT_FIELD_WIDTH_M)
        ax.set_aspect("equal")

    territory_info = {}
    for team, color, alpha in [("Home", TEAM_A_BLUE, 0.25), ("Away", TEAM_B_RED, 0.25)]:
        team_tids = [t for t, tm in team_map.items() if tm == team]
        positions = []
        for tid in team_tids:
            grp = df[df["track_id"] == tid]
            if len(grp) >= 3:
                positions.append([grp["x_field"].mean(), grp["y_field"].mean()])

        if len(positions) >= 3:
            pts = np.array(positions)
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=alpha, zorder=2)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=color, linewidth=2,
                        alpha=0.7, zorder=3)
                territory_info[team] = {
                    "area_m2": round(float(hull.volume), 1),  # 2D: volume = area
                    "n_tracks": len(positions),
                }
            except Exception:
                territory_info[team] = {"area_m2": 0, "n_tracks": len(positions)}

            ax.scatter(pts[:, 0], pts[:, 1], s=80, c=color, edgecolors="white",
                       linewidth=1, zorder=5, alpha=0.8)

    ax.set_title("Team Territory (Convex Hull)", color="white", fontsize=14, fontweight="bold")

    # Right: zone breakdown bar chart (thirds)
    ax = axes[1]
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="white")

    zones = ["Defensive", "Midfield", "Attacking"]
    third = DEFAULT_FIELD_LENGTH_M / 3.0

    home_zones = [0, 0, 0]
    away_zones = [0, 0, 0]
    for tid, tm in team_map.items():
        grp = df[df["track_id"] == tid]
        if len(grp) < 3:
            continue
        avg_x = grp["x_field"].mean()
        zone_idx = min(2, int(avg_x / third))
        if tm == "Home":
            home_zones[zone_idx] += 1
        elif tm == "Away":
            away_zones[zone_idx] += 1

    x_pos = np.arange(len(zones))
    width = 0.35
    ax.bar(x_pos - width/2, home_zones, width, label="Home", color=TEAM_A_BLUE, alpha=0.85)
    ax.bar(x_pos + width/2, away_zones, width, label="Away", color=TEAM_B_RED, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(zones, color="white")
    ax.set_ylabel("Number of Tracks", color="white")
    ax.set_title("Track Distribution by Zone", color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    # Annotate territory areas
    info_lines = []
    for team, info in territory_info.items():
        info_lines.append(f"{team}: {info['area_m2']}m² ({info['n_tracks']} tracks)")
    if info_lines:
        ax.text(0.02, 0.95, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=9, color="white", va="top",
                bbox=dict(boxstyle="round", facecolor="#161b22", edgecolor="#30363d", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(outdir, "11_team_territory.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved team territory to {path}")


# ---------------------------------------------------------------------------
# STAGES 12-15: Player ID, Stats, Passes
# ---------------------------------------------------------------------------

def run_player_identification(df, frames):
    """Run the full fingerprint pipeline: face + jersey OCR + gait + cleat + height.
    Uses roster-size-based track selection (matches app pipeline).
    Returns (df_with_ids, assignments) or (df_unchanged, None) on failure."""
    try:
        from database.roster_db import RosterDB
        from pipeline.face_reid import FaceReID
        from pipeline.gait import GaitAnalyzer
        from pipeline.cleat import CleatExtractor
        from pipeline.jersey_ocr import JerseyOCR
        from pipeline.fingerprint import PlayerFingerprinter

        db = RosterDB()
        players = db.get_all_players()
        if not players:
            print("  No roster players found in DB — skipping player ID.")
            return df, None

        roster_size = len(players)
        face = FaceReID()
        print("  Building roster face embeddings...")
        face.build_roster_embeddings(players, db)
        n_embeddings = len(face.roster_embeddings)
        print(f"  {n_embeddings}/{len(players)} face embeddings ready")

        gait = GaitAnalyzer()
        cleat = CleatExtractor()
        jersey_ocr = JerseyOCR()
        fp = PlayerFingerprinter(db, face, gait, cleat, jersey_ocr=jersey_ocr)

        # Roster-size-based track selection (matches app)
        track_counts = df.groupby("track_id").size().reset_index(name="count")
        id_per_team = roster_size + 4
        id_keep_ids = set(
            track_counts.nlargest(id_per_team, "count")["track_id"].tolist()
        )
        # Broad: all tracks with ≥3 detections for stats
        stats_keep_ids = set(
            track_counts[track_counts["count"] >= 3]["track_id"].tolist()
        )
        all_keep_ids = id_keep_ids | stats_keep_ids
        noise_tids = set(df["track_id"].unique()) - all_keep_ids

        df_for_id = df[df["track_id"].isin(id_keep_ids)].copy()
        df = df[df["track_id"].isin(all_keep_ids)].copy()

        print(f"  Narrow (face ID): {len(id_keep_ids)} tracks | "
              f"Broad (stats): {len(all_keep_ids)} tracks | "
              f"Dropped: {len(noise_tids)} noise tracks")
        print(f"  Running multi-modal identification on {len(id_keep_ids)} tracks...")

        assignments, pending = fp.identify_all_tracks(df_for_id, frames)

        # Mark remaining broad tracks as unknown/Sub
        for tid in (all_keep_ids - id_keep_ids):
            assignments[tid] = {"player_id": None, "confidence": 0.0, "status": "unknown"}

        auto_count = sum(1 for a in assignments.values() if a["status"] == "auto_assigned")
        conf_count = sum(1 for a in assignments.values() if a["status"] == "needs_confirmation")
        unk_count = sum(1 for a in assignments.values() if a["status"] == "unknown")
        print(f"  Results: auto={auto_count}, needs_confirm={conf_count}, unknown={unk_count}")

        df = fp.merge_track_ids(assignments, df)
        return df, assignments

    except ImportError as e:
        print(f"  Skipping player ID — missing dependency: {e}")
        return df, None
    except Exception as e:
        print(f"  Player ID failed: {e}")
        import traceback
        traceback.print_exc()
        return df, None


def run_pass_detection(df, video_meta):
    """Run pass detection on all my-team tracks (matches app — no pre-filter)."""
    if df.empty or "x_field" not in df.columns:
        return []

    try:
        from pipeline.passes import PassDetector
        pd_det = PassDetector()
        passes = pd_det.detect_passes(
            df, video_meta["fps"],
            DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M
        )
        return passes
    except Exception as e:
        print(f"  Pass detection failed: {e}")
        return []


def run_player_stats(df, passes, video_meta):
    """Compute per-player stats. Returns stats DataFrame."""
    if "player_name" not in df.columns:
        return pd.DataFrame()

    try:
        from pipeline.stats import StatsCalculator
        sc = StatsCalculator(DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M,
                             video_meta["fps"])
        stats_df = sc.calculate_all_stats(df, passes)
        return stats_df
    except Exception as e:
        print(f"  Stats calculation failed: {e}")
        return pd.DataFrame()


def draw_player_id_results(df, assignments, outdir):
    """Visualize player identification results: who matched, confidence breakdown."""
    print(f"  Generating player ID results plot...")
    if assignments is None:
        print("  No assignments — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only show tracks with at least some confidence
    identified = {tid: a for tid, a in assignments.items()
                  if a["confidence"] > 0}
    all_tracks = assignments

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("#161b22")

    # Left: confidence distribution by status
    ax = axes[0]
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="white")

    statuses = {"auto_assigned": [], "needs_confirmation": [], "unknown": []}
    for a in all_tracks.values():
        statuses[a["status"]].append(a["confidence"])

    status_colors = {"auto_assigned": "#4caf50", "needs_confirmation": "#ff9800", "unknown": "#f44336"}
    status_labels = {"auto_assigned": f"Auto ({len(statuses['auto_assigned'])})",
                     "needs_confirmation": f"Needs Confirm ({len(statuses['needs_confirmation'])})",
                     "unknown": f"Unknown ({len(statuses['unknown'])})"}

    bar_data = []
    bar_colors = []
    bar_labels = []
    for status, confs in statuses.items():
        if confs:
            bar_data.append(np.mean(confs))
            bar_colors.append(status_colors[status])
            bar_labels.append(status_labels[status])

    if bar_data:
        bars = ax.bar(range(len(bar_data)), bar_data, color=bar_colors, alpha=0.85)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, color="white", fontsize=9)
        ax.set_ylabel("Avg Confidence", color="white")
        ax.set_title("Player ID Status Distribution", color="white", fontweight="bold")
        ax.set_ylim(0, 1.0)

        # Add count labels
        for i, b in enumerate(bars):
            count = len(statuses[list(statuses.keys())[i]])
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{count}", ha="center", color="white", fontsize=11, fontweight="bold")

    # Right: per-track identification table (top matches)
    ax = axes[1]
    ax.set_facecolor("#161b22")
    ax.axis("off")

    # Build table data: track → player name, confidence, status
    table_rows = []
    if "player_name" in df.columns:
        for tid in sorted(all_tracks.keys()):
            a = all_tracks[tid]
            rows = df[df["track_id"] == tid]
            if rows.empty:
                continue
            name = rows.iloc[0].get("player_name", "?")
            jersey = rows.iloc[0].get("jersey_number", "?")
            conf = a["confidence"]
            status = a["status"].replace("_", " ").title()
            n_dets = len(rows)
            if n_dets >= 2:  # Only show tracks with some data
                table_rows.append([f"T{tid}", name, str(jersey), f"{conf:.2f}", status, str(n_dets)])

    if table_rows:
        # Sort by confidence descending, show top 25
        table_rows.sort(key=lambda r: float(r[3]), reverse=True)
        table_rows = table_rows[:25]

        table = ax.table(
            cellText=table_rows,
            colLabels=["Track", "Player", "#", "Conf", "Status", "Dets"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("#161b22")
            cell.set_edgecolor("#30363d")
            cell.set_text_props(color="white")
            if row == 0:
                cell.set_facecolor("#1f2937")
                cell.set_text_props(color="white", fontweight="bold")

        ax.set_title("Top Track Identifications", color="white",
                     fontsize=14, fontweight="bold", pad=20)
    else:
        ax.text(0.5, 0.5, "No identified tracks", ha="center", va="center",
                color="white", fontsize=14)

    plt.tight_layout()
    path = os.path.join(outdir, "12_player_id.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved player ID results to {path}")


def draw_player_stats_table(stats_df, outdir):
    """Render player stats as a visual table."""
    print(f"  Generating player stats table...")
    if stats_df.empty:
        print("  No player stats — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, max(4, len(stats_df) * 0.5 + 2)))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")
    ax.axis("off")

    # Build table columns
    cols = ["#", "Name", "Team", "Min", "Dist(m)", "Top Spd", "Sprints",
            "Att%", "Mid%", "Def%", "Passes", "Conf"]
    rows = []
    for _, row in stats_df.iterrows():
        rows.append([
            str(int(row.get("jersey_number", 0))),
            str(row.get("name", "?"))[:15],
            str(row.get("team", "?")),
            f"{row.get('minutes_played', 0):.1f}",
            f"{row.get('distance_m', 0):.0f}",
            f"{row.get('top_speed_ms', 0):.1f}",
            str(int(row.get("sprint_count", 0))),
            f"{row.get('pct_att_third', 0):.0f}",
            f"{row.get('pct_mid_third', 0):.0f}",
            f"{row.get('pct_def_third', 0):.0f}",
            str(int(row.get("passes_made", 0))),
            f"{row.get('id_confidence', 0):.2f}",
        ])

    if rows:
        table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("#161b22")
            cell.set_edgecolor("#30363d")
            cell.set_text_props(color="white")
            if r == 0:
                cell.set_facecolor("#1f2937")
                cell.set_text_props(color="white", fontweight="bold")

        ax.set_title("Per-Player Statistics", color="white",
                     fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(outdir, "13_player_stats.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved player stats to {path}")


def draw_pass_map(passes, df, outdir):
    """Draw detected passes on a pitch diagram."""
    print(f"  Generating pass map...")
    if not passes:
        print("  No passes detected — skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from mplsoccer import Pitch
        has_mplsoccer = True
    except ImportError:
        has_mplsoccer = False

    fig, ax = plt.subplots(figsize=(12, 8))

    if has_mplsoccer:
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=DEFAULT_FIELD_LENGTH_M,
            pitch_width=DEFAULT_FIELD_WIDTH_M,
            pitch_color=PITCH_GRASS,
            line_color=PITCH_LINES,
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.patch.set_facecolor(PITCH_GRASS)
    else:
        ax.set_facecolor(PITCH_GRASS)
        ax.set_xlim(0, DEFAULT_FIELD_LENGTH_M)
        ax.set_ylim(0, DEFAULT_FIELD_WIDTH_M)
        ax.set_aspect("equal")
        fig.patch.set_facecolor(PITCH_GRASS)

    # Draw pass arrows
    for p in passes:
        x1, y1 = p["passer_pos"]
        x2, y2 = p["receiver_pos"]
        team = p.get("team", "Home")
        color = TEAM_A_BLUE if team == "Home" else TEAM_B_RED

        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.5, alpha=0.6))

        # Small dot at passer position
        ax.scatter([x1], [y1], s=30, c=color, edgecolors="white",
                   linewidth=0.5, zorder=5, alpha=0.7)

    home_passes = sum(1 for p in passes if p.get("team") == "Home")
    away_passes = len(passes) - home_passes

    ax.set_title(f"Detected Passes — Home: {home_passes}, Away: {away_passes} "
                 f"(total: {len(passes)})",
                 color="white", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, "14_pass_map.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved pass map to {path}")


def write_summary(df, video_meta, track_colors, team_map, team_centers, elapsed, outdir,
                  track_stats=None, assignments=None, player_stats_df=None, passes=None):
    """Write a text summary report."""
    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE DIAGNOSTIC SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Video: {video_meta.get('frame_w')}x{video_meta.get('frame_h')} "
                 f"@ {video_meta.get('fps'):.1f}fps")
    lines.append(f"Processing time: {elapsed:.1f}s")
    lines.append("")

    if df.empty:
        lines.append("NO DETECTIONS FOUND")
    else:
        n_frames = df["frame"].nunique()
        n_tracks = df["track_id"].nunique()
        n_dets = len(df)
        lines.append(f"Frames processed: {n_frames}")
        lines.append(f"Total detections: {n_dets}")
        lines.append(f"Unique tracks: {n_tracks}")
        lines.append(f"Avg detections/frame: {n_dets/n_frames:.1f}")
        lines.append(f"Avg track length: {n_dets/n_tracks:.1f} detections")
        lines.append("")

        # Track length distribution
        track_lengths = df.groupby("track_id").size()
        lines.append("Track length distribution:")
        lines.append(f"  Min: {track_lengths.min()}")
        lines.append(f"  Median: {track_lengths.median():.0f}")
        lines.append(f"  Mean: {track_lengths.mean():.1f}")
        lines.append(f"  Max: {track_lengths.max()}")
        lines.append(f"  Tracks with 1 detection: {(track_lengths == 1).sum()}")
        lines.append(f"  Tracks with >5 detections: {(track_lengths > 5).sum()}")
        lines.append("")

        # Confidence stats
        lines.append(f"Detection confidence:")
        lines.append(f"  Mean: {df['conf'].mean():.3f}")
        lines.append(f"  Median: {df['conf'].median():.3f}")
        lines.append(f"  Min: {df['conf'].min():.3f}")
        lines.append(f"  Max: {df['conf'].max():.3f}")
        lines.append("")

        # Bbox size stats
        df_copy = df.copy()
        df_copy["bw"] = df_copy["bbox_x2"] - df_copy["bbox_x1"]
        df_copy["bh"] = df_copy["bbox_y2"] - df_copy["bbox_y1"]
        lines.append(f"Bbox width: min={df_copy['bw'].min():.0f}, "
                     f"median={df_copy['bw'].median():.0f}, "
                     f"max={df_copy['bw'].max():.0f}")
        lines.append(f"Bbox height: min={df_copy['bh'].min():.0f}, "
                     f"median={df_copy['bh'].median():.0f}, "
                     f"max={df_copy['bh'].max():.0f}")
        lines.append("")

        # Team assignment
        if team_map:
            home_count = sum(1 for t in team_map.values() if t == "Home")
            away_count = sum(1 for t in team_map.values() if t == "Away")
            unk_count = sum(1 for t in team_map.values() if t == "Unknown")
            lines.append(f"Team assignment: Home={home_count}, Away={away_count}, Unknown={unk_count}")
            if team_centers:
                for team, bgr in team_centers.items():
                    lines.append(f"  {team} color (BGR): ({bgr[0]:.0f}, {bgr[1]:.0f}, {bgr[2]:.0f}) "
                                 f"= RGB({bgr[2]:.0f}, {bgr[1]:.0f}, {bgr[0]:.0f})")
            lines.append("")

        # Per-track detail
        # Movement summary (if available)
        if track_stats:
            active_tracks = {k: v for k, v in track_stats.items() if v["n_detections"] >= 3}
            if active_tracks:
                all_dists = [v["distance_m"] for v in active_tracks.values()]
                all_speeds = [v["top_speed_ms"] for v in active_tracks.values()]
                all_sprints = [v["sprint_count"] for v in active_tracks.values()]
                lines.append(f"Movement stats ({len(active_tracks)} active tracks, >=3 dets):")
                lines.append(f"  Total distance: {sum(all_dists):.0f}m across all tracks")
                lines.append(f"  Avg distance/track: {np.mean(all_dists):.1f}m")
                lines.append(f"  Max top speed: {max(all_speeds):.2f} m/s")
                lines.append(f"  Total sprints: {sum(all_sprints)}")
                lines.append(f"  Sprint threshold: {SPRINT_THRESHOLD} m/s")
                lines.append("")

        lines.append("Per-track details:")
        has_movement = track_stats and any(track_stats.values())
        if has_movement:
            lines.append(f"{'Track':>6} {'Dets':>5} {'Team':>6} {'Color (RGB)':>20} "
                         f"{'Avg Conf':>9} {'Dist(m)':>8} {'TopSpd':>7} {'Sprints':>8}")
            lines.append("-" * 85)
        else:
            lines.append(f"{'Track':>6} {'Dets':>5} {'Team':>6} {'Color (RGB)':>20} "
                         f"{'Avg Conf':>9} {'Avg BW':>7} {'Avg BH':>7}")
            lines.append("-" * 70)

        for tid in sorted(df["track_id"].unique()):
            t = df[df["track_id"] == tid]
            team = team_map.get(tid, "?")
            color = track_colors.get(tid)
            color_str = (f"({color[2]:.0f},{color[1]:.0f},{color[0]:.0f})"
                         if color is not None else "N/A")
            avg_conf = t["conf"].mean()

            if has_movement and tid in track_stats:
                ms = track_stats[tid]
                lines.append(f"T{tid:>5} {len(t):>5} {team:>6} {color_str:>20} "
                             f"{avg_conf:>9.3f} {ms['distance_m']:>8.1f} "
                             f"{ms['top_speed_ms']:>7.2f} {ms['sprint_count']:>8}")
            else:
                avg_bw = (t["bbox_x2"] - t["bbox_x1"]).mean()
                avg_bh = (t["bbox_y2"] - t["bbox_y1"]).mean()
                lines.append(f"T{tid:>5} {len(t):>5} {team:>6} {color_str:>20} "
                             f"{avg_conf:>9.3f} {avg_bw:>7.0f} {avg_bh:>7.0f}")

        # Player identification summary
        if assignments:
            lines.append("")
            auto_count = sum(1 for a in assignments.values() if a["status"] == "auto_assigned")
            conf_count = sum(1 for a in assignments.values() if a["status"] == "needs_confirmation")
            unk_count = sum(1 for a in assignments.values() if a["status"] == "unknown")
            lines.append(f"Player identification:")
            lines.append(f"  Auto-assigned: {auto_count}")
            lines.append(f"  Needs confirmation: {conf_count}")
            lines.append(f"  Unknown: {unk_count}")
            if "player_name" in df.columns:
                identified = df[~df["player_name"].str.startswith("Unknown_")]
                unique_players = identified["player_name"].nunique()
                lines.append(f"  Unique players identified: {unique_players}")
                if unique_players > 0:
                    for name in sorted(identified["player_name"].unique()):
                        n = len(identified[identified["player_name"] == name])
                        conf = identified[identified["player_name"] == name]["id_confidence"].mean()
                        lines.append(f"    {name}: {n} detections, conf={conf:.2f}")

        # Pass detection summary
        if passes:
            lines.append("")
            lines.append(f"Passes detected: {len(passes)}")
            home_passes = [p for p in passes if p.get("team") == "Home"]
            away_passes = [p for p in passes if p.get("team") == "Away"]
            lines.append(f"  Home: {len(home_passes)}, Away: {len(away_passes)}")
            if passes:
                avg_dist = np.mean([p["pass_distance_m"] for p in passes])
                lines.append(f"  Avg pass distance: {avg_dist:.1f}m")

        # Player stats summary
        if player_stats_df is not None and not player_stats_df.empty:
            lines.append("")
            lines.append(f"Player stats ({len(player_stats_df)} players):")
            for _, row in player_stats_df.iterrows():
                lines.append(f"  #{int(row.get('jersey_number', 0)):>2} {row.get('name', '?'):>15}: "
                             f"dist={row.get('distance_m', 0):.0f}m, "
                             f"top_spd={row.get('top_speed_ms', 0):.1f}m/s, "
                             f"sprints={int(row.get('sprint_count', 0))}, "
                             f"passes={int(row.get('passes_made', 0))}")

    report = "\n".join(lines)
    path = os.path.join(outdir, "summary.txt")
    with open(path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n  Summary saved to {path}")
    return report


def main():
    args = parse_args()
    outdir = args.outdir
    dirs = setup_dirs(outdir)

    print(f"\n{'='*60}")
    print(f"SOCCER PIPELINE DIAGNOSTIC")
    print(f"{'='*60}")

    # Step 1: Detection + Tracking
    print(f"\n[1/13] Running YOLO detection + {args.tracker}...")
    t0 = time.time()
    df, frames, video_meta = run_detection(
        args.video, args.start, args.frames,
        args.sample_rate, args.confidence, args.model,
        tracker_type=args.tracker, yolo_version=args.yolo_version
    )
    t_detect = time.time() - t0
    print(f"  Detection complete: {len(df)} detections, "
          f"{df['track_id'].nunique() if not df.empty else 0} tracks in {t_detect:.1f}s")

    # Step 2: Draw detection frames
    print(f"\n[2/13] Drawing detection annotations...")
    draw_detections(df, frames, dirs["01_detections"])

    # Step 3: Torso crop diagnostics
    print(f"\n[3/13] Drawing torso crop diagnostics...")
    draw_torso_crops(df, frames, dirs["02_torso_crops"])

    # Step 3.5: Homography calibrate → field zone filter → transform (matches app order)
    from pipeline.homography import FieldHomography
    homography = FieldHomography(DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M)
    homography.calibrate_auto(video_meta["frame_h"], video_meta["frame_w"], df)
    pre_count = len(df)
    df = homography.filter_to_field_zone(df)
    post_count = len(df)
    if pre_count > post_count:
        print(f"  Field zone filter: {pre_count:,} → {post_count:,} detections "
              f"(removed {pre_count - post_count:,} off-field)")
    df = homography.transform_df(df)
    val_score, val_msg = homography.validate(df)
    if val_score < 0.8:
        print(f"  ⚠️  Homography validation: {val_msg}")
    print(f"  Homography calibrated — field: {DEFAULT_FIELD_LENGTH_M}x{DEFAULT_FIELD_WIDTH_M}m, "
          f"validation={val_score:.2f}")

    # Step 4: Team classification
    print(f"\n[4/13] Running team classification...")
    if args.my_team:
        print(f"  My team jersey: {args.my_team}")
    t1 = time.time()
    team_map, track_colors, team_centers = run_team_classification(df, frames, my_team_color=args.my_team, opponent_color=args.opponent)
    t_classify = time.time() - t1
    print(f"  Classification complete in {t_classify:.1f}s")

    # Add team column to df for downstream stages (pass detection needs it)
    if not df.empty and team_map:
        df["team"] = df["track_id"].map(team_map).fillna("Unknown")

    draw_team_assignment(df, frames, team_centers, team_map, dirs["04_team_assignment"])
    draw_track_timeline(df, outdir)
    draw_color_clusters(track_colors, team_map, team_centers, outdir)

    # === MY-TEAM-ONLY FILTER (matches app pipeline) ===
    if not df.empty and "team" in df.columns:
        home_det_count = (df["team"] == "Home").sum()
        away_det_count = (df["team"] == "Away").sum()
        total_det = home_det_count + away_det_count
        possession_pct = (home_det_count / total_det * 100) if total_det > 0 else 50.0
        n_tracks_before = df["track_id"].nunique()
        df = df[df["team"] == "Home"].copy()
        n_tracks_after = df["track_id"].nunique()
        # Remove opponent frames from frames dict (keep only frames with my-team detections)
        my_frames = set(df["frame"].unique())
        frames = {k: v for k, v in frames.items() if k in my_frames}
        print(f"  *** FILTERED TO MY TEAM: {n_tracks_after} tracks "
              f"(dropped {n_tracks_before - n_tracks_after} opponent), "
              f"possession est. {possession_pct:.0f}% ***")

    # Step 5: Detection heatmap
    print(f"\n[5/13] Drawing detection heatmap...")
    draw_detection_heatmap(df, frames, outdir)

    # Step 6: Movement stats (homography already applied in Step 3.5)
    print(f"\n[6/13] Computing per-track movement stats...")
    track_stats = compute_track_movement(df, video_meta["fps"])
    active = {k: v for k, v in track_stats.items() if v["n_detections"] >= 3}
    print(f"  Computed stats for {len(track_stats)} tracks ({len(active)} with >=3 detections)")

    # Step 7: Field positions + movement charts
    print(f"\n[7/13] Drawing field positions and movement stats...")
    draw_field_positions(df, team_map, team_centers, outdir)
    draw_movement_stats(track_stats, team_map, outdir)

    # Step 8: Formation detection (snapshot only; timeline needs player_name from step 10)
    print(f"\n[8/13] Detecting team formations...")
    draw_formation_snapshot(df, team_map, outdir)

    # Step 9: Team territory
    print(f"\n[9/13] Drawing team territory...")
    draw_team_territory(df, team_map, outdir)

    # Step 10: Player identification (face + gait + cleat + height)
    print(f"\n[10/13] Running player identification...")
    t3 = time.time()
    df, assignments = run_player_identification(df, frames)
    t_id = time.time() - t3
    print(f"  Player ID complete in {t_id:.1f}s")

    # Auto-label unnamed tracks (matches app pipeline)
    if "player_name" in df.columns and assignments:
        unnamed_mask = df["player_name"].isna() | df["player_name"].str.startswith("Unknown_")
        if unnamed_mask.any():
            unnamed_tids = df.loc[unnamed_mask, "track_id"].unique()
            existing_named = df.loc[~unnamed_mask, "track_id"].nunique()
            unnamed_counts = df[unnamed_mask].groupby("track_id").size()
            slots = max(0, DEFAULT_PLAYERS_PER_TEAM - existing_named)
            top_unnamed = set(unnamed_counts.nlargest(slots).index.tolist())

            counter = existing_named
            for tid in unnamed_tids:
                if tid in top_unnamed:
                    counter += 1
                    name = f"Player {counter}"
                else:
                    name = "Sub"
                df.loc[df["track_id"] == tid, "player_name"] = name
            df["player_id"] = df["player_id"].fillna("")
            df["jersey_number"] = df["jersey_number"].fillna(0).astype(int)
            df["id_confidence"] = df["id_confidence"].fillna(0.0)
            print(f"  Auto-labeled: {len(top_unnamed)} as Player N, "
                  f"{len(unnamed_tids) - len(top_unnamed)} as Sub")

    draw_player_id_results(df, assignments, outdir)

    # Formation timeline + compactness (needs player_name from step 10)
    from pipeline.formation import FormationDetector
    formation_detector = FormationDetector(players_per_team=DEFAULT_PLAYERS_PER_TEAM)
    if "player_name" in df.columns and not df.empty:
        home_formation_timeline = formation_detector.formation_over_time(df, "Home")
        home_compactness = formation_detector.compactness_over_time(df, "Home")
        if home_formation_timeline:
            formations_str = ", ".join(set(t["formation"] for t in home_formation_timeline))
            print(f"  Formation timeline: {len(home_formation_timeline)} windows — {formations_str}")
        if home_compactness:
            avg_compact = np.mean([c["compactness"] for c in home_compactness])
            print(f"  Avg compactness: {avg_compact:.1f}m²")

    # Step 11: Pass detection
    print(f"\n[11/13] Detecting passes...")
    passes = run_pass_detection(df, video_meta)
    print(f"  {len(passes)} passes detected")

    # Build pass matrix (matches app pipeline)
    from pipeline.passes import PassDetector
    pass_detector = PassDetector()
    if "player_name" in df.columns and not df.empty:
        all_players = df["player_name"].dropna().unique().tolist()
        all_players = [p for p in all_players if p != "Sub"]
        home_pass_matrix = pass_detector.build_pass_matrix(passes, all_players)
        if not home_pass_matrix.empty:
            total_passes = int(home_pass_matrix.values.sum())
            print(f"  Pass matrix: {len(all_players)} players, {total_passes} total passes")
    else:
        home_pass_matrix = pd.DataFrame()

    draw_pass_map(passes, df, outdir)

    # Step 12: Player stats (uses StatsCalculator — matches app)
    print(f"\n[12/13] Computing player stats...")
    player_stats_df = run_player_stats(df, passes, video_meta)
    if not player_stats_df.empty:
        print(f"  Stats computed for {len(player_stats_df)} players")
    draw_player_stats_table(player_stats_df, outdir)

    # Step 13: Summary
    print(f"\n[13/13] Writing summary report...")
    total_elapsed = time.time() - t0
    write_summary(df, video_meta, track_colors, team_map, team_centers,
                  total_elapsed, outdir, track_stats=track_stats,
                  assignments=assignments, player_stats_df=player_stats_df,
                  passes=passes)

    print(f"\n{'='*60}")
    print(f"ALL DIAGNOSTICS SAVED TO: {os.path.abspath(outdir)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
