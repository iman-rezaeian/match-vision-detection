"""Training Session Analysis — Streamlit page for scrimmages and drills."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (CUSTOM_CSS, DEFAULT_SAMPLE_RATE, DEFAULT_CONFIDENCE,
                    DEFAULT_MODEL_SIZE, DEFAULT_PLAYERS_PER_TEAM,
                    DEFAULT_TRACKER, YOLO_VERSION)


def show():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("⚽ Training Session Analysis")

    # Session type selection
    session_type = st.radio(
        "Session Type",
        ["Scrimmage", "Drills"],
        horizontal=True,
        help="Scrimmage = intra-team game (2 teams). Drills = whole team exercises."
    )

    st.divider()

    # =====================================================================
    # Configuration Sidebar
    # =====================================================================
    with st.sidebar:
        st.header("⚙️ Training Config")

        # Camera calibration
        st.subheader("Camera")
        calib_dir = Path(__file__).parent.parent / "data" / "calibration"
        calib_files = list(calib_dir.glob("*.npz")) if calib_dir.exists() else []

        if calib_files:
            calib_options = ["None (no fisheye)"] + [f.stem for f in calib_files]
            calib_choice = st.selectbox("Fisheye Calibration", calib_options)
            calibration_path = None
            if calib_choice != "None (no fisheye)":
                calibration_path = calib_dir / f"{calib_choice}.npz"
        else:
            st.info("No calibration files found. Run `tools/calibrate_fisheye.py` first.")
            calibration_path = None

        # Field setup
        st.subheader("Field")
        col1, col2 = st.columns(2)
        with col1:
            field_length = st.number_input("Length (m)", value=40.0, min_value=10.0, max_value=120.0)
        with col2:
            field_width = st.number_input("Width (m)", value=30.0, min_value=10.0, max_value=90.0)

        flag_color = st.selectbox("Flag Color", ["red", "orange", "pink", "yellow", "green_neon", "blue_neon"], index=0)

        # Detection settings
        st.subheader("Detection")
        sample_rate = st.slider("Sample Rate", 1, 30, DEFAULT_SAMPLE_RATE,
                                help="Process every Nth frame (3=10fps best tracking, 10=3fps faster)")
        confidence = st.slider("Confidence", 0.1, 0.9, DEFAULT_CONFIDENCE, 0.05)
        model_size = st.selectbox("Model Size", ["n", "s", "m", "l", "x"],
                                  index=["n", "s", "m", "l", "x"].index(DEFAULT_MODEL_SIZE))

        if session_type == "Scrimmage":
            st.subheader("Teams")
            color_options = ["auto", "black", "white", "red", "blue", "green",
                            "yellow", "orange", "purple"]
            my_team_color = st.selectbox("My Team Color", color_options, index=0)
            opponent_color = st.selectbox("Opponent Color", color_options, index=0)
            if my_team_color == "auto":
                my_team_color = None
            if opponent_color == "auto":
                opponent_color = None
        else:
            my_team_color = None
            opponent_color = None

    # =====================================================================
    # Video Upload
    # =====================================================================
    st.subheader("📹 Upload Training Video")
    uploaded_video = st.file_uploader(
        "Select training session video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Record with fisheye lens. Place 4 neon flags at field corners."
    )

    if uploaded_video is None:
        st.info("Upload a training session video to begin analysis.")
        _show_setup_guide(session_type)
        return

    # Save uploaded video to temp file
    import tempfile
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success(f"Video loaded: {uploaded_video.name}")

    # =====================================================================
    # Processing Pipeline
    # =====================================================================
    if st.button("🚀 Analyze Training Session", type="primary"):
        _run_pipeline(
            video_path=video_path,
            session_type=session_type,
            calibration_path=calibration_path,
            field_length=field_length,
            field_width=field_width,
            flag_color=flag_color,
            sample_rate=sample_rate,
            confidence=confidence,
            model_size=model_size,
            my_team_color=my_team_color,
            opponent_color=opponent_color,
        )


def _detect_and_track(video_path, calibration_path, flag_color,
                      field_length, field_width, sample_rate,
                      confidence, model_size, progress):
    """Stage 1: Detection + Tracking + Homography with progress feedback."""
    import cv2
    import numpy as np
    from pipeline.flag_detector import FlagDetector
    from pipeline.homography import FlagHomography
    from pipeline.detector import VideoDetector

    # Init fisheye
    fisheye_calib = None
    if calibration_path and Path(calibration_path).exists():
        from pipeline.fisheye import FisheyeCalibration
        fisheye_calib = FisheyeCalibration(calibration_path)

    # Init flag detector and homography
    flag_detector = FlagDetector(flag_color=flag_color)
    homography = FlagHomography(field_length, field_width)

    # Detect flags from first frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flag_frames = []
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            if fisheye_calib:
                frame = fisheye_calib.undistort(frame)
            flag_frames.append(frame)
    cap.release()

    # Try flag-based homography
    use_flags = False
    if flag_frames:
        centroids = flag_detector.detect_stable(flag_frames)
        if len(centroids) >= 4:
            corners = flag_detector.assign_corners(centroids, flag_frames[0].shape)
            if corners:
                homography.calibrate_from_flags(corners)
                use_flags = True

    # Run detection + tracking (cap at 1500 sampled frames max)
    detector = VideoDetector(
        model_size=model_size,
        confidence=confidence,
        sample_rate=sample_rate,
    )

    max_raw_frames = 1500 * sample_rate
    end_frame = min(total_frames, max_raw_frames)
    expected_sampled = end_frame // sample_rate

    def det_progress(current, total, n_dets):
        sampled_so_far = current // sample_rate
        pct = 0.05 + 0.35 * (sampled_so_far / max(expected_sampled, 1))
        progress.progress(min(pct, 0.39),
                         text=f"Stage 1/4: {sampled_so_far}/{expected_sampled} frames ({n_dets} detections)")

    detections_df, video_meta = detector.process(
        video_path, progress_callback=det_progress, end_frame=end_frame
    )

    if detections_df.empty:
        return detections_df, video_meta, use_flags

    # Apply homography
    if not use_flags:
        homography.calibrate_auto(frame_h, frame_w, detections_df)

    detections_df = homography.filter_to_field_zone(detections_df)
    detections_df = homography.transform_df(detections_df)

    return detections_df, video_meta, use_flags


def _run_pipeline(video_path, session_type, calibration_path, field_length,
                  field_width, flag_color, sample_rate, confidence, model_size,
                  my_team_color, opponent_color):
    """Execute the training session analysis pipeline."""
    import cv2

    total_stages = 4
    progress = st.progress(0, text="Starting...")

    # -----------------------------------------------------------------
    # Stage 1: Undistort + Detect + Track (cached in session_state)
    # -----------------------------------------------------------------
    progress.progress(0.05, text="Stage 1/4: Detection & Tracking...")

    fisheye_calib = None
    if calibration_path and Path(calibration_path).exists():
        from pipeline.fisheye import FisheyeCalibration
        fisheye_calib = FisheyeCalibration(calibration_path)
        st.caption(f"🔧 Fisheye calibration: {Path(calibration_path).stem}")

    # Use session_state to cache stage 1 results
    cache_key = f"stage1_{video_path}_{sample_rate}_{confidence}_{model_size}"
    if cache_key in st.session_state:
        detections_df, video_meta, use_flags = st.session_state[cache_key]
        st.caption("⚡ Using cached detection results")
    else:
        detections_df, video_meta, use_flags = _detect_and_track(
            video_path, calibration_path, flag_color,
            field_length, field_width, sample_rate,
            confidence, model_size, progress
        )
        st.session_state[cache_key] = (detections_df, video_meta, use_flags)

    if use_flags:
        st.caption("✓ Flag-based homography")
    else:
        st.caption("⚠️ Using density-based homography (flags not detected)")

    if detections_df.empty:
        st.error("No players detected. Check video and settings.")
        return

    n_tracks = detections_df["track_id"].nunique()
    st.caption(f"✓ {len(detections_df):,} detections, {n_tracks} tracks")
    progress.progress(0.40, text="Stage 1/4: Complete ✓")

    # -----------------------------------------------------------------
    # Stage 2: Team Assignment (scrimmage only)
    # -----------------------------------------------------------------
    if session_type == "Scrimmage":
        progress.progress(0.42, text="Stage 2/4: Team Classification...")
        from pipeline.team_classifier import TeamClassifier

        classifier = TeamClassifier()

        # Sample frames for classification using VideoCapture
        sample_frames_dict = {}
        sample_frame_ids = detections_df["frame"].unique()[::10][:50]
        cap = cv2.VideoCapture(video_path)
        for fid in sample_frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if ret:
                if fisheye_calib:
                    frame = fisheye_calib.undistort(frame)
                sample_frames_dict[int(fid)] = frame
        cap.release()

        detections_df = classifier.classify_teams(
            detections_df, sample_frames_dict,
            my_team_color=my_team_color,
            opponent_color=opponent_color,
        )
        team_map = detections_df.groupby("track_id")["team"].first().to_dict()

        home_n = sum(1 for v in team_map.values() if v == "Home")
        away_n = sum(1 for v in team_map.values() if v == "Away")
        st.caption(f"✓ Teams: Home={home_n}, Away={away_n}")
    else:
        # No team split for drills
        detections_df["team"] = "All"

    progress.progress(0.55, text="Stage 2/4: Complete ✓")

    # -----------------------------------------------------------------
    # Stage 3: Player Identification (manual via crops)
    # -----------------------------------------------------------------
    progress.progress(0.57, text="Stage 3/4: Extracting player crops...")

    # Only extract crops for tracks with enough detections (real players, not noise)
    # A real player at sample_rate=10 should appear in at least 10 sampled frames
    track_counts = detections_df["track_id"].value_counts()
    min_detections = max(10, len(detections_df) // 200)  # at least 10 or 0.5% of total
    stable_tracks = track_counts[track_counts >= min_detections].index.tolist()

    player_crops = {}
    cap = cv2.VideoCapture(video_path)
    for tid in stable_tracks:
        track_dets = detections_df[detections_df["track_id"] == tid]
        # Pick the middle detection (most likely to be well-framed)
        mid_idx = len(track_dets) // 2
        best = track_dets.iloc[mid_idx]
        fid = int(best["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            if fisheye_calib:
                frame = fisheye_calib.undistort(frame)
            x1, y1 = int(best["bbox_x1"]), int(best["bbox_y1"])
            x2, y2 = int(best["bbox_x2"]), int(best["bbox_y2"])
            # Pad crop slightly for context
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # Resize crop to save memory (max 200px tall)
                h_crop = crop.shape[0]
                if h_crop > 200:
                    scale = 200 / h_crop
                    crop = cv2.resize(crop, (int(crop.shape[1] * scale), 200))
                player_crops[tid] = crop
    cap.release()

    st.caption(f"✓ Extracted crops for {len(player_crops)} tracks")
    progress.progress(0.75, text="Stage 3/4: Complete ✓")

    # Store crops in session state for manual assignment later
    st.session_state["player_crops"] = player_crops
    st.session_state["detections_df"] = detections_df

    # -----------------------------------------------------------------
    # Stage 4: Analysis (mode-specific)
    # -----------------------------------------------------------------
    progress.progress(0.77, text="Stage 4/4: Computing metrics...")

    if session_type == "Scrimmage":
        _analyze_scrimmage(detections_df, video_meta, field_length, field_width)
    else:
        fps = video_meta.get("fps", 30.0)
        _analyze_drills(detections_df, video_meta, fps)

    progress.progress(1.0, text="Analysis Complete ✓")

    # -----------------------------------------------------------------
    # Player Identification via Crops (manual assignment)
    # -----------------------------------------------------------------
    if player_crops:
        st.subheader("🧑 Assign Players")
        st.caption("Select each player's name from your roster. "
                   "Bibs cover jerseys so manual assignment is needed.")

        # Get roster from DB
        try:
            from database import RosterDB
            roster_db = RosterDB()
            players = roster_db.get_all_players()
            roster_names = ["(unknown)", "(coach/ref)"] + [p["name"] for p in players]
        except Exception:
            roster_names = ["(unknown)", "(coach/ref)"]

        # Group tracks by team
        team_tracks = {}
        for tid in player_crops:
            team = detections_df[detections_df["track_id"] == tid]["team"].mode()
            team_label = team.iloc[0] if not team.empty else "Unknown"
            team_tracks.setdefault(team_label, []).append(tid)

        assignments = {}
        for team_label in sorted(team_tracks.keys()):
            st.write(f"**{team_label} Team:**")
            tids = team_tracks[team_label]
            # Display in rows of 4
            for row_start in range(0, len(tids), 4):
                cols = st.columns(4)
                for i, tid in enumerate(tids[row_start:row_start + 4]):
                    with cols[i]:
                        crop = player_crops[tid]
                        # Convert BGR to RGB for display
                        crop_rgb = crop[:, :, ::-1]
                        st.image(crop_rgb, caption=f"Track {tid}", width=120)
                        name = st.selectbox(
                            f"Player", roster_names,
                            key=f"assign_{tid}",
                            label_visibility="collapsed"
                        )
                        if name != "(unknown)":
                            assignments[tid] = name

        if assignments:
            # Mark coaches/refs for exclusion
            coach_tids = [tid for tid, name in assignments.items() if name == "(coach/ref)"]
            player_assignments = {tid: name for tid, name in assignments.items()
                                  if name != "(coach/ref)"}
            detections_df["player_name"] = detections_df["track_id"].map(
                lambda t: player_assignments.get(t, None))
            # Filter out coach tracks from analysis
            if coach_tids:
                detections_df = detections_df[~detections_df["track_id"].isin(coach_tids)]
                st.caption(f"Excluded {len(coach_tids)} coach/ref tracks from stats")
            st.success(f"Assigned {len(player_assignments)} players")


def _analyze_scrimmage(df, video_meta, field_length, field_width):
    """Run game-style analysis for scrimmage."""
    import numpy as np
    from pipeline.stats import StatsCalculator
    from pipeline.passes import PassDetector
    from pipeline.formation import FormationDetector

    st.subheader("📊 Scrimmage Results")

    fps = video_meta.get("fps", 30.0)

    # Stats
    stats_calc = StatsCalculator(field_length=field_length, field_width=field_width, fps=fps)
    passes_detector = PassDetector()
    passes = passes_detector.detect_passes(df, fps=fps,
                                           field_length=field_length,
                                           field_width=field_width)

    stats_df = stats_calc.calculate_all_stats(df, passes)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Passes", len(passes))
    with col2:
        if not stats_df.empty and "distance_m" in stats_df.columns:
            st.metric("Avg Distance", f"{stats_df['distance_m'].mean():.0f}m")
    with col3:
        if not stats_df.empty and "top_speed_ms" in stats_df.columns:
            st.metric("Top Speed", f"{stats_df['top_speed_ms'].max():.1f} m/s")

    # Formation
    formation_detector = FormationDetector(players_per_team=DEFAULT_PLAYERS_PER_TEAM)
    if "x_field" in df.columns:
        for team_label in ["Home", "Away"]:
            team_df = df[df["team"] == team_label]
            if not team_df.empty:
                avg_pos = team_df.groupby("track_id")[["x_field", "y_field"]].mean().values
                if len(avg_pos) >= 4:
                    formation, conf = formation_detector.detect_formation(avg_pos)
                    if team_label == "Home":
                        home_formation = f"{formation} ({conf:.0%})"
                    else:
                        away_formation = f"{formation} ({conf:.0%})"
                else:
                    if team_label == "Home":
                        home_formation = "Unknown"
                    else:
                        away_formation = "Unknown"
            else:
                if team_label == "Home":
                    home_formation = "Unknown"
                else:
                    away_formation = "Unknown"
        st.write(f"**Formations:** Home: {home_formation} | Away: {away_formation}")

    # Stats table
    if not stats_df.empty:
        st.subheader("Player Stats")
        display_cols = [c for c in ["player_name", "track_id", "team", "distance_m",
                                     "top_speed_ms", "avg_speed_ms", "sprint_count"]
                        if c in stats_df.columns]
        st.dataframe(stats_df[display_cols].sort_values("distance_m", ascending=False),
                     use_container_width=True)


def _analyze_drills(df, video_meta, fps):
    """Run drill segmentation and per-drill metrics."""
    from pipeline.drill_segmenter import DrillSegmenter
    from pipeline.drill_metrics import DrillMetricsCalculator

    st.subheader("🏋️ Drill Analysis")

    # Segment drills
    segmenter = DrillSegmenter()
    segments = segmenter.segment(df, fps)

    if not segments:
        st.warning("No drill segments detected. Try adjusting idle threshold.")
        return

    st.write(f"**{len(segments)} drills detected**")

    # Drill timeline
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 2))
    colors = {"sprint": "#e74c3c", "possession": "#f39c12", "passing": "#27ae60",
              "agility": "#9b59b6", "tactical": "#3498db", "general": "#95a5a6",
              "unknown": "#bdc3c7"}
    for seg in segments:
        ax.barh(0, seg.duration_s, left=seg.start_time_s,
                color=colors.get(seg.drill_type, "#95a5a6"),
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Drill Timeline")
    st.pyplot(fig)
    plt.close(fig)

    # Drill details
    for seg in segments:
        with st.expander(f"Drill {seg.index + 1}: {seg.drill_type.title()} "
                         f"({seg.duration_s:.0f}s, {seg.player_count} players)"):
            st.write(f"- Time: {seg.start_time_s:.0f}s – {seg.end_time_s:.0f}s")
            st.write(f"- Avg Intensity: {seg.avg_intensity:.1f} m/s")
            st.write(f"- Peak Intensity: {seg.max_intensity:.1f} m/s")

    # Per-drill metrics
    metrics_calc = DrillMetricsCalculator()
    metrics_df = metrics_calc.compute_all(df, segments, fps)

    if not metrics_df.empty:
        st.subheader("Player Metrics")
        session_summary = metrics_calc.summarize_session(metrics_df)
        if not session_summary.empty:
            st.dataframe(session_summary.sort_values("total_distance_m", ascending=False),
                         use_container_width=True)


def _show_setup_guide(session_type):
    """Show setup instructions for new users."""
    with st.expander("📋 Setup Guide", expanded=True):
        st.markdown("""
        ### Before Recording

        1. **Calibrate your fisheye lens** (one-time):
           ```bash
           python tools/calibrate_fisheye.py /path/to/checkerboard_video.mov
           ```

        2. **Place 4 neon flags** at field corners
           - Use bright orange, pink, or yellow flags/cones
           - Measure field dimensions (length × width in meters)

        3. **Mount camera** at midfield sideline on tripod
           - Aim for full field coverage with fisheye lens
           - Ensure all 4 flags are visible in frame

        4. **Record continuously** through the entire session
           - Don't stop recording between drills
           - Keep camera position fixed
        """)

        if session_type == "Drills":
            st.markdown("""
            ### Drill-Specific Tips
            - The system auto-detects drill boundaries from movement patterns
            - Clear transitions (players walking/standing) between drills help segmentation
            - All 16 players can be on the field simultaneously
            """)


# Entry point when loaded as a Streamlit page
show()
