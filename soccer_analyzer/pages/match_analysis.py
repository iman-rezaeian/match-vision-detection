"""Match Analysis — Main analysis Streamlit page."""

import gc
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os

# Raise PIL decompression bomb limit for large heatmap figures
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import io
import zipfile
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (CUSTOM_CSS, DEFAULT_SAMPLE_RATE, DEFAULT_CONFIDENCE,
                    DEFAULT_MODEL_SIZE, DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M,
                    DEFAULT_PLAYERS_PER_TEAM, SEGMENT_LABELS)
from database.roster_db import RosterDB
from database.match_db import MatchDB
from pipeline.detector import VideoDetector
from pipeline.multi_segment import GameSegment, MultiSegmentProcessor, get_frame_at_time, VideoFrameReader
from pipeline.homography import FieldHomography
from pipeline.team_classifier import TeamClassifier
try:
    from pipeline.face_reid import FaceReID
except ImportError:
    FaceReID = None
from pipeline.gait import GaitAnalyzer
from pipeline.cleat import CleatExtractor
from pipeline.fingerprint import PlayerFingerprinter
from pipeline.passes import PassDetector
from pipeline.stats import StatsCalculator
from pipeline.formation import FormationDetector
from visualization.heatmaps import plot_player_heatmap, plot_all_heatmaps_grid
from visualization.passing_network import plot_passing_network, plot_pass_matrix_table
from visualization.pitch_overview import plot_pitch_overview
from visualization.timeline import plot_zone_timeline
from visualization.formation_plot import (plot_formation, plot_formation_timeline,
                                           plot_compactness_chart)
from visualization.report import generate_pdf_report


def get_db():
    if "roster_db" not in st.session_state:
        st.session_state["roster_db"] = RosterDB()
    return st.session_state["roster_db"]


def get_match_db():
    if "match_db" not in st.session_state:
        st.session_state["match_db"] = MatchDB()
    return st.session_state["match_db"]


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("⚽ Match Analysis")

db = get_db()
match_db = get_match_db()

# --- Step 1: Match Setup ---
st.header("Match Setup")
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
with col1:
    match_date = st.date_input("Date", value=date.today())
with col2:
    opponent = st.text_input("Opponent", placeholder="e.g. Blue Thunder FC")
with col3:
    fields = db.get_all_fields()
    field_names = [f["name"] for f in fields]
    selected_field = st.selectbox("Field", field_names if field_names else ["Auto (no calibration)"])
with col4:
    result = st.text_input("Result", placeholder="e.g. 3-1 W")
with col5:
    _jersey_options = ["⬛ Black", "⬜ White", "🟥 Red", "🟦 Blue", "🟩 Green", "🟨 Yellow", "🟧 Orange", "🟪 Purple"]
    my_team_jersey = st.selectbox(
        "My Team Jersey",
        _jersey_options,
        index=0,
        help="Select your team's jersey color"
    )
    my_team_color = my_team_jersey.split()[-1].lower()
with col6:
    opponent_jersey = st.selectbox(
        "Opponent Jersey",
        _jersey_options,
        index=1,
        help="Select the opponent's jersey color"
    )
    opponent_color = opponent_jersey.split()[-1].lower()

# --- Step 2: Video Upload ---
st.header("Video Upload")

input_method = st.radio(
    "How to load video(s)?",
    ["📂 Local folder or file paths", "⬆️ Upload through browser"],
    horizontal=True,
)

video_files = None
local_paths_input = None

if input_method.startswith("⬆️"):
    video_files = st.file_uploader(
        "Drop game video(s) — one per half/period (MP4/MOV)",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=True,
        help="Upload one or more clips. You'll set kickoff and whistle times for each."
    )
else:
    local_paths_input = st.text_input(
        "Paste folder path or file path(s)",
        placeholder="/Users/you/Movies/Game vs Thunder",
        help="Paste a folder path to auto-discover all video files, or a single file path."
    )
    if local_paths_input:
        p = local_paths_input.strip().strip('"').strip("'")
        if os.path.isdir(p):
            # Auto-discover video files in folder
            VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}
            found = sorted([
                f for f in os.listdir(p)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS
            ])
            if found:
                st.success(f"Found {len(found)} video(s) in folder:")
                for f in found:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📹 `{f}`")
                # Store as newline-separated full paths for processing below
                local_paths_input = "\n".join(os.path.join(p, f) for f in found)
            else:
                st.warning("No video files (.mp4, .mov, .avi) found in this folder.")
                local_paths_input = None

# Build temp_paths list from either input method
temp_paths = []

if video_files:
    video_files = sorted(video_files, key=lambda f: f.name)
    for vf in video_files:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vf.read())
            temp_paths.append((vf.name, tmp.name))

elif local_paths_input:
    for line in local_paths_input.strip().splitlines():
        p = line.strip().strip('"').strip("'")
        if not p:
            continue
        if not os.path.isfile(p):
            st.error(f"File not found: `{p}`")
            temp_paths = []
            break
        temp_paths.append((os.path.basename(p), p))
    # Sort by filename
    temp_paths = sorted(temp_paths, key=lambda x: x[0])

if temp_paths:

    # Build GameSegment objects and read video info
    segments: list[GameSegment] = []
    for idx, (orig_name, tmp_path) in enumerate(temp_paths):
        if len(temp_paths) == 1:
            default_label = "Full Game"
        else:
            # Skip "Full Game" (index 0) when multiple clips
            half_labels = SEGMENT_LABELS[1:]
            default_label = half_labels[idx] if idx < len(half_labels) else f"Segment {idx + 1}"
        seg = GameSegment(
            video_path=tmp_path,
            segment_id=idx,
            label=default_label,
        )
        seg.read_video_info()
        segments.append(seg)

    # --- Step 2b: Segment Trim UI ---
    st.header("Trim Segments")
    st.caption("Set the kickoff and final whistle for each clip to exclude pre/post-game footage.")

    for seg in segments:
        orig_name = temp_paths[seg.segment_id][0]
        with st.expander(f"📹 {orig_name}", expanded=True):
            col_info1, col_info2, col_info3, col_label = st.columns([1, 1, 1, 1])
            with col_info1:
                st.metric("Duration", f"{seg.duration_s / 60:.1f} min")
            with col_info2:
                st.metric("Resolution", f"{seg.frame_w}×{seg.frame_h}")
            with col_info3:
                st.metric("FPS", f"{seg.fps:.0f}")
            with col_label:
                default_idx = SEGMENT_LABELS.index(seg.label) if seg.label in SEGMENT_LABELS else 0
                seg.label = st.selectbox(
                    "Label", SEGMENT_LABELS,
                    index=default_idx,
                    key=f"label_{seg.segment_id}"
                )

            col_kick, col_whis = st.columns(2)
            with col_kick:
                kick_s = st.slider(
                    "⚽ Kickoff",
                    min_value=0.0,
                    max_value=seg.duration_s,
                    value=0.0,
                    step=1.0,
                    format="%.0f s",
                    key=f"kick_{seg.segment_id}",
                    help="Drag to the moment of kickoff"
                )
                seg.kickoff_s = kick_s
                # Show preview frame at kickoff
                kick_frame = get_frame_at_time(seg.video_path, kick_s)
                if kick_frame is not None:
                    st.image(kick_frame, caption=f"Kickoff @ {kick_s:.0f}s", width=320)

            with col_whis:
                whis_s = st.slider(
                    "🏁 Final Whistle",
                    min_value=0.0,
                    max_value=seg.duration_s,
                    value=seg.duration_s,
                    step=1.0,
                    format="%.0f s",
                    key=f"whis_{seg.segment_id}",
                    help="Drag to the moment of the final whistle"
                )
                seg.whistle_s = whis_s
                # Show preview frame at whistle
                whis_frame = get_frame_at_time(seg.video_path, whis_s)
                if whis_frame is not None:
                    st.image(whis_frame, caption=f"Whistle @ {whis_s:.0f}s", width=320)

            gameplay = seg.gameplay_duration_s
            st.success(f"✅ Gameplay: **{gameplay / 60:.1f} min** "
                       f"(frames {seg.kickoff_frame:,} → {seg.whistle_frame:,})")

    # Total gameplay summary
    total_gameplay = sum(seg.gameplay_duration_s for seg in segments)
    st.info(f"📊 **Total gameplay: {total_gameplay / 60:.1f} min** across {len(segments)} segment(s)")

    # Video info for processing estimate (use first segment)
    fps = segments[0].fps
    frame_w = segments[0].frame_w
    frame_h = segments[0].frame_h

    # --- Step 3: Processing Settings (auto-tuned from video metadata) ---
    # Auto-compute optimal values
    _auto_sample_rate = max(1, round(fps / 10))  # target ~10 effective fps
    _auto_confidence = 0.25 if frame_w >= 3840 else 0.30 if frame_w >= 1920 else 0.35
    _auto_model = "s"  # best speed/accuracy on MPS; "m" is 2x slower for ~3% gain

    with st.expander("⚙️ Processing Settings"):
        st.caption(f"Auto-tuned for {frame_w}×{frame_h} @ {fps:.0f}fps • "
                   f"Recommended: rate={_auto_sample_rate}, conf={_auto_confidence}, model={_auto_model}")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            sample_rate = st.slider("Sample Rate (every N frames)",
                                     1, 10, _auto_sample_rate)
        with col2:
            confidence = st.slider("Detection Confidence",
                                    0.1, 0.9, _auto_confidence, 0.05)
        with col3:
            _model_options = ["n (Nano - Fast)", "s (Small)", "x (Large - Accurate)"]
            _model_default = next((i for i, o in enumerate(_model_options) if o.startswith(DEFAULT_MODEL_SIZE)), 1)
            model_size = st.selectbox("YOLOv8 Model",
                                       _model_options,
                                       index=_model_default)
            model_size = model_size[0]  # Extract letter
        with col4:
            players_per_team = st.number_input("Players per Team",
                                                min_value=5, max_value=15,
                                                value=DEFAULT_PLAYERS_PER_TEAM, step=1)

        col_fl, col_fw = st.columns(2)
        with col_fl:
            field_length_input = st.number_input("Field Length (m)",
                                                  min_value=20.0, max_value=120.0,
                                                  value=DEFAULT_FIELD_LENGTH_M, step=1.0)
        with col_fw:
            field_width_input = st.number_input("Field Width (m)",
                                                 min_value=15.0, max_value=90.0,
                                                 value=DEFAULT_FIELD_WIDTH_M, step=1.0)

    # --- Step 4: Analyze Button ---
    if st.button("⚽ Analyze Match", use_container_width=True, type="primary"):
        st.session_state["analyzing"] = True

        st.warning("⚠️ **Do NOT switch tabs** while analysis is running — it will cancel the process!", icon="🚫")

        # Stage 1: Detection & Tracking (all segments)
        stage1 = st.empty()
        progress1 = st.progress(0, text="Stage 1/5: Detecting & tracking players...")

        # Check for cached Stage 1 results first
        match_key = f"{match_date}|{opponent}"
        cached = MultiSegmentProcessor.load_stage1_cache(segments, match_key)
        if cached is not None:
            detections_df, video_meta = cached
            progress1.progress(1.0, text="Stage 1/5: Loaded from cache ✓")
            multi_processor = MultiSegmentProcessor(
                VideoDetector(model_size=model_size, confidence=confidence,
                              sample_rate=sample_rate)
            )
        else:
            detector = VideoDetector(model_size=model_size, confidence=confidence,
                                     sample_rate=sample_rate)
            multi_processor = MultiSegmentProcessor(detector)
            multi_processor._match_key = match_key

            def segment_progress(seg_id, seg_label, msg):
                # Parse overall progress from msg if available
                if "overall" in msg:
                    try:
                        parts = msg.split("overall ")[1]
                        done, total = parts.split("/")
                        frac = min(int(done) / int(total), 0.99)
                    except (IndexError, ValueError):
                        frac = min((seg_id + 0.5) / len(segments), 0.99)
                else:
                    frac = min((seg_id + 0.5) / len(segments), 0.99)
                progress1.progress(frac, text=f"Stage 1/5: {msg}")

            detections_df, video_meta = multi_processor.process_segments(
                segments, progress_callback=segment_progress
            )
            progress1.progress(1.0, text="Stage 1/5: Detection complete ✓")

        if detections_df.empty:
            st.error("No players detected. Try lowering the confidence threshold or check video brightness.")
            st.stop()

        # Checkpoint: save Stage 1 results so they survive tab switches
        st.session_state["stage1_checkpoint"] = {
            "detections_df": detections_df,
            "video_meta": video_meta,
            "segments": segments,
            "multi_processor": multi_processor,
            "field_length_input": field_length_input,
            "field_width_input": field_width_input,
            "selected_field": selected_field,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "players_per_team": players_per_team,
            "match_date": str(match_date),
            "opponent": opponent,
            "result": result,
            "fps": fps,
        }

        # Set up homography (data-driven: uses detection positions to find field zone)
        homography = FieldHomography(field_length_m=field_length_input, field_width_m=field_width_input)
        if selected_field and selected_field != "Auto (no calibration)":
            loaded = homography.load(selected_field, db)
            if not loaded:
                homography.calibrate_auto(frame_h, frame_w, detections_df)
        else:
            homography.calibrate_auto(frame_h, frame_w, detections_df)

        # Filter detections to the playing field zone (remove sideline spectators/coaches)
        pre_filter = len(detections_df)
        detections_df = homography.filter_to_field_zone(detections_df)
        post_filter = len(detections_df)
        if pre_filter > post_filter:
            st.caption(f"Field zone filter: {pre_filter:,} → {post_filter:,} detections "
                      f"(removed {pre_filter - post_filter:,} off-field)")

        # Transform coordinates
        detections_df = homography.transform_df(detections_df)
        val_score, val_msg = homography.validate(detections_df)
        if val_score < 0.8:
            st.warning(val_msg)

        # Stage 2: Building player fingerprints
        progress2 = st.progress(0, text="Stage 2/5: Building player fingerprints...")

        # Check Stage 2 cache (detections with team labels)
        stage2_cached = MultiSegmentProcessor.load_stage2_cache(match_key)
        if stage2_cached is not None:
            detections_df, cached_team_colors = stage2_cached
            team_classifier = TeamClassifier()
            team_classifier.team_colors = cached_team_colors
            frames = None  # Will create VideoFrameReader in Stage 3 if required
            progress2.progress(1.0, text="Stage 2/5: Loaded from cache ✓")
        else:
            # On-demand frame reader — reads 4K frames from video as needed, no bulk RAM
            progress2.progress(0.1, text="Stage 2/5: Preparing frame reader...")
            frames = VideoFrameReader(segments, detections_df)
            progress2.progress(0.5, text="Stage 2/5: Classifying teams...")

            # Team classification
            team_classifier = TeamClassifier()

            def team_classify_progress(done, total):
                frac = 0.5 + min(done / total * 0.45, 0.45)
                progress2.progress(frac, text=f"Stage 2/5: Classifying teams ({done}/{total} tracks)...")

            detections_df = team_classifier.classify_teams(
                detections_df, frames, progress_callback=team_classify_progress,
                my_team_color=my_team_color,
                opponent_color=opponent_color,
            )
            progress2.progress(1.0, text="Stage 2/5: Fingerprints built ✓")

            # Save Stage 2 to disk cache
            MultiSegmentProcessor.save_stage2_cache(match_key, detections_df, team_classifier.team_colors)

        # Free Stage 1 memory — release YOLO model from cache to reclaim ~200MB
        from pipeline.detector import load_yolo_model
        load_yolo_model.clear()
        del multi_processor
        if "stage1_checkpoint" in st.session_state:
            del st.session_state["stage1_checkpoint"]
        gc.collect()

        # Store intermediate state for team confirmation
        # Close VideoFrameReader if open — will create fresh one in Stage 3
        if hasattr(frames, 'close'):
            frames.close()
        st.session_state["team_confirm"] = {
            "detections_df": detections_df,
            "team_classifier": team_classifier,
            "video_meta": video_meta,
            "homography": homography,
            "segments": segments,
            "players_per_team": players_per_team,
            "match_date": str(match_date),
            "opponent": opponent,
            "result": result,
            "selected_field": selected_field,
            "fps": fps,
        }
        st.rerun()

# --- Team Confirmation Step (auto-assigned via jersey color selector) ---
if "team_confirm" in st.session_state and "match_results" not in st.session_state:
    tc = st.session_state["team_confirm"]
    team_classifier = tc["team_classifier"]
    detections_df = tc["detections_df"]

    # Show team colors for info
    home_rgb = team_classifier.get_team_color_rgb("Home")
    away_rgb = team_classifier.get_team_color_rgb("Away")

    def _contrast_text(r, g, b):
        """Return black or white text depending on background luminance."""
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if lum > 140 else "#ffffff"

    st.subheader("🎽 Team Colors Detected")
    col_a, col_b = st.columns(2)
    home_txt = _contrast_text(*home_rgb)
    away_txt = _contrast_text(*away_rgb)
    home_shadow = "0 0 4px rgba(0,0,0,0.7)" if home_txt == "#ffffff" else "0 0 4px rgba(255,255,255,0.5)"
    away_shadow = "0 0 4px rgba(0,0,0,0.7)" if away_txt == "#ffffff" else "0 0 4px rgba(255,255,255,0.5)"
    with col_a:
        st.markdown(
            f'<div style="background-color:rgb({home_rgb[0]},{home_rgb[1]},{home_rgb[2]});'
            f'width:100%;height:60px;border-radius:10px;border:3px solid #555;'
            f'display:flex;align-items:center;justify-content:center;color:{home_txt};'
            f'font-weight:bold;font-size:18px;text-shadow:{home_shadow};">My Team</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div style="background-color:rgb({away_rgb[0]},{away_rgb[1]},{away_rgb[2]});'
            f'width:100%;height:60px;border-radius:10px;border:3px solid #555;'
            f'display:flex;align-items:center;justify-content:center;color:{away_txt};'
            f'font-weight:bold;font-size:18px;text-shadow:{away_shadow};">Opponent</div>',
            unsafe_allow_html=True,
        )

    # Auto-continue (user can swap if wrong)
    col_ok, col_swap = st.columns(2)
    with col_ok:
        pick_ok = st.button("✅ Looks correct — continue", key="pick_ok", use_container_width=True)
    with col_swap:
        pick_swap = st.button("🔄 Swap teams", key="pick_swap", use_container_width=True)

    if pick_ok or pick_swap:
        if pick_swap:
            detections_df = team_classifier.swap_teams(detections_df)

        homography = tc["homography"]
        segments = tc["segments"]
        video_meta = tc["video_meta"]
        players_per_team = tc["players_per_team"]
        fps = tc["fps"]
        db = RosterDB()
        match_key = f"{tc['match_date']}|{tc['opponent']}"

        # === FILTER: Keep only MyTeam (Home) tracks ===
        # Compute possession estimate BEFORE dropping opponent
        home_det_count = (detections_df["team"] == "Home").sum()
        away_det_count = (detections_df["team"] == "Away").sum()
        total_det = home_det_count + away_det_count
        possession_pct = (home_det_count / total_det * 100) if total_det > 0 else 50.0

        # Drop opponent and unknown tracks
        detections_df = detections_df[detections_df["team"] == "Home"].copy()
        st.caption(f"Filtered to My Team: {detections_df['track_id'].nunique()} tracks, "
                   f"{len(detections_df):,} detections (possession est. {possession_pct:.0f}%)")

        # Create fresh on-demand frame reader for Stage 3
        frames = VideoFrameReader(segments, detections_df)

        # Stage 3: Identifying players
        progress3 = st.progress(0, text="Stage 3/5: Identifying players...")

        # Check Stage 3 cache first
        stage3_cached = MultiSegmentProcessor.load_stage3_cache(match_key)
        if stage3_cached is not None:
            detections_df = stage3_cached
            # Ensure cached data is also my-team-only
            if "team" in detections_df.columns:
                detections_df = detections_df[detections_df["team"] == "Home"].copy()
            assignments = {}
            pending = []
            progress3.progress(1.0, text="Stage 3/5: Loaded from cache ✓")
            st.caption("Player ID: loaded from Stage 3 cache")
        else:
            # Track filtering — MyTeam only (opponent already removed)
            track_counts = detections_df.groupby("track_id").size().reset_index(name="count")
            n_before = track_counts.shape[0]

            players = db.get_all_players()
            has_roster = len(players) > 0 and any(p.get("photo_path") for p in players)

            # Narrow: top tracks for player ID (face matching)
            # Use roster size + buffer to account for subs throughout the game
            roster_size = len(players) if has_roster else players_per_team
            id_per_team = roster_size + 4
            id_keep_ids = set(
                track_counts.nlargest(id_per_team, "count")["track_id"].tolist()
            )

            # Broad: all tracks with ≥3 detections for stats
            MIN_DET_STATS = 3
            stats_keep_ids = set(
                track_counts[track_counts["count"] >= MIN_DET_STATS]["track_id"].tolist()
            )
            all_keep_ids = id_keep_ids | stats_keep_ids

            det_in_keep = detections_df[detections_df["track_id"].isin(all_keep_ids)].shape[0]
            det_total = len(detections_df)
            pct = det_in_keep / det_total * 100 if det_total else 0
            st.caption(
                f"Player ID: {len(id_keep_ids)} tracks | "
                f"Stats: {len(all_keep_ids)} tracks "
                f"({det_in_keep:,}/{det_total:,} detections = {pct:.0f}%) "
                f"from {n_before:,} total"
            )

            # Save broad set; filter narrow set for player ID
            broad_detections_df = detections_df[
                detections_df["track_id"].isin(all_keep_ids)
            ].copy()
            detections_for_id = detections_df[
                detections_df["track_id"].isin(id_keep_ids)
            ].copy()

            # Rebuild frame reader with narrow set (fewer frames to decode)
            if hasattr(frames, 'close'):
                frames.close()
            frames = VideoFrameReader(segments, detections_for_id)

            if has_roster:
                # Create fresh VideoFrameReader if frames was None (Stage 2 from cache)
                if not isinstance(frames, VideoFrameReader):
                    frames = VideoFrameReader(segments, detections_for_id)

                # Full identification pipeline — jersey OCR, face, gait, cleat
                face_reid = FaceReID()
                progress3.progress(0.1, text="Stage 3/5: Building roster embeddings...")
                face_reid.build_roster_embeddings(players, db)

                progress3.progress(0.2, text="Stage 3/5: Initializing jersey OCR, gait & cleat analyzers...")
                from pipeline.jersey_ocr import JerseyOCR
                jersey_ocr = JerseyOCR()
                gait_analyzer = GaitAnalyzer()
                cleat_extractor = CleatExtractor()

                fingerprinter = PlayerFingerprinter(db, face_reid, gait_analyzer, cleat_extractor, jersey_ocr=jersey_ocr)

                def id_progress(current, total):
                    frac = 0.25 + min(current / total * 0.7, 0.7)
                    progress3.progress(frac,
                                      text=f"Stage 3/5: Identifying players ({current}/{total} tracks)")

                assignments, pending = fingerprinter.identify_all_tracks(
                    detections_for_id, frames, progress_callback=id_progress
                )

                progress3.progress(0.95, text="Stage 3/5: Merging track assignments...")
                detections_for_id = fingerprinter.merge_track_ids(assignments, detections_for_id)
                if hasattr(frames, 'close'):
                    frames.close()
                del frames, face_reid, gait_analyzer, cleat_extractor, fingerprinter

                # Transfer ID results from narrow → broad set
                id_info = detections_for_id.groupby("track_id")[
                    ["player_id", "player_name", "jersey_number", "id_confidence"]
                ].first()
                for col in ["player_id", "player_name", "jersey_number", "id_confidence"]:
                    broad_detections_df[col] = broad_detections_df["track_id"].map(
                        id_info[col]
                    )
                # Auto-label unidentified broad tracks (my team only)
                unnamed_mask = broad_detections_df["player_name"].isna()
                if unnamed_mask.any():
                    unnamed_tids = broad_detections_df.loc[unnamed_mask, "track_id"].unique()

                    # Only top tracks get individual labels; rest → "Sub"
                    unnamed_counts = broad_detections_df[unnamed_mask].groupby("track_id").size()
                    # How many named players already exist
                    existing_named = broad_detections_df.loc[
                        ~unnamed_mask, "track_id"
                    ].nunique()
                    # Fill up to players_per_team
                    slots = max(0, players_per_team - existing_named)
                    top_unnamed_home = set(
                        unnamed_counts.nlargest(slots).index.tolist()
                    )

                    home_c = existing_named
                    for tid in unnamed_tids:
                        if tid in top_unnamed_home:
                            home_c += 1
                            name = f"Player {home_c}"
                        else:
                            name = "Sub"
                        broad_detections_df.loc[
                            broad_detections_df["track_id"] == tid, "player_name"
                        ] = name
                    broad_detections_df["player_id"] = broad_detections_df["player_id"].fillna("")
                    broad_detections_df["jersey_number"] = broad_detections_df["jersey_number"].fillna(0).astype(int)
                    broad_detections_df["id_confidence"] = broad_detections_df["id_confidence"].fillna(0.0)
                detections_df = broad_detections_df
            else:
                # No roster — auto-label top tracks on the broad set
                progress3.progress(0.3, text="Stage 3/5: Auto-labeling tracks (no roster photos)...")

                assignments = {}
                pending = []
                track_name_map = {}

                # Rank tracks by detection count
                broad_track_counts = broad_detections_df.groupby("track_id").size().reset_index(name="count")

                # Top players_per_team tracks get individual labels
                top_ids = set(
                    broad_track_counts.nlargest(players_per_team, "count")["track_id"].tolist()
                )

                counter = 0
                for tid in broad_detections_df["track_id"].unique():
                    if tid in top_ids:
                        counter += 1
                        track_name_map[tid] = f"Player {counter}"
                    else:
                        track_name_map[tid] = "Sub"

                    assignments[tid] = {
                        "player_id": None,
                        "confidence": 0.0,
                        "status": "auto_labeled",
                    }

                broad_detections_df["player_id"] = None
                broad_detections_df["player_name"] = broad_detections_df["track_id"].map(track_name_map)
                broad_detections_df["jersey_number"] = 0
                broad_detections_df["id_confidence"] = 0.0
                detections_df = broad_detections_df
                if hasattr(frames, 'close'):
                    frames.close()
                del frames

            gc.collect()
            MultiSegmentProcessor.save_stage3_cache(match_key, detections_df)
            progress3.progress(1.0, text="Stage 3/5: Identification complete ✓")

        # Stage 4 & 5: Stats, passes, formations (my team only)
        stage45_cached = MultiSegmentProcessor.load_stage45_cache(match_key)
        if stage45_cached is not None:
            stats_df = stage45_cached["stats_df"]
            passes = stage45_cached["passes"]
            home_pass_matrix = stage45_cached["home_pass_matrix"]
            home_formation_timeline = stage45_cached["home_formation_timeline"]
            home_compactness = stage45_cached["home_compactness"]
            n_players = len(stats_df) if not stats_df.empty else 0
            progress4 = st.progress(1.0, text=f"Stage 4/5: Loaded from cache — {n_players} players, {len(passes)} passes ✓")
            progress5 = st.progress(1.0, text="Stage 5/5: Loaded from cache ✓")
        else:
            progress4 = st.progress(0, text="Stage 4/5: Detecting passes...")

            # Pass detection
            pass_detector = PassDetector()
            passes = pass_detector.detect_passes(
                detections_df, fps, homography.field_length, homography.field_width
            )
            progress4.progress(0.4, text=f"Stage 4/5: {len(passes)} passes detected. Computing player stats...")

            # Stats calculation
            stats_calc = StatsCalculator(homography.field_length, homography.field_width, fps)
            stats_df = stats_calc.calculate_all_stats(detections_df, passes)
            n_players = len(stats_df) if not stats_df.empty else 0
            progress4.progress(1.0, text=f"Stage 4/5: Statistics complete — {n_players} players, {len(passes)} passes ✓")

            # Stage 5: Generating visualizations
            progress5 = st.progress(0, text="Stage 5/5: Detecting formations...")

            # Formation detection (my team only)
            formation_detector = FormationDetector(players_per_team=players_per_team)
            home_formation_timeline = formation_detector.formation_over_time(detections_df, "Home")
            progress5.progress(0.3, text="Stage 5/5: Computing team compactness...")
            home_compactness = formation_detector.compactness_over_time(detections_df, "Home")

            progress5.progress(0.6, text="Stage 5/5: Building passing network...")
            # Pass matrix (my team only)
            all_players = stats_df["name"].tolist()
            home_pass_matrix = pass_detector.build_pass_matrix(passes, all_players)

            progress5.progress(0.95, text="Stage 5/5: Caching results...")
            MultiSegmentProcessor.save_stage45_cache(
                match_key, stats_df, passes,
                home_pass_matrix, pd.DataFrame(),
                home_formation_timeline, [],
                home_compactness, [],
            )
            progress5.progress(1.0, text="Stage 5/5: Visualizations ready ✓")

        # Clean up team_confirm state
        del st.session_state["team_confirm"]

        # Store results in session state
        match_date = tc["match_date"]
        opponent = tc["opponent"]
        result = tc["result"]
        selected_field = tc["selected_field"]
        match_key = f"{match_date}|{opponent}"

        st.session_state["match_results"] = {
            "detections_df": detections_df,
            "stats_df": stats_df,
            "passes": passes,
            "home_pass_matrix": home_pass_matrix,
            "home_formation_timeline": home_formation_timeline,
            "home_compactness": home_compactness,
            "possession_pct": possession_pct,
            "assignments": assignments,
            "pending": pending,
            "video_meta": video_meta,
            "field_length": homography.field_length,
            "field_width": homography.field_width,
            "segments": segments,
            "players_per_team": players_per_team,
            "match_key": match_key,
            "match_info": {
                "date": str(match_date),
                "opponent": opponent,
                "result": result,
                "field_name": selected_field,
                "n_segments": len(segments),
                "segment_labels": [s.label for s in segments],
            },
        }

        st.success("✅ Analysis complete!")
        st.rerun()

# --- Resume from checkpoint if Stage 1 was interrupted ---
if "stage1_checkpoint" in st.session_state and "team_confirm" not in st.session_state and "match_results" not in st.session_state:
    st.info("🔄 Stage 1 results recovered from previous run. Click **Resume Analysis** to continue from Stage 2.")
    if st.button("🔄 Resume Analysis", use_container_width=True, type="primary"):
        ckpt = st.session_state["stage1_checkpoint"]
        detections_df = ckpt["detections_df"]
        video_meta = ckpt["video_meta"]
        segments = ckpt["segments"]
        multi_processor = ckpt["multi_processor"]
        db = RosterDB()

        # Set up homography (data-driven)
        homography = FieldHomography(
            field_length_m=ckpt["field_length_input"],
            field_width_m=ckpt["field_width_input"],
        )
        sel_field = ckpt["selected_field"]
        if sel_field and sel_field != "Auto (no calibration)":
            loaded = homography.load(sel_field, db)
            if not loaded:
                homography.calibrate_auto(ckpt["frame_h"], ckpt["frame_w"], detections_df)
        else:
            homography.calibrate_auto(ckpt["frame_h"], ckpt["frame_w"], detections_df)

        detections_df = homography.filter_to_field_zone(detections_df)
        detections_df = homography.transform_df(detections_df)

        # Stage 2
        progress2 = st.progress(0, text="Stage 2/5: Building player fingerprints...")
        resume_match_key = f"{ckpt['match_date']}|{ckpt['opponent']}"

        stage2_cached = MultiSegmentProcessor.load_stage2_cache(resume_match_key)
        if stage2_cached is not None:
            detections_df, cached_team_colors = stage2_cached
            team_classifier = TeamClassifier()
            team_classifier.team_colors = cached_team_colors
            progress2.progress(1.0, text="Stage 2/5: Loaded from cache ✓")
        else:
            # On-demand frame reader for resume path
            progress2.progress(0.1, text="Stage 2/5: Preparing frame reader...")
            frames = VideoFrameReader(segments, detections_df)
            progress2.progress(0.5, text="Stage 2/5: Classifying teams...")

            team_classifier = TeamClassifier()

            def team_classify_progress_r(done, total):
                frac = 0.5 + min(done / total * 0.45, 0.45)
                progress2.progress(frac, text=f"Stage 2/5: Classifying teams ({done}/{total} tracks)...")

            detections_df = team_classifier.classify_teams(
                detections_df, frames, progress_callback=team_classify_progress_r,
                my_team_color=my_team_color if 'my_team_color' in dir() else "black",
                opponent_color=opponent_color if 'opponent_color' in dir() else "white",
            )
            progress2.progress(1.0, text="Stage 2/5: Fingerprints built ✓")

            MultiSegmentProcessor.save_stage2_cache(resume_match_key, detections_df, team_classifier.team_colors)

        fps = ckpt["fps"]
        del st.session_state["stage1_checkpoint"]
        from pipeline.detector import load_yolo_model
        load_yolo_model.clear()
        del multi_processor
        if hasattr(frames, 'close'):
            frames.close()
        gc.collect()
        st.session_state["team_confirm"] = {
            "detections_df": detections_df,
            "team_classifier": team_classifier,
            "video_meta": video_meta,
            "homography": homography,
            "segments": segments,
            "players_per_team": ckpt["players_per_team"],
            "match_date": ckpt["match_date"],
            "opponent": ckpt["opponent"],
            "result": ckpt["result"],
            "selected_field": sel_field,
            "fps": fps,
        }
        st.rerun()

# --- Step 6 & 7: Results Display ---
if "match_results" in st.session_state:
    results = st.session_state["match_results"]
    detections_df = results["detections_df"]
    stats_df = results["stats_df"]
    passes = results["passes"]
    field_length = results["field_length"]
    field_width = results["field_width"]

    st.divider()

    # Confirmation queue for uncertain IDs
    pending = results.get("pending", [])
    if pending:
        st.header("🔍 Confirm Player Identifications")
        st.markdown(f"**{len(pending)} track(s)** need manual confirmation:")

        for item in pending:
            with st.expander(f"Track #{item['track_id']} (confidence: {item['confidence']:.0%})"):
                candidates = item.get("candidates", [])
                players = db.get_all_players()
                player_map = {p["id"]: p["name"] for p in players}

                options = ["Unknown"] + [player_map.get(c["player_id"], f"Player {c['player_id']}")
                                          for c in candidates]
                selected = st.selectbox(f"Who is Track #{item['track_id']}?",
                                         options, key=f"confirm_{item['track_id']}")

                if st.button("Confirm", key=f"btn_{item['track_id']}"):
                    if selected != "Unknown":
                        # Update assignment
                        for pid, name in player_map.items():
                            if name == selected:
                                results["assignments"][item['track_id']] = {
                                    "player_id": pid,
                                    "confidence": 0.85,
                                    "status": "manually_confirmed",
                                }
                                break
                    st.success(f"Confirmed Track #{item['track_id']} as {selected}")


def _render_overview_tab(results, stats_df, detections_df, field_length, field_width):
    """Render the overview tab with key metrics."""
    # Filter out aggregate "Sub" rows for display
    display_stats = stats_df[~stats_df["name"].isin(["Sub"])]
    display_det = detections_df[~detections_df["player_name"].isin(["Sub"])]

    # Metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Players Tracked", len(display_stats))
        st.metric("Team Distance", f"{stats_df['distance_m'].sum():.0f} m")
    with col2:
        avg_conf = display_stats["id_confidence"].mean() if not display_stats.empty else 0
        st.metric("Avg ID Confidence", f"{avg_conf:.0%}")
    with col3:
        possession_pct = results.get("possession_pct", 50.0)
        st.metric("Possession Est.", f"{possession_pct:.0f}%")

    # Pitch overview — only named players (not Sub aggregates)
    st.subheader("Average Positions")
    fig = plot_pitch_overview(display_det, field_length, field_width)
    st.pyplot(fig)
    plt.close(fig)


def _render_players_tab(stats_df):
    """Render the players statistics tab."""
    if stats_df.empty:
        st.info("No player statistics available.")
        return

    # Exclude aggregate Sub rows
    display_stats = stats_df[~stats_df["name"].isin(["Sub"])]
    display_stats = display_stats.sort_values("jersey_number")

    display_cols = ["name", "jersey_number", "minutes_played", "distance_m",
                    "top_speed_ms", "sprint_count", "pct_att_third",
                    "pct_mid_third", "pct_def_third", "passes_made", "id_confidence"]

    st.subheader("My Team")
    if not display_stats.empty:
        display = display_stats[display_cols].copy()
        display.columns = ["Name", "#", "Min", "Dist(m)", "Top Spd",
                          "Sprints", "Att%", "Mid%", "Def%", "Passes", "ID Conf"]
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_heatmaps_tab(detections_df, field_length, field_width):
    """Render the heatmaps tab."""
    if detections_df.empty:
        st.info("No tracking data available.")
        return

    # Filter out Sub aggregates and Unknown for display
    display_df = detections_df[
        ~detections_df["player_name"].isin(["Sub"])
        & ~detections_df["player_name"].str.startswith("Unknown_", na=True)
    ]

    # Player selector
    players = sorted(display_df["player_name"].unique())

    if players:
        selected_player = st.selectbox("Select Player", players)
        player_data = display_df[display_df["player_name"] == selected_player]

        fig = plot_player_heatmap(player_data, selected_player, "Home", field_length, field_width)
        st.pyplot(fig)
        plt.close(fig)

    # Grid of all heatmaps
    st.subheader("All Players")
    fig = plot_all_heatmaps_grid(display_df, "Home", field_length, field_width)
    st.pyplot(fig)
    plt.close(fig)


def _render_passing_tab(results, stats_df, field_length, field_width):
    """Render the passing network tab."""
    st.info("⚠️ Passing network is AI-inferred from movement patterns without dedicated ball tracking. "
            "Directional patterns are reliable; exact pass counts may vary ±20%.")

    detections_df = results["detections_df"]

    # Filter out Sub aggregates for network display
    display_det = detections_df[~detections_df["player_name"].isin(["Sub"])]

    # Player positions for network
    player_avg = display_det.groupby("player_name")[["x_field", "y_field"]].mean()

    st.subheader("Passing Network")
    positions = {}
    for name, pos in player_avg.iterrows():
        positions[name] = (pos["x_field"], pos["y_field"])

    pass_matrix = results.get("home_pass_matrix", pd.DataFrame())
    fig = plot_passing_network(pass_matrix, positions, "Home", field_length, field_width)
    st.pyplot(fig)
    plt.close(fig)

    # Pass matrix
    st.subheader("Pass Matrix")
    if not pass_matrix.empty:
        fig = plot_pass_matrix_table(pass_matrix, "Home")
        st.pyplot(fig)
        plt.close(fig)

    # Top passing combinations
    st.subheader("Top 5 Passing Combinations")
    if not pass_matrix.empty:
        combos = []
        for i in pass_matrix.index:
            for j in pass_matrix.columns:
                if i != j and pass_matrix.loc[i, j] > 0:
                    combos.append({"Passer": i, "Receiver": j,
                                   "Passes": int(pass_matrix.loc[i, j])})
        if combos:
            combos_df = pd.DataFrame(combos).sort_values("Passes", ascending=False).head(5)
            st.dataframe(combos_df, use_container_width=True, hide_index=True)


def _render_formation_tab(results, detections_df, field_length, field_width):
    """Render the formation tab."""
    # Formation timeline
    st.subheader("Formation Timeline")
    fig = plot_formation_timeline(results["home_formation_timeline"], "Home")
    st.pyplot(fig)
    plt.close(fig)

    # Compactness
    st.subheader("Team Compactness")
    fig = plot_compactness_chart(results["home_compactness"], "Home")
    st.pyplot(fig)
    plt.close(fig)

    # Formation snapshots
    st.subheader("Formation Snapshots")
    formation_detector = FormationDetector(players_per_team=results.get("players_per_team", 7))
    duration_s = results["video_meta"]["duration_s"]
    snapshot_times = [0, duration_s * 0.25, duration_s * 0.5, duration_s * 0.75]

    cols = st.columns(4)
    for i, t in enumerate(snapshot_times):
        with cols[i]:
            snapshot = formation_detector.get_formation_snapshot(
                detections_df, "Home", t, window=30.0
            )
            st.markdown(f"**{t/60:.0f} min:** {snapshot['formation']}")

    # Coaching note
    if results["home_formation_timeline"]:
        from collections import Counter
        formations = [t["formation"] for t in results["home_formation_timeline"]]
        most_common = Counter(formations).most_common(1)[0]
        pct = most_common[1] / len(formations) * 100
        st.info(f"📋 **Coaching Note:** Team played {most_common[0]} for {pct:.0f}% of the game.")


def _render_export_tab(results, stats_df, detections_df, field_length, field_width):
    """Render the export tab."""
    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        if not stats_df.empty:
            csv = stats_df.to_csv(index=False)
            st.download_button("📊 Download Stats CSV", csv,
                              file_name="match_stats.csv", mime="text/csv",
                              use_container_width=True)

        # PDF report
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_pdf_report(
                    results["match_info"], stats_df
                )
                st.download_button("⬇️ Download PDF Report", pdf_bytes,
                                  file_name=f"match_report_{results['match_info']['date']}.pdf",
                                  mime="application/pdf",
                                  use_container_width=True)

    with col2:
        # Save to season history
        if st.button("💾 Save to Season History", use_container_width=True):
            match_db = get_match_db()
            match_id = match_db.save_match(
                date=results["match_info"]["date"],
                opponent=results["match_info"]["opponent"],
                result=results["match_info"]["result"],
            )
            if not stats_df.empty:
                match_db.save_player_stats(match_id, stats_df.to_dict("records"))
            st.success("✅ Saved to season history!")

        # Heatmaps ZIP
        if st.button("🔥 Download All Heatmaps (ZIP)", use_container_width=True):
            with st.spinner("Generating heatmaps..."):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    players = detections_df[~detections_df["player_name"].str.startswith("Unknown_")]["player_name"].unique()
                    for player in players:
                        pdata = detections_df[detections_df["player_name"] == player]
                        team = pdata["team"].iloc[0] if not pdata.empty else "Home"
                        fig = plot_player_heatmap(pdata, player, team, field_length, field_width)
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight",
                                    facecolor=fig.get_facecolor())
                        plt.close(fig)
                        img_buffer.seek(0)
                        zf.writestr(f"{player.replace(' ', '_')}_heatmap.png", img_buffer.read())

                zip_buffer.seek(0)
                st.download_button("⬇️ Download Heatmaps ZIP", zip_buffer.getvalue(),
                                  file_name="heatmaps.zip", mime="application/zip",
                                  use_container_width=True)

# --- Render results tabs (after function definitions) ---
if "match_results" in st.session_state:
    _res = st.session_state["match_results"]
    _stats = _res["stats_df"]
    _det = _res["detections_df"]
    _fl = _res["field_length"]
    _fw = _res["field_width"]

    st.header("📊 Results")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "👤 Players", "🔥 Heatmaps",
        "🔗 Passing", "🏟️ Formation", "⬇️ Export"
    ])

    with tab1:
        _render_overview_tab(_res, _stats, _det, _fl, _fw)
    with tab2:
        _render_players_tab(_stats)
    with tab3:
        _render_heatmaps_tab(_det, _fl, _fw)
    with tab4:
        _render_passing_tab(_res, _stats, _fl, _fw)
    with tab5:
        _render_formation_tab(_res, _det, _fl, _fw)
    with tab6:
        _render_export_tab(_res, _stats, _det, _fl, _fw)
