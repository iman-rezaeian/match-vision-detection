"""Multi-segment video processing — analyze multiple clips as one game."""

import cv2
import hashlib
import numpy as np
import pandas as pd
import subprocess
import shutil
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional

from config import TRACK_ID_OFFSET

# Directory for caching segment detection results
CACHE_DIR = os.path.join(tempfile.gettempdir(), "soccer_analysis_cache")

# Max width for pipeline input — disabled: native 4K feeds YOLO for better accuracy
# (YOLO internally resizes to config.YOLO_IMGSZ; full-res frames used for color/face)
# Max width for pipeline input — disabled: native 4K feeds YOLO for better accuracy
# (YOLO internally resizes to config.YOLO_IMGSZ; full-res frames used for color/face)
MAX_INPUT_WIDTH = None  # set to e.g. 1920 to re-enable downscaling


class VideoFrameReader:
    """Dict-like lazy frame reader — seeks into video files on demand.

    Quacks like ``dict[int, np.ndarray]`` so it can replace the old
    pre-extracted ``frames`` dict everywhere (classify_teams, batch_match_video,
    etc.) without any API changes.  Keeps a small LRU cache to avoid redundant
    seeks when multiple tracks share the same frame.
    """

    def __init__(self, segments, detections_df, max_cache: int = 8):
        self._segments = segments
        self._max_cache = max_cache
        self._cache = {}        # frame_num → np.ndarray
        self._cache_order = []  # LRU order
        self._caps = {}         # segment_id → VideoCapture

        # Build frame_num → segment mapping
        self._frame_to_seg = {}
        for seg in segments:
            seg_frames = detections_df.loc[
                detections_df["segment_id"] == seg.segment_id, "frame"
            ].unique()
            for fn in seg_frames:
                self._frame_to_seg[int(fn)] = seg

    # ---- dict-like interface -------------------------------------------
    def __contains__(self, frame_num):
        return int(frame_num) in self._frame_to_seg

    def __getitem__(self, frame_num):
        frame_num = int(frame_num)
        if frame_num in self._cache:
            return self._cache[frame_num]

        seg = self._frame_to_seg.get(frame_num)
        if seg is None:
            raise KeyError(frame_num)

        sid = seg.segment_id
        if sid not in self._caps:
            self._caps[sid] = cv2.VideoCapture(seg.video_path)
        cap = self._caps[sid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            raise KeyError(frame_num)

        # LRU eviction
        self._cache[frame_num] = frame
        self._cache_order.append(frame_num)
        while len(self._cache_order) > self._max_cache:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return frame

    def get(self, frame_num, default=None):
        try:
            return self[frame_num]
        except KeyError:
            return default

    def keys(self):
        return self._frame_to_seg.keys()

    def __len__(self):
        return len(self._frame_to_seg)

    def __iter__(self):
        return iter(self._frame_to_seg)

    def __del__(self):
        self.close()

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._cache.clear()
        self._cache_order.clear()


@dataclass
class GameSegment:
    """Metadata for one video clip within a game."""
    video_path: str
    segment_id: int
    label: str                     # "Half 1", "Half 2", etc.
    kickoff_s: float = 0.0        # trim start (seconds into this clip)
    whistle_s: float = 0.0        # trim end (seconds into this clip)
    fps: float = 30.0
    total_frames: int = 0
    frame_w: int = 0
    frame_h: int = 0
    duration_s: float = 0.0       # raw clip duration

    @property
    def kickoff_frame(self) -> int:
        return int(self.kickoff_s * self.fps)

    @property
    def whistle_frame(self) -> int:
        return int(self.whistle_s * self.fps)

    @property
    def gameplay_duration_s(self) -> float:
        return self.whistle_s - self.kickoff_s

    def read_video_info(self):
        """Populate fps, total_frames, dimensions from the video file."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_s = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
        # Default whistle to end of clip if not set
        if self.whistle_s <= 0:
            self.whistle_s = self.duration_s


def downscale_video(video_path: str, max_width: int = MAX_INPUT_WIDTH,
                    progress_callback=None) -> str:
    """
    Downscale a video to max_width if it's wider.  Uses ffmpeg with hardware
    encoding (VideoToolbox on macOS) when available, falls back to libx264.

    Returns path to the downscaled temp file, or the original path if
    no downscale was needed.
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if w <= max_width:
        return video_path  # already small enough

    if not shutil.which("ffmpeg"):
        # No ffmpeg — skip downscale, pipeline will still work (just slower)
        return video_path

    # Build output path in system temp dir (avoids OneDrive sync issues)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base}_1080p.mp4")

    # Skip if already transcoded
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        return out_path

    # Try hardware-accelerated encoder first, fall back to libx264
    scale_filter = f"scale={max_width}:-2"
    encoders = [
        ["h264_videotoolbox", "-b:v", "8M"],   # macOS HW
        ["libx264", "-preset", "fast", "-crf", "18"],  # CPU fallback
    ]

    for enc_args in encoders:
        encoder = enc_args[0]
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", scale_filter,
            "-c:v", encoder, *enc_args[1:],
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _, stderr = proc.communicate()
            if proc.returncode == 0:
                return out_path
        except Exception:
            continue

    # All encoders failed — use original
    return video_path


def get_frame_at_time(video_path: str, time_s: float) -> Optional[np.ndarray]:
    """Extract a single frame at a given timestamp for UI preview."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class MultiSegmentProcessor:
    """Orchestrates detection across multiple video segments and merges results."""

    def __init__(self, detector):
        """
        Args:
            detector: a VideoDetector instance (reused across segments)
        """
        self.detector = detector

    @staticmethod
    def _segment_cache_key(seg: GameSegment, match_key: str = "") -> str:
        """Build a unique cache key for a segment based on path + match identity."""
        raw = f"{seg.video_path}|{seg.segment_id}|{match_key}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _match_cache_key(segments: List[GameSegment], match_key: str = "") -> str:
        """Build a cache key for the full match (video files + match identity)."""
        paths = sorted(set(seg.video_path for seg in segments))
        raw = "|".join(paths) + f"||{match_key}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def load_stage1_cache(segments: List[GameSegment], match_key: str = ""):
        """Try to load cached Stage 1 results. Returns (detections_df, video_meta) or None."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = MultiSegmentProcessor._match_cache_key(segments, match_key)
        df_path = os.path.join(CACHE_DIR, f"stage1_{key}.parquet")
        meta_path = os.path.join(CACHE_DIR, f"stage1_{key}_meta.npy")
        if os.path.exists(df_path) and os.path.exists(meta_path):
            try:
                df = pd.read_parquet(df_path)
                meta = np.load(meta_path, allow_pickle=True).item()
                return df, meta
            except Exception:
                pass
        return None

    @staticmethod
    def save_stage1_cache(segments: List[GameSegment], detections_df: pd.DataFrame, video_meta: dict,
                          match_key: str = ""):
        """Persist Stage 1 results to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = MultiSegmentProcessor._match_cache_key(segments, match_key)
        df_path = os.path.join(CACHE_DIR, f"stage1_{key}.parquet")
        meta_path = os.path.join(CACHE_DIR, f"stage1_{key}_meta.npy")
        detections_df.to_parquet(df_path, index=False)
        np.save(meta_path, video_meta)

    @staticmethod
    def load_stage2_cache(match_key: str):
        """Try to load cached Stage 2 results (detections with teams + team colors).
        Returns (detections_df, team_colors_dict) or None."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5(match_key.encode()).hexdigest()
        df_path = os.path.join(CACHE_DIR, f"stage2_{key}.parquet")
        colors_path = os.path.join(CACHE_DIR, f"stage2_{key}_colors.npy")
        if os.path.exists(df_path) and os.path.exists(colors_path):
            try:
                df = pd.read_parquet(df_path)
                colors = np.load(colors_path, allow_pickle=True).item()
                return df, colors
            except Exception:
                pass
        return None

    @staticmethod
    def save_stage2_cache(match_key: str, detections_df: pd.DataFrame, team_colors: dict):
        """Persist Stage 2 results (detections with team labels + team colors) to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5(match_key.encode()).hexdigest()
        df_path = os.path.join(CACHE_DIR, f"stage2_{key}.parquet")
        colors_path = os.path.join(CACHE_DIR, f"stage2_{key}_colors.npy")
        detections_df.to_parquet(df_path, index=False)
        np.save(colors_path, team_colors)

    @staticmethod
    def load_stage3_cache(match_key: str):
        """Try to load cached Stage 3 results (detections with player IDs).
        Returns detections_df or None."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5(match_key.encode()).hexdigest()
        df_path = os.path.join(CACHE_DIR, f"stage3_{key}.parquet")
        if os.path.exists(df_path):
            try:
                df = pd.read_parquet(df_path)
                return df
            except Exception:
                pass
        return None

    @staticmethod
    def save_stage3_cache(match_key: str, detections_df: pd.DataFrame):
        """Persist Stage 3 results (detections with player IDs) to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5(match_key.encode()).hexdigest()
        df_path = os.path.join(CACHE_DIR, f"stage3_{key}.parquet")
        # Coerce mixed-type columns to string for parquet compatibility
        df = detections_df.copy()
        for col in ["player_id", "player_name", "jersey_number"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        df.to_parquet(df_path, index=False)

    @staticmethod
    def load_stage45_cache(match_key: str):
        """Try to load cached Stage 4-5 results (stats, passes, formations).
        Returns dict with all results or None."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5(match_key.encode()).hexdigest()
        stats_path = os.path.join(CACHE_DIR, f"stage4_{key}_stats.parquet")
        data_path = os.path.join(CACHE_DIR, f"stage4_{key}_data.npy")
        if os.path.exists(stats_path) and os.path.exists(data_path):
            try:
                stats_df = pd.read_parquet(stats_path)
                data = np.load(data_path, allow_pickle=True).item()
                data["stats_df"] = stats_df
                return data
            except Exception:
                pass
        return None

    @staticmethod
    def save_stage45_cache(match_key: str, stats_df: pd.DataFrame,
                           passes, home_pass_matrix, away_pass_matrix,
                           home_formation_timeline, away_formation_timeline,
                           home_compactness, away_compactness):
        """Persist Stage 4-5 results to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5(match_key.encode()).hexdigest()
        stats_path = os.path.join(CACHE_DIR, f"stage4_{key}_stats.parquet")
        data_path = os.path.join(CACHE_DIR, f"stage4_{key}_data.npy")
        stats_df.to_parquet(stats_path, index=False)
        np.save(data_path, {
            "passes": passes,
            "home_pass_matrix": home_pass_matrix,
            "away_pass_matrix": away_pass_matrix,
            "home_formation_timeline": home_formation_timeline,
            "away_formation_timeline": away_formation_timeline,
            "home_compactness": home_compactness,
            "away_compactness": away_compactness,
        })

    def process_segments(self, segments: List[GameSegment],
                         progress_callback=None) -> tuple:
        """
        Run detection+tracking on each segment, merge into unified DataFrame.

        Args:
            segments: list of GameSegment with trim points set
            progress_callback: optional (segment_id, segment_label, stage_msg) callback

        Returns:
            merged_df: unified detections DataFrame with segment_id, segment_time_s
            merged_meta: aggregated video metadata
        """
        all_dfs = []
        time_offset = 0.0
        total_gameplay = sum(seg.gameplay_duration_s for seg in segments)
        match_key = self._match_key if hasattr(self, '_match_key') else ""

        # Optionally downscale 4K+ segments (disabled by default for accuracy)
        if MAX_INPUT_WIDTH is not None:
            for seg in segments:
                if seg.frame_w > MAX_INPUT_WIDTH:
                    if progress_callback:
                        progress_callback(
                            seg.segment_id, seg.label,
                            f"Downscaling {seg.label} from {seg.frame_w}×{seg.frame_h} → {MAX_INPUT_WIDTH}p "
                            f"(one-time, speeds up all stages)..."
                        )
                    new_path = downscale_video(seg.video_path, max_width=MAX_INPUT_WIDTH)
                    if new_path != seg.video_path:
                        seg.video_path = new_path
                        seg.read_video_info()  # refresh fps/frame_count/dimensions

        # Pre-compute total frames across all segments for overall progress
        total_frames_all = sum(
            (seg.whistle_frame or 0) - (seg.kickoff_frame or 0)
            for seg in segments
        )
        frames_done_before = 0

        for seg_idx, seg in enumerate(segments):
            seg_total = (seg.whistle_frame or 0) - (seg.kickoff_frame or 0)

            # Check per-segment cache
            os.makedirs(CACHE_DIR, exist_ok=True)
            seg_key = self._segment_cache_key(seg, match_key)
            seg_cache_path = os.path.join(CACHE_DIR, f"seg_{seg_key}.parquet")

            if os.path.exists(seg_cache_path):
                try:
                    det_df = pd.read_parquet(seg_cache_path)
                    if not det_df.empty:
                        if progress_callback:
                            progress_callback(
                                seg.segment_id, seg.label,
                                f"{seg.label}: loaded from cache ({len(det_df)} detections) — "
                                f"overall {frames_done_before + seg_total}/{total_frames_all}"
                            )
                        all_dfs.append(det_df)
                        frames_done_before += seg_total
                        time_offset += seg.gameplay_duration_s
                        continue
                except Exception:
                    pass  # cache corrupt — re-process

            if progress_callback:
                progress_callback(
                    seg.segment_id, seg.label,
                    f"Processing {seg.label} ({seg.gameplay_duration_s:.0f}s gameplay)"
                )

            # Build per-frame callback that feeds into overall progress
            def make_frame_cb(seg_offset, seg_label):
                def frame_cb(frame_done, frame_total, n_dets):
                    if progress_callback:
                        overall_done = seg_offset + frame_done
                        progress_callback(
                            seg.segment_id, seg_label,
                            f"{seg_label}: frame {frame_done}/{frame_total} "
                            f"({n_dets} detections) — "
                            f"overall {overall_done}/{total_frames_all}"
                        )
                return frame_cb

            # Reset tracker for each segment (fresh ByteTrack state)
            self.detector.tracker.reset()

            # Run detection on trimmed range
            det_df, meta = self.detector.process(
                seg.video_path,
                start_frame=seg.kickoff_frame,
                end_frame=seg.whistle_frame,
                progress_callback=make_frame_cb(frames_done_before, seg.label),
            )

            frames_done_before += seg_total

            if det_df.empty:
                continue

            # Add segment metadata columns
            det_df["segment_id"] = seg.segment_id
            det_df["segment_label"] = seg.label
            det_df["segment_time_s"] = det_df["time_s"]  # local time within clip

            # Offset track IDs to avoid collisions across segments
            det_df["track_id"] = det_df["track_id"] + (seg.segment_id * TRACK_ID_OFFSET)

            # Rebase time_s to continuous game clock
            # time_s within the segment is relative to kickoff_frame
            det_df["time_s"] = (det_df["frame"] - seg.kickoff_frame) / seg.fps + time_offset

            all_dfs.append(det_df)
            time_offset += seg.gameplay_duration_s

            # Save segment to disk cache
            try:
                det_df.to_parquet(seg_cache_path, index=False)
            except Exception:
                pass  # non-critical

        if not all_dfs:
            return pd.DataFrame(), self._empty_meta()

        merged_df = pd.concat(all_dfs, ignore_index=True)

        merged_meta = self._aggregate_meta(segments, total_gameplay)

        # Save full Stage 1 result to disk
        self.save_stage1_cache(segments, merged_df, merged_meta, match_key)

        return merged_df, merged_meta

    def extract_frames_for_tracks(self, segments: List[GameSegment],
                                  detections_df: pd.DataFrame,
                                  sample_every: int = 10,
                                  progress_callback=None) -> dict:
        """
        Extract frames across all segments for fingerprinting.
        Reads at native resolution (no downscale) for best color/face accuracy.

        Returns: {frame_number: np.ndarray}
        """
        frames = {}

        # Collect all frame numbers needed across segments
        all_frame_tasks = []
        for seg in segments:
            seg_df = detections_df[detections_df["segment_id"] == seg.segment_id]
            if seg_df.empty:
                continue
            sampled = seg_df.groupby("track_id").apply(
                lambda g: g.iloc[::sample_every]
            ).reset_index(drop=True)
            frame_numbers = sorted(sampled["frame"].unique())
            all_frame_tasks.append((seg, frame_numbers))

        total_frames = sum(len(fns) for _, fns in all_frame_tasks)
        done = 0

        for seg, frame_numbers in all_frame_tasks:
            cap = cv2.VideoCapture(seg.video_path)
            for fn in frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if ret:
                    frames[fn] = frame
                done += 1
                if progress_callback and done % 20 == 0:
                    progress_callback(done, total_frames)
            cap.release()

        return frames

    def _aggregate_meta(self, segments: List[GameSegment],
                        total_gameplay: float) -> dict:
        """Build merged video_meta dict from all segments."""
        return {
            "fps": segments[0].fps if segments else 30.0,
            "total_frames": sum(seg.whistle_frame - seg.kickoff_frame for seg in segments),
            "duration_s": total_gameplay,
            "frame_h": segments[0].frame_h if segments else 0,
            "frame_w": segments[0].frame_w if segments else 0,
            "n_segments": len(segments),
            "segments": [
                {
                    "segment_id": seg.segment_id,
                    "label": seg.label,
                    "video_path": seg.video_path,
                    "kickoff_s": seg.kickoff_s,
                    "whistle_s": seg.whistle_s,
                    "gameplay_duration_s": seg.gameplay_duration_s,
                }
                for seg in segments
            ],
        }

    def _empty_meta(self) -> dict:
        return {
            "fps": 30.0, "total_frames": 0, "duration_s": 0,
            "frame_h": 0, "frame_w": 0, "n_segments": 0, "segments": [],
        }
