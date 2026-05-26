"""YOLO detection + centroid-distance tracker for high sample rates."""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from config import (
    DEVICE, DEFAULT_CONFIDENCE, DEFAULT_SAMPLE_RATE, YOLO_IMGSZ,
    DEFAULT_TRACKER, YOLO_VERSION,
)


@st.cache_resource
def load_yolo_model(model_size: str = "s", yolo_version: str = "8"):
    """Cache YOLO model across Streamlit reruns."""
    if yolo_version == "11":
        model = YOLO(f"yolo11{model_size}.pt")
    else:
        model = YOLO(f"yolov8{model_size}.pt")
    return model


class CentroidTracker:
    """Simple centroid-distance tracker using Hungarian algorithm.
    
    Works well with high sample rates where IoU-based trackers fail
    because players move too far between sampled frames.
    """
    
    def __init__(self, max_distance: float = 200, max_lost: int = 15):
        """
        Args:
            max_distance: max pixel distance to match a detection to a track
            max_lost: frames a track persists without a match before removal
        """
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # track_id -> {"centroid": (x,y), "bbox": array, "lost": int}
    
    def update(self, xyxy: np.ndarray) -> list:
        """Update tracks with new detections.
        
        Args:
            xyxy: Nx4 array of [x1, y1, x2, y2]
            
        Returns:
            List of (track_id, bbox) for matched/new detections
        """
        # Compute centroids of new detections
        centroids = np.column_stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
        ])
        
        results = []
        
        if not self.tracks:
            # No existing tracks — create new ones for all detections
            for i in range(len(xyxy)):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"centroid": centroids[i], "bbox": xyxy[i], "lost": 0}
                results.append((tid, xyxy[i]))
            return results
        
        # Build cost matrix: distance from each track to each detection
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid]["centroid"] for tid in track_ids])
        
        # Euclidean distance matrix
        diff = track_centroids[:, None, :] - centroids[None, :, :]
        cost_matrix = np.sqrt((diff ** 2).sum(axis=2))
        
        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] <= self.max_distance:
                tid = track_ids[r]
                self.tracks[tid] = {"centroid": centroids[c], "bbox": xyxy[c], "lost": 0}
                results.append((tid, xyxy[c]))
                matched_tracks.add(tid)
                matched_dets.add(c)
        
        # Increment lost counter for unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1
        
        # Remove tracks that have been lost too long
        to_remove = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in to_remove:
            del self.tracks[tid]
        
        # Create new tracks for unmatched detections
        for i in range(len(xyxy)):
            if i not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"centroid": centroids[i], "bbox": xyxy[i], "lost": 0}
                results.append((tid, xyxy[i]))
        
        return results


class VideoDetector:
    # Minimum bbox area in pixels at 4K — filters birds, shadows, line marks
    MIN_BBOX_AREA = 400  # ~20x20 px

    def __init__(self, model_size: str = "s", confidence: float = DEFAULT_CONFIDENCE,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 tracker_type: str = DEFAULT_TRACKER,
                 yolo_version: str = YOLO_VERSION):
        self.model = load_yolo_model(model_size, yolo_version)
        self.model_size = model_size
        self.tracker_type = "centroid"
        self.confidence = confidence
        self.sample_rate = sample_rate

        # Max distance scales with sample_rate: higher rate = more movement between frames
        # At 4K (3840px wide), a player running fast covers ~150px per 10-frame gap
        max_distance = 80 * sample_rate  # e.g. sample_rate=10 → 800px max match distance
        max_distance = min(max_distance, 1200)  # cap at reasonable value
        
        # Tracks persist for ~10s of video without a match
        max_lost = int(300 / sample_rate)  # 300 raw frames = 10s at 30fps
        
        self.tracker = CentroidTracker(max_distance=max_distance, max_lost=max_lost)

    def process(self, video_path: str, progress_callback=None,
                start_frame: int = 0, end_frame: int = None):
        """
        Process video, detect all persons, track with consistent IDs per stint.

        Args:
            video_path: path to video file
            progress_callback: optional (frame_id, total, n_detections) callback
            start_frame: first frame to process (skip earlier frames)
            end_frame: last frame to process (stop after this frame)

        Returns:
            detections_df: DataFrame with columns:
                frame | time_s | track_id | x_px | y_px |
                bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 |
                frame_h | frame_w | fps
            video_meta: dict with fps, total_frames, duration_s, frame_h, frame_w
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0

        video_meta = {
            "fps": fps,
            "total_frames": total_frames,
            "duration_s": duration_s,
            "frame_h": frame_h,
            "frame_w": frame_w,
        }

        if end_frame is None:
            end_frame = total_frames

        # Seek to start_frame if skipping early frames
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        all_detections = []
        frame_id = start_frame

        while True:
            if frame_id > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % self.sample_rate == 0:
                # Run YOLO detection (person class only)
                results = self.model(
                    frame,
                    conf=self.confidence,
                    classes=[0],  # person only
                    device=DEVICE,
                    imgsz=YOLO_IMGSZ,
                    verbose=False
                )

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    confidence_scores = boxes.conf.cpu().numpy()

                    # Filter out tiny detections (shadows, birds, line marks)
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    valid = areas >= self.MIN_BBOX_AREA
                    xyxy = xyxy[valid]
                    confidence_scores = confidence_scores[valid]

                    if len(xyxy) == 0:
                        frame_id += 1
                        continue

                    # Centroid-distance tracker — robust to high sample rates
                    tracked = self.tracker.update(xyxy)
                    time_s = frame_id / fps if fps > 0 else 0
                    for track_id, bbox in tracked:
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        all_detections.append({
                            "frame": frame_id,
                            "time_s": round(time_s, 3),
                            "track_id": int(track_id),
                            "x_px": float(x_center),
                            "y_px": float(y_center),
                            "bbox_x1": float(bbox[0]),
                            "bbox_y1": float(bbox[1]),
                            "bbox_x2": float(bbox[2]),
                            "bbox_y2": float(bbox[3]),
                            "frame_h": frame_h,
                            "frame_w": frame_w,
                            "fps": fps,
                        })

            frame_id += 1

            # Progress callback every sampled frame (real-time Streamlit updates)
            if progress_callback and frame_id % self.sample_rate == 0:
                progress_callback(frame_id - start_frame, end_frame - start_frame, len(all_detections))

        cap.release()

        detections_df = pd.DataFrame(all_detections)
        return detections_df, video_meta

    def extract_frame(self, video_path: str, frame_number: int) -> np.ndarray:
        """Extract a single frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_number}")
        return frame

    def extract_frames_for_tracks(self, video_path: str, detections_df: pd.DataFrame,
                                  sample_every: int = 10) -> dict:
        """
        Extract frames for each track_id at regular intervals.
        Returns: {frame_number: np.ndarray}
        """
        # Get unique frames needed
        sampled = detections_df.groupby("track_id").apply(
            lambda g: g.iloc[::sample_every]
        ).reset_index(drop=True)

        frame_numbers = sorted(sampled["frame"].unique())
        frames = {}

        cap = cv2.VideoCapture(video_path)
        for fn in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if ret:
                # Downscale to save memory (keep aspect ratio, max 1280px wide)
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)))
                frames[fn] = frame
        cap.release()

        return frames
