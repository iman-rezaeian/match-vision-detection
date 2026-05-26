"""Phase 3C — Enhanced tracker with appearance-based re-identification.

Provides a wrapper around ByteTrack that adds simple appearance features
to reduce ID switches when players reappear after brief occlusion.

Uses color histograms as cheap appearance descriptors — no heavy
ReID models needed. For production, upgrade to BoT-SORT with OSNet.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class AppearanceTracker:
    """
    ByteTrack wrapper with appearance-based ID correction.

    After ByteTrack assigns IDs, this post-processor checks if
    newly created tracks match recently lost tracks by appearance.
    If so, it corrects the ID to reduce fragmentation.
    """

    def __init__(self, max_lost_frames: int = 60,
                 appearance_threshold: float = 0.7,
                 hist_bins: int = 32):
        """
        Args:
            max_lost_frames: How many frames to remember lost tracks
            appearance_threshold: Min histogram correlation to re-ID
            hist_bins: Number of bins per channel for color histogram
        """
        import supervision as sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=150,
            minimum_matching_threshold=0.9,
        )

        self.max_lost_frames = max_lost_frames
        self.appearance_threshold = appearance_threshold
        self.hist_bins = hist_bins

        # Active track appearances: track_id -> histogram
        self.active_appearances: Dict[int, np.ndarray] = {}
        # Lost tracks: track_id -> (histogram, last_frame, last_position)
        self.lost_tracks: Dict[int, tuple] = {}
        # ID remapping: bytetrack_id -> corrected_id
        self.id_remap: Dict[int, int] = {}
        # Track seen history: track_id -> first_seen_frame
        self.first_seen: Dict[int, int] = {}

        self.frame_count = 0

    def update(self, frame: np.ndarray,
               detections) -> 'Detections':
        """
        Update tracker with detections and apply appearance-based ID correction.

        Args:
            frame: Current video frame
            detections: supervision.Detections object

        Returns:
            Updated detections with corrected tracker IDs
        """
        import supervision as sv

        # Run ByteTrack
        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            self.frame_count += 1
            self._clean_lost_tracks()
            return tracked

        # Compute appearance for each tracked detection
        current_ids = set()
        for i, track_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]

            if crop.size == 0:
                continue

            hist = self._compute_histogram(crop)

            # Check if this is a new track (not seen before or gap > 5 frames)
            bt_id = int(track_id)
            is_new = bt_id not in self.first_seen

            if is_new:
                self.first_seen[bt_id] = self.frame_count
                # Try to match against lost tracks
                best_match = self._find_best_lost_match(hist, tracked.xyxy[i])
                if best_match is not None:
                    self.id_remap[bt_id] = best_match
                    # Remove from lost tracks
                    if best_match in self.lost_tracks:
                        del self.lost_tracks[best_match]

            # Update appearance model
            actual_id = self.id_remap.get(bt_id, bt_id)
            self.active_appearances[actual_id] = hist
            current_ids.add(actual_id)

        # Detect lost tracks
        prev_active = set(self.active_appearances.keys())
        lost_this_frame = prev_active - current_ids
        for lost_id in lost_this_frame:
            if lost_id in self.active_appearances:
                # Find last known position
                self.lost_tracks[lost_id] = (
                    self.active_appearances[lost_id],
                    self.frame_count,
                    None,  # position not tracked here
                )
                del self.active_appearances[lost_id]

        # Apply ID remapping to the detections
        if self.id_remap:
            new_ids = []
            for tid in tracked.tracker_id:
                bt_id = int(tid)
                new_ids.append(self.id_remap.get(bt_id, bt_id))
            tracked.tracker_id = np.array(new_ids)

        self.frame_count += 1
        self._clean_lost_tracks()

        return tracked

    def _compute_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram as appearance descriptor."""
        # Resize to fixed size for consistency
        crop_resized = cv2.resize(crop, (32, 64))

        # Use HSV for better color invariance
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        # Compute histogram for H and S channels (ignore V for lighting invariance)
        hist_h = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])

        # Concatenate and normalize
        hist = np.concatenate([hist_h, hist_s]).flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()

        return hist

    def _find_best_lost_match(self, hist: np.ndarray,
                               bbox: np.ndarray) -> Optional[int]:
        """Find the best matching lost track by appearance."""
        best_score = -1
        best_id = None

        for lost_id, (lost_hist, lost_frame, _) in self.lost_tracks.items():
            # Skip if too old
            if self.frame_count - lost_frame > self.max_lost_frames:
                continue

            # Compare histograms using correlation
            score = cv2.compareHist(
                hist.reshape(-1, 1).astype(np.float32),
                lost_hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL,
            )

            if score > best_score and score > self.appearance_threshold:
                best_score = score
                best_id = lost_id

        return best_id

    def _clean_lost_tracks(self):
        """Remove lost tracks that are too old."""
        to_remove = []
        for lost_id, (_, lost_frame, _) in self.lost_tracks.items():
            if self.frame_count - lost_frame > self.max_lost_frames:
                to_remove.append(lost_id)
        for rid in to_remove:
            del self.lost_tracks[rid]

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "active_tracks": len(self.active_appearances),
            "lost_tracks": len(self.lost_tracks),
            "id_remaps": len(self.id_remap),
            "total_ids_seen": len(self.first_seen),
        }
