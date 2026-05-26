"""Team-color-aware multi-object tracker.

Runs a separate ByteTrack instance per team so that IDs can never
swap across teams. Jersey color is extracted from the upper-body
region of each detection and classified via K-means clustering
(auto-discovered on the first N frames) or via pre-set team colors.

This eliminates cross-team ID switches entirely and reduces
within-team switches by shrinking each tracker's matching pool.
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class TeamTracker:
    """
    Multi-team tracker: classifies each detection by jersey color,
    then tracks each team independently with its own ByteTrack.
    """

    def __init__(self, n_teams: int = 2,
                 warmup_frames: int = 30,
                 lost_track_buffer: int = 150,
                 track_activation_threshold: float = 0.30,
                 minimum_matching_threshold: float = 0.9):
        """
        Args:
            n_teams: Number of teams (2 + referees handled separately)
            warmup_frames: Frames to collect colors before clustering
            lost_track_buffer: ByteTrack lost-track buffer per tracker
            track_activation_threshold: Min confidence to activate track
            minimum_matching_threshold: IoU threshold for ByteTrack matching
        """
        import supervision as sv
        self._sv = sv

        self.n_teams = n_teams
        self.warmup_frames = warmup_frames
        self.frame_count = 0

        # Per-team ByteTrack instances (created after warmup)
        self._trackers: Dict[int, object] = {}
        self._tracker_params = {
            "lost_track_buffer": lost_track_buffer,
            "track_activation_threshold": track_activation_threshold,
            "minimum_matching_threshold": minimum_matching_threshold,
        }

        # Fallback single tracker used during warmup
        self._warmup_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

        # Color samples collected during warmup
        self._color_samples: List[np.ndarray] = []
        # Cluster centers (set after warmup)
        self._team_centers: Optional[np.ndarray] = None
        # Team ID offset so team tracker IDs don't collide
        # Team 0 → IDs 0-9999, Team 1 → 10000-19999, Team 2 → 20000+
        self._id_offset = 10000

    def update(self, frame: np.ndarray, detections) -> object:
        """
        Classify detections by team color, then track per team.

        Args:
            frame: BGR video frame
            detections: supervision.Detections with .xyxy, .confidence, .class_id

        Returns:
            supervision.Detections with .tracker_id set (team-aware)
        """
        sv = self._sv

        if detections is None or len(detections) == 0:
            self.frame_count += 1
            return detections

        # Extract jersey color for each detection
        colors = self._extract_jersey_colors(frame, detections.xyxy)

        # During warmup: collect color samples, use single tracker
        if self._team_centers is None:
            self._color_samples.extend(colors)
            self.frame_count += 1

            if self.frame_count >= self.warmup_frames and len(self._color_samples) > 20:
                self._fit_team_clusters()

            # Use fallback single tracker during warmup
            tracked = self._warmup_tracker.update_with_detections(detections)
            return tracked

        # Classify each detection into a team
        team_labels = self._classify_teams(colors)

        # Split detections by team and track independently
        all_xyxy = []
        all_conf = []
        all_class_id = []
        all_tracker_id = []

        unique_teams = np.unique(team_labels)
        for team_id in unique_teams:
            team_id = int(team_id)
            mask = team_labels == team_id
            if not np.any(mask):
                continue

            # Create sub-detections for this team
            team_xyxy = detections.xyxy[mask]
            team_conf = detections.confidence[mask] if detections.confidence is not None else np.ones(mask.sum())
            team_class = detections.class_id[mask] if detections.class_id is not None else np.zeros(mask.sum(), dtype=int)

            team_det = sv.Detections(
                xyxy=team_xyxy,
                confidence=team_conf,
                class_id=team_class,
            )

            # Get or create per-team tracker
            if team_id not in self._trackers:
                self._trackers[team_id] = sv.ByteTrack(
                    **self._tracker_params
                )

            tracked_team = self._trackers[team_id].update_with_detections(team_det)

            if tracked_team.tracker_id is not None and len(tracked_team.tracker_id) > 0:
                # Offset IDs to prevent collisions across teams
                offset_ids = tracked_team.tracker_id + (team_id * self._id_offset)
                all_xyxy.append(tracked_team.xyxy)
                all_conf.append(tracked_team.confidence if tracked_team.confidence is not None else np.ones(len(tracked_team)))
                all_class_id.append(np.full(len(tracked_team), team_id, dtype=int))
                all_tracker_id.append(offset_ids)

        self.frame_count += 1

        if not all_xyxy:
            return sv.Detections.empty()

        # Merge all teams back into one Detections object
        merged = sv.Detections(
            xyxy=np.vstack(all_xyxy),
            confidence=np.concatenate(all_conf),
            class_id=np.concatenate(all_class_id),
        )
        merged.tracker_id = np.concatenate(all_tracker_id).astype(int)

        return merged

    def _extract_jersey_colors(self, frame: np.ndarray,
                                boxes: np.ndarray) -> List[np.ndarray]:
        """
        Extract the dominant jersey color from each detection.

        Focuses on the upper 40-70% of the bbox (torso area)
        to avoid shorts/shoes and head/hair.
        Returns a list of HSV color vectors.
        """
        colors = []
        h_frame, w_frame = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            bh = y2 - y1
            bw = x2 - x1
            if bh < 4 or bw < 4:
                colors.append(np.array([0, 0, 128], dtype=np.float32))
                continue

            # Upper-body crop: 20%-60% of height (torso/jersey)
            torso_y1 = y1 + int(bh * 0.20)
            torso_y2 = y1 + int(bh * 0.60)
            # Inset sides 10% to avoid arms/background
            torso_x1 = x1 + int(bw * 0.10)
            torso_x2 = x2 - int(bw * 0.10)

            crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
            if crop.size == 0:
                colors.append(np.array([0, 0, 128], dtype=np.float32))
                continue

            # Convert to HSV and compute mean color
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_color = hsv.reshape(-1, 3).astype(np.float32).mean(axis=0)
            colors.append(mean_color)

        return colors

    def _fit_team_clusters(self):
        """
        Fit K-means on collected jersey colors to discover team clusters.
        Uses n_teams + 1 clusters (extra for referees/goalkeepers).
        """
        samples = np.array(self._color_samples, dtype=np.float32)
        if len(samples) < 10:
            return

        # Normalize: H channel has range 0-180, S and V are 0-255
        # Weight H heavily since it captures color, S moderately, V less
        weights = np.array([2.0, 1.0, 0.3], dtype=np.float32)
        weighted = samples * weights

        n_clusters = self.n_teams + 1  # +1 for referees/outliers
        n_clusters = min(n_clusters, len(samples))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, labels, centers = cv2.kmeans(
            weighted, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Un-weight centers for classification
        self._team_centers = centers / weights
        self._cluster_weights = weights

        # Report discovered teams
        print(f"    Team color clustering: {n_clusters} groups discovered")
        for i, c in enumerate(self._team_centers):
            h, s, v = c
            print(f"      Group {i}: H={h:.0f} S={s:.0f} V={v:.0f} "
                  f"({self._hsv_to_name(h, s, v)})")

        # Clear samples to free memory
        self._color_samples = []

    def _classify_teams(self, colors: List[np.ndarray]) -> np.ndarray:
        """Assign each detection to the nearest team cluster."""
        if not colors:
            return np.array([], dtype=int)

        samples = np.array(colors, dtype=np.float32)
        weighted = samples * self._cluster_weights
        centers_weighted = self._team_centers * self._cluster_weights

        # Compute distances to each cluster center
        labels = np.zeros(len(samples), dtype=int)
        for i, s in enumerate(weighted):
            dists = np.linalg.norm(centers_weighted - s, axis=1)
            labels[i] = np.argmin(dists)

        return labels

    @staticmethod
    def _hsv_to_name(h: float, s: float, v: float) -> str:
        """Convert HSV to approximate color name for logging."""
        if s < 40:
            if v < 80:
                return "black"
            elif v > 200:
                return "white"
            else:
                return "gray"
        if h < 10 or h > 170:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 85:
            return "green"
        elif h < 130:
            return "blue"
        elif h < 170:
            return "purple"
        return "unknown"

    def get_stats(self) -> dict:
        return {
            "teams_discovered": len(self._trackers),
            "warmup_complete": self._team_centers is not None,
            "frames_processed": self.frame_count,
        }
