"""MediaPipe pose-based gait signature extraction."""

import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None
from typing import Optional
from scipy.fft import fft
from scipy.signal import find_peaks
from config import MIN_GAIT_FRAMES


class GaitAnalyzer:
    def __init__(self):
        if mp is None:
            self.pose = None
            return
        # mediapipe >=0.10.14 removed mp.solutions; use mp.tasks instead
        if hasattr(mp, 'solutions'):
            self.pose = mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._legacy = True
        else:
            # New API not yet wired — degrade gracefully
            self.pose = None
            self._legacy = False

    def extract_keypoints(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """
        Crop player region, run MediaPipe Pose.
        Return 33 normalized keypoint coords as flat array (99 values: x,y,visibility).
        Return None if pose not detected.
        """
        if self.pose is None:
            return None

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks is None:
            return None

        # Extract normalized keypoints
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.visibility])

        return np.array(keypoints, dtype=np.float32)

    def build_gait_signature(self, keypoint_sequence: list) -> Optional[np.ndarray]:
        """
        Input: list of keypoint arrays over 60+ frames (~2 seconds at sample_rate=3)

        Extract gait features:
        - Stride frequency (dominant frequency of ankle keypoint oscillation)
        - Step width (average lateral distance between ankles)
        - Arm swing amplitude (wrist keypoint oscillation)
        - Body lean angle (shoulder-to-hip vector angle during running)
        - Cadence (steps per second)
        - Knee lift height (normalized knee height during stride)

        Return 128-dim gait signature vector.
        """
        if len(keypoint_sequence) < MIN_GAIT_FRAMES:
            return None

        kp_array = np.array(keypoint_sequence)  # shape: (N, 99)

        # MediaPipe landmark indices (x=idx*3, y=idx*3+1, vis=idx*3+2)
        # Left ankle: 27, Right ankle: 28
        # Left knee: 25, Right knee: 26
        # Left hip: 23, Right hip: 24
        # Left shoulder: 11, Right shoulder: 12
        # Left wrist: 15, Right wrist: 16

        features = []

        # 1. Ankle oscillation (stride frequency)
        left_ankle_y = kp_array[:, 27 * 3 + 1]
        right_ankle_y = kp_array[:, 28 * 3 + 1]

        # FFT of ankle movement
        left_fft = np.abs(fft(left_ankle_y - np.mean(left_ankle_y)))[:len(left_ankle_y) // 2]
        right_fft = np.abs(fft(right_ankle_y - np.mean(right_ankle_y)))[:len(right_ankle_y) // 2]

        # Pad/truncate to fixed size (16 features each)
        features.extend(self._normalize_vector(left_fft, 16))
        features.extend(self._normalize_vector(right_fft, 16))

        # 2. Step width (lateral distance between ankles)
        left_ankle_x = kp_array[:, 27 * 3]
        right_ankle_x = kp_array[:, 28 * 3]
        step_widths = np.abs(left_ankle_x - right_ankle_x)
        features.append(np.mean(step_widths))
        features.append(np.std(step_widths))

        # 3. Arm swing amplitude
        left_wrist_y = kp_array[:, 15 * 3 + 1]
        right_wrist_y = kp_array[:, 16 * 3 + 1]
        left_arm_fft = np.abs(fft(left_wrist_y - np.mean(left_wrist_y)))[:len(left_wrist_y) // 2]
        right_arm_fft = np.abs(fft(right_wrist_y - np.mean(right_wrist_y)))[:len(right_wrist_y) // 2]
        features.extend(self._normalize_vector(left_arm_fft, 16))
        features.extend(self._normalize_vector(right_arm_fft, 16))

        # 4. Body lean angle
        left_shoulder_x = kp_array[:, 11 * 3]
        left_shoulder_y = kp_array[:, 11 * 3 + 1]
        left_hip_x = kp_array[:, 23 * 3]
        left_hip_y = kp_array[:, 23 * 3 + 1]

        lean_angles = np.arctan2(left_shoulder_x - left_hip_x,
                                 left_hip_y - left_shoulder_y)
        features.append(np.mean(lean_angles))
        features.append(np.std(lean_angles))

        # 5. Cadence estimation
        # Find peaks in ankle oscillation
        peaks_l, _ = find_peaks(left_ankle_y, distance=5)
        peaks_r, _ = find_peaks(right_ankle_y, distance=5)
        cadence_l = len(peaks_l) / max(len(keypoint_sequence), 1)
        cadence_r = len(peaks_r) / max(len(keypoint_sequence), 1)
        features.append(cadence_l)
        features.append(cadence_r)

        # 6. Knee lift height
        left_knee_y = kp_array[:, 25 * 3 + 1]
        right_knee_y = kp_array[:, 26 * 3 + 1]
        left_hip_y_series = kp_array[:, 23 * 3 + 1]
        right_hip_y_series = kp_array[:, 24 * 3 + 1]

        left_knee_lift = left_hip_y_series - left_knee_y
        right_knee_lift = right_hip_y_series - right_knee_y
        features.append(np.mean(left_knee_lift))
        features.append(np.std(left_knee_lift))
        features.append(np.mean(right_knee_lift))
        features.append(np.std(right_knee_lift))

        # 7. Shoulder-hip ratio and body proportions
        right_shoulder_y = kp_array[:, 12 * 3 + 1]
        right_hip_y_series = kp_array[:, 24 * 3 + 1]
        torso_length = np.mean(right_hip_y_series - right_shoulder_y)
        features.append(torso_length)

        # 8. Overall movement amplitude stats
        for joint_idx in [11, 12, 23, 24, 25, 26, 27, 28, 15, 16]:
            joint_y = kp_array[:, joint_idx * 3 + 1]
            features.append(np.std(joint_y))

        # Pad or truncate to exactly 128 dimensions
        features = np.array(features, dtype=np.float32)
        signature = self._normalize_vector(features, 128)

        # L2 normalize
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature = signature / norm

        return np.array(signature, dtype=np.float32)

    def _normalize_vector(self, vec: np.ndarray, target_len: int) -> list:
        """Pad with zeros or truncate to target length."""
        if len(vec) >= target_len:
            return vec[:target_len].tolist()
        else:
            return np.pad(vec, (0, target_len - len(vec))).tolist()

    def similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Cosine similarity between two gait signatures. Returns 0-1."""
        if sig1 is None or sig2 is None:
            return 0.0
        dot = np.dot(sig1, sig2)
        return float(max(0, min(1, (dot + 1) / 2)))  # Map [-1,1] to [0,1]

    def build_roster_signatures(self, detections_df, frames: dict, db, players: list):
        """
        Build gait signatures for roster players from game footage.
        Build progressively during first 5 minutes of game.
        Save to DB.
        """
        # Get first 5 minutes of detections
        early_detections = detections_df[detections_df["time_s"] <= 300]

        track_ids = early_detections["track_id"].unique()

        for tid in track_ids:
            track_data = early_detections[early_detections["track_id"] == tid]

            if len(track_data) < MIN_GAIT_FRAMES:
                continue

            # Extract keypoints for this track
            keypoint_sequence = []
            for _, row in track_data.iterrows():
                frame_num = int(row["frame"])
                if frame_num in frames:
                    bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
                    kps = self.extract_keypoints(frames[frame_num], bbox)
                    if kps is not None:
                        keypoint_sequence.append(kps)

            if len(keypoint_sequence) >= MIN_GAIT_FRAMES:
                signature = self.build_gait_signature(keypoint_sequence)
                if signature is not None:
                    # Store temporarily keyed by track_id
                    # Will be mapped to player_id after face matching
                    pass

    def extract_track_signature(self, track_data, frames: dict) -> Optional[np.ndarray]:
        """Extract gait signature for a specific track from its detections."""
        if len(track_data) < MIN_GAIT_FRAMES:
            return None

        keypoint_sequence = []
        for _, row in track_data.iterrows():
            frame_num = int(row["frame"])
            if frame_num in frames:
                bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
                kps = self.extract_keypoints(frames[frame_num], bbox)
                if kps is not None:
                    keypoint_sequence.append(kps)

            if len(keypoint_sequence) >= MIN_GAIT_FRAMES * 2:
                break  # Enough frames

        if len(keypoint_sequence) >= MIN_GAIT_FRAMES:
            return self.build_gait_signature(keypoint_sequence)
        return None
