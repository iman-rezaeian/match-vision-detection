"""Cleat color extraction and matching for player fingerprinting."""

import cv2
import numpy as np
from typing import Optional
from sklearn.cluster import KMeans


class CleatExtractor:
    def extract_cleat_color(self, frame: np.ndarray, bbox: tuple) -> Optional[dict]:
        """
        Crop bottom 20% of bounding box (foot/cleat region).
        Convert to HSV.
        Use KMeans k=2 to find dominant non-grass color.
        Return dominant HSV values as dict {"h": x, "s": y, "v": z}.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h = y2 - y1
        w = x2 - x1

        # Bottom 20% of bounding box
        cleat_y1 = y2 - int(h * 0.20)
        cleat_y2 = y2

        # Clamp to frame bounds
        cleat_y1 = max(0, cleat_y1)
        cleat_y2 = min(frame.shape[0], cleat_y2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)

        if cleat_y2 <= cleat_y1 or x2 <= x1:
            return None

        crop = frame[cleat_y1:cleat_y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        if len(pixels) < 10:
            return None

        # Filter out grass-colored pixels (green: H 35-85, S>40, V>40)
        grass_mask = (
            (pixels[:, 0] >= 35) & (pixels[:, 0] <= 85) &
            (pixels[:, 1] > 40) & (pixels[:, 2] > 40)
        )
        non_grass = pixels[~grass_mask]

        if len(non_grass) < 5:
            return None

        # KMeans k=2 on non-grass pixels
        n_clusters = min(2, len(non_grass))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        kmeans.fit(non_grass)

        # Return dominant cluster (most pixels)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_color = kmeans.cluster_centers_[dominant_idx]

        return {
            "h": float(dominant_color[0]),
            "s": float(dominant_color[1]),
            "v": float(dominant_color[2]),
        }

    def build_cleat_profiles(self, detections_df, frames: dict) -> dict:
        """
        For each track_id, aggregate cleat color samples across multiple frames.
        Return stable cleat color per track_id.
        """
        track_ids = detections_df["track_id"].unique()
        profiles = {}

        for tid in track_ids:
            track_data = detections_df[detections_df["track_id"] == tid]
            colors = []

            # Sample every 10th detection
            sample_indices = range(0, len(track_data), 10)

            for idx in sample_indices:
                if idx >= len(track_data):
                    break
                row = track_data.iloc[idx]
                frame_num = int(row["frame"])

                if frame_num not in frames:
                    continue

                frame = frames[frame_num]
                bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
                color = self.extract_cleat_color(frame, bbox)

                if color is not None:
                    colors.append(color)

            if len(colors) >= 3:
                # Average the HSV values
                avg_h = np.median([c["h"] for c in colors])
                avg_s = np.median([c["s"] for c in colors])
                avg_v = np.median([c["v"] for c in colors])

                profiles[tid] = {
                    "h": float(avg_h),
                    "s": float(avg_s),
                    "v": float(avg_v),
                    "sample_count": len(colors),
                }

        return profiles

    def similarity(self, hsv1: dict, hsv2: dict) -> float:
        """
        Compare two cleat colors.
        Weight hue heavily, saturation medium, value low.
        Handle black cleats (low saturation) as special case.
        Return 0-1 similarity.
        """
        if hsv1 is None or hsv2 is None:
            return 0.0

        h1, s1, v1 = hsv1["h"], hsv1["s"], hsv1["v"]
        h2, s2, v2 = hsv2["h"], hsv2["s"], hsv2["v"]

        # Special case: both are black cleats (low saturation)
        if s1 < 30 and s2 < 30:
            # Both black - compare value only
            v_diff = abs(v1 - v2) / 255.0
            return 1.0 - v_diff

        # Special case: one is black, other is not
        if (s1 < 30) != (s2 < 30):
            return 0.1  # Very different

        # Normal comparison with weighted distances
        # Hue is circular (0-180 in OpenCV)
        h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 90.0
        s_diff = abs(s1 - s2) / 255.0
        v_diff = abs(v1 - v2) / 255.0

        # Weighted combination
        distance = (h_diff * 0.6) + (s_diff * 0.25) + (v_diff * 0.15)
        similarity = max(0.0, 1.0 - distance)

        return similarity

    def build_roster_profiles(self, players: list, db) -> dict:
        """
        If roster has cleat color manually entered → use that.
        Otherwise: will be extracted from first game automatically.
        """
        profiles = {}
        for player in players:
            cleat_color = db.get_cleat_color(player["id"])
            if cleat_color:
                profiles[player["id"]] = cleat_color
        return profiles
