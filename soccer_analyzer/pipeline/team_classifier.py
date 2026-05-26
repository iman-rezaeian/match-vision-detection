"""KMeans jersey color classification for team assignment."""

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter


class TeamClassifier:
    def __init__(self):
        self.team_colors = {}  # {team_label: dominant_hsv}

    def extract_jersey_color(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract dominant jersey color from the torso region of a player.
        Returns BGR median of non-green, non-skin pixels in the tight chest crop.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h = y2 - y1
        w = x2 - x1

        # Tight torso crop — narrower vertically to avoid shorts/hands
        if w < 30:
            torso_y1 = y1 + int(h * 0.22)
            torso_y2 = y1 + int(h * 0.52)
            torso_x1 = x1 + int(w * 0.15)
            torso_x2 = x2 - int(w * 0.15)
        else:
            torso_y1 = y1 + int(h * 0.25)
            torso_y2 = y1 + int(h * 0.50)
            torso_x1 = x1 + int(w * 0.25)
            torso_x2 = x2 - int(w * 0.25)

        # Clamp to frame bounds
        torso_y1 = max(0, torso_y1)
        torso_y2 = min(frame.shape[0], torso_y2)
        torso_x1 = max(0, torso_x1)
        torso_x2 = min(frame.shape[1], torso_x2)

        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return np.array([0, 0, 0])

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        pixels_bgr = crop.reshape(-1, 3).astype(np.float32)

        # Filter out green field pixels AND skin-tone pixels using HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels_hsv = hsv.reshape(-1, 3).astype(np.float32)

        # Green field: H 35-85, S > 40
        green_mask = (
            (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) &
            (pixels_hsv[:, 1] > 40)
        )
        # Skin tone: H 0-25, S 30-130, V 60-230
        # Key: cap saturation at 130 so saturated reds (S>130) are NOT masked.
        # Red jerseys have H<10 but high saturation (>130), skin is lower (30-130).
        skin_mask = (
            (pixels_hsv[:, 0] <= 25) &
            (pixels_hsv[:, 1] >= 30) & (pixels_hsv[:, 1] <= 130) &
            (pixels_hsv[:, 2] > 60) & (pixels_hsv[:, 2] < 230)
        )
        filtered = pixels_bgr[~green_mask & ~skin_mask]

        if len(filtered) < 5:
            # Relax: try without skin filter
            filtered = pixels_bgr[~green_mask]
            if len(filtered) < 5:
                return np.array([0, 0, 0])

        # Use the most saturated pixels as they carry actual jersey color.
        # At tiny bbox sizes (30-40px), mixed/washed pixels dominate;
        # high-saturation pixels are the true fabric color signal.
        filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
        filtered_hsv = cv2.cvtColor(
            filtered_uint8.reshape(1, -1, 3), cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)
        sat_values = filtered_hsv[:, 1].astype(np.float32)

        # Take top 40% most saturated pixels (at least 3)
        n_take = max(3, int(len(sat_values) * 0.4))
        top_idx = np.argsort(sat_values)[-n_take:]
        top_pixels = filtered[top_idx]

        # For very dark pixels (black jerseys), saturation is naturally low.
        # If median saturation is very low (<15), the jersey is dark — use all pixels.
        if np.median(sat_values) < 15:
            return np.median(filtered, axis=0)

        return np.median(top_pixels, axis=0)

    def classify_teams(self, detections_df: pd.DataFrame, frames: dict,
                       sample_per_track: int = 5,
                       progress_callback=None,
                       my_team_color: str = None,
                       opponent_color: str = None) -> pd.DataFrame:
        """
        Classify all tracked players into two teams based on jersey color.

        Uses DETECTION-LEVEL classification: extract jersey color from
        individual detections, cluster ALL colors at once, then assign
        each track by majority vote. This avoids dependence on track
        continuity (which ByteTrack struggles with for small/distant players).

        Args:
            my_team_color: Optional hint — "black" or "green". If provided,
                the cluster closest to this color is labeled "Home" (my team).

        Returns detections_df with added 'team' column ('Home' or 'Away').
        """
        # --- Step 1: Filter detections to on-field players only -------------
        df = detections_df.copy()
        df["_bw"] = df["bbox_x2"] - df["bbox_x1"]
        df["_bh"] = df["bbox_y2"] - df["bbox_y1"]
        df["_area"] = df["_bw"] * df["_bh"]

        # Remove sideline coaches/parents (huge bboxes) and tiny noise
        area_p90 = df["_area"].quantile(0.90)
        size_mask = (
            (df["_bw"] >= 12) & (df["_bh"] >= 25) &   # not too small
            (df["_area"] < max(area_p90, 8000))          # not sideline giants
        )
        candidates = df[size_mask]
        if len(candidates) < 10:
            candidates = df  # fallback

        # --- Step 2: Sample detections for color extraction -----------------
        # Sample evenly across frames, picking highest-confidence detections.
        # Cap at ~500 frames to avoid 300K+ video seeks on long games.
        sampled = []
        max_per_frame = 15  # enough for all players on field
        grouped = candidates.groupby("frame")
        frame_nums = sorted(grouped.groups.keys())
        frame_nums = [f for f in frame_nums if f in frames]

        max_sample_frames = 500
        if len(frame_nums) > max_sample_frames:
            indices = np.linspace(0, len(frame_nums) - 1, max_sample_frames, dtype=int)
            frame_nums = [frame_nums[i] for i in indices]

        for frame_num in frame_nums:
            group = grouped.get_group(frame_num)
            top = group.nlargest(max_per_frame, "_area")
            for _, row in top.iterrows():
                sampled.append(row)

        if not sampled:
            detections_df = detections_df.copy()
            detections_df["team"] = "Unknown"
            return detections_df

        sampled_df = pd.DataFrame(sampled)

        # --- Step 3: Extract jersey color for each sampled detection --------
        det_colors = []  # (index_in_sampled, BGR color)
        for idx, row in sampled_df.iterrows():
            frame_num = int(row["frame"])
            if frame_num not in frames:
                continue
            frame = frames[frame_num]
            frame_h_actual, frame_w_actual = frame.shape[:2]
            det_w = row.get("frame_w", frame_w_actual)
            det_h = row.get("frame_h", frame_h_actual)
            sx = frame_w_actual / det_w if det_w > 0 else 1.0
            sy = frame_h_actual / det_h if det_h > 0 else 1.0
            bbox = (
                row["bbox_x1"] * sx, row["bbox_y1"] * sy,
                row["bbox_x2"] * sx, row["bbox_y2"] * sy,
            )
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw < 8 or bh < 15:
                continue
            color = self.extract_jersey_color(frame, bbox)
            if color.sum() > 0:
                det_colors.append({
                    "track_id": int(row["track_id"]),
                    "frame": frame_num,
                    "color_b": color[0],
                    "color_g": color[1],
                    "color_r": color[2],
                })

            if progress_callback and len(det_colors) % 50 == 0:
                progress_callback(len(det_colors), len(sampled_df))

        if len(det_colors) < 4:
            detections_df = detections_df.copy()
            detections_df["team"] = "Unknown"
            return detections_df

        color_df = pd.DataFrame(det_colors)
        color_array = color_df[["color_b", "color_g", "color_r"]].values

        # Filter out detections whose extracted color is green-ish (field bleed).
        # Convert to HSV to check — green field in HSV: H 35-85, S>30
        color_bgr_for_hsv = color_array.reshape(1, -1, 3).astype(np.uint8)
        color_hsv_check = cv2.cvtColor(color_bgr_for_hsv, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        not_green = ~(
            (color_hsv_check[:, 0] >= 30) & (color_hsv_check[:, 0] <= 90) &
            (color_hsv_check[:, 1] > 25)
        )
        color_df = color_df[not_green].reset_index(drop=True)
        if len(color_df) < 4:
            detections_df = detections_df.copy()
            detections_df["team"] = "Unknown"
            return detections_df
        color_array = color_df[["color_b", "color_g", "color_r"]].values

        # --- Step 4: Cluster using ONLY the best detections (largest bboxes) --
        # Merge bbox area info back into color_df
        color_df_with_area = color_df.merge(
            df[["track_id", "frame", "_area"]].drop_duplicates(),
            on=["track_id", "frame"], how="left"
        )
        # Use upper 40% of bbox area for clustering (clearest jersey views)
        area_threshold = color_df_with_area["_area"].quantile(0.60)
        elite_mask = color_df_with_area["_area"] >= area_threshold
        elite_df = color_df[elite_mask].reset_index(drop=True)
        if len(elite_df) < 6:
            elite_df = color_df  # fallback
        elite_colors = elite_df[["color_b", "color_g", "color_r"]].values

        # Cluster elite detections into 3 groups in LAB space:
        # Expect: team-A, team-B, and noise/field-bleed
        elite_bgr_img = elite_colors.reshape(1, -1, 3).astype(np.uint8)
        elite_lab_img = cv2.cvtColor(elite_bgr_img, cv2.COLOR_BGR2LAB)
        elite_lab = elite_lab_img.reshape(-1, 3).astype(np.float32)

        n_clusters = min(3, len(elite_lab))
        kmeans3 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels3 = kmeans3.fit_predict(elite_lab)

        # Pick the 2 team clusters from 3.
        # LAB: L=lightness (0-255 in OpenCV), A=green-red (128=neutral),
        #       B=blue-yellow (128=neutral)
        centers3 = kmeans3.cluster_centers_
        if n_clusters == 3:
            # Strategy: pick the 2 clusters most distant from each other in LAB space.
            # This correctly handles black vs white (max lightness difference)
            # as well as black vs colored (different chromaticity).
            from itertools import combinations
            pair_dists = []
            for i, j in combinations(range(3), 2):
                d = np.linalg.norm(centers3[i] - centers3[j])
                pair_dists.append((d, i, j))
            pair_dists.sort(reverse=True)
            team_c0, team_c1 = pair_dists[0][1], pair_dists[0][2]

            # Ensure team_c0 is the darker cluster (lower L)
            if centers3[team_c0][0] > centers3[team_c1][0]:
                team_c0, team_c1 = team_c1, team_c0
        elif n_clusters == 2:
            team_c0, team_c1 = 0, 1
        else:
            team_c0, team_c1 = 0, 0

        # --- Assign Home/Away based on user's jersey color hints ---
        # Reference colors in BGR for common jersey colors
        JERSEY_REFS_BGR = {
            "black":  np.array([30, 30, 30], dtype=np.uint8),
            "white":  np.array([230, 230, 230], dtype=np.uint8),
            "red":    np.array([40, 40, 200], dtype=np.uint8),
            "blue":   np.array([200, 80, 40], dtype=np.uint8),
            "green":  np.array([50, 120, 50], dtype=np.uint8),
            "yellow": np.array([30, 220, 230], dtype=np.uint8),
            "orange": np.array([30, 130, 240], dtype=np.uint8),
            "purple": np.array([140, 40, 130], dtype=np.uint8),
        }

        def _color_to_lab(name):
            bgr = JERSEY_REFS_BGR[name].reshape(1, 1, 3)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)

        # Convert all 3 cluster centers to LAB for comparison
        # centers3 is already in LAB space

        # If both team colors are provided, pick the 2 clusters that best match
        if (my_team_color and my_team_color in JERSEY_REFS_BGR and
                opponent_color and opponent_color in JERSEY_REFS_BGR):
            home_lab = _color_to_lab(my_team_color)
            away_lab = _color_to_lab(opponent_color)

            if n_clusters == 3:
                # Try all 3 possible pairings of (home_cluster, away_cluster)
                from itertools import permutations
                best_score = float("inf")
                best_pair = (team_c0, team_c1)
                for ci, cj in permutations(range(3), 2):
                    d_home = np.linalg.norm(centers3[ci] - home_lab)
                    d_away = np.linalg.norm(centers3[cj] - away_lab)
                    score = d_home + d_away
                    if score < best_score:
                        best_score = score
                        best_pair = (ci, cj)
                team_c0, team_c1 = best_pair
            else:
                # 2 clusters — just assign Home/Away
                d00 = np.linalg.norm(centers3[0] - home_lab)
                d01 = np.linalg.norm(centers3[0] - away_lab)
                if d00 > d01:
                    team_c0, team_c1 = 1, 0
                else:
                    team_c0, team_c1 = 0, 1
            home_is_c0 = True  # team_c0 is already matched to home

        elif my_team_color and my_team_color in JERSEY_REFS_BGR:
            # Only home color known — match it to nearest cluster
            ref_lab = _color_to_lab(my_team_color)
            d_c0 = np.linalg.norm(centers3[team_c0] - ref_lab)
            d_c1 = np.linalg.norm(centers3[team_c1] - ref_lab)
            home_is_c0 = d_c0 < d_c1
        else:
            home_is_c0 = True  # default

        # Rebuild team_centers_lab with final cluster assignments
        team_centers_lab = np.array([centers3[team_c0], centers3[team_c1]])

        # Convert team cluster centers back to BGR
        lab_centers_img = team_centers_lab.reshape(1, -1, 3).astype(np.uint8)
        bgr_centers_img = cv2.cvtColor(lab_centers_img, cv2.COLOR_LAB2BGR)
        bgr_centers = bgr_centers_img.reshape(-1, 3).astype(np.float32)

        # Assign ALL detections to nearest team center
        all_bgr_img = color_array.reshape(1, -1, 3).astype(np.uint8)
        all_lab_img = cv2.cvtColor(all_bgr_img, cv2.COLOR_BGR2LAB)
        all_lab = all_lab_img.reshape(-1, 3).astype(np.float32)

        d0 = np.linalg.norm(all_lab - team_centers_lab[0], axis=1)
        d1 = np.linalg.norm(all_lab - team_centers_lab[1], axis=1)
        labels = (d1 < d0).astype(int)  # 0=team_c0, 1=team_c1
        color_df["cluster"] = labels
        lab_centers = team_centers_lab

        # Store BGR centers with correct Home/Away mapping
        if home_is_c0:
            self.team_colors = {
                "Home": bgr_centers[0],
                "Away": bgr_centers[1],
            }
        else:
            self.team_colors = {
                "Home": bgr_centers[1],
                "Away": bgr_centers[0],
            }

        # --- Step 5: Assign each track by majority vote --------------------
        team_map = {}
        for tid, group in color_df.groupby("track_id"):
            counts = group["cluster"].value_counts()
            majority_cluster = counts.idxmax()
            if home_is_c0:
                team_map[int(tid)] = "Home" if majority_cluster == 0 else "Away"
            else:
                team_map[int(tid)] = "Away" if majority_cluster == 0 else "Home"

        # For tracks not in the sampled set, batch-classify by frame.
        # Skip noise tracks (<=2 detections); cap frame reads for speed.
        all_track_ids = set(detections_df["track_id"].unique())
        unclassified = all_track_ids - set(team_map.keys())
        track_det_counts = detections_df.groupby("track_id").size()
        unc_meaningful = {tid for tid in unclassified
                         if track_det_counts.get(tid, 0) > 2}

        if unc_meaningful:
            unc_df = detections_df[detections_df["track_id"].isin(unc_meaningful)].copy()
            unc_df["_bw_unc"] = unc_df["bbox_x2"] - unc_df["bbox_x1"]
            best_idx = unc_df.groupby("track_id")["_bw_unc"].idxmax()
            unc_best = unc_df.loc[best_idx]

            # Group by frame; cap to 1000 frames to avoid 15-minute stalls
            frame_groups = [(fn, grp) for fn, grp in unc_best.groupby("frame")
                           if int(fn) in frames]
            frame_groups.sort(key=lambda x: -len(x[1]))  # densest frames first
            max_fallback_frames = 1000
            if len(frame_groups) > max_fallback_frames:
                frame_groups = frame_groups[:max_fallback_frames]

            for fn, group in frame_groups:
                fn = int(fn)
                frame = frames[fn]
                fh, fw = frame.shape[:2]
                for _, row in group.iterrows():
                    det_w_val = row.get("frame_w", fw)
                    det_h_val = row.get("frame_h", fh)
                    sx = fw / det_w_val if det_w_val > 0 else 1.0
                    sy = fh / det_h_val if det_h_val > 0 else 1.0
                    bbox = (
                        row["bbox_x1"] * sx, row["bbox_y1"] * sy,
                        row["bbox_x2"] * sx, row["bbox_y2"] * sy,
                    )
                    if (bbox[2] - bbox[0]) < 8 or (bbox[3] - bbox[1]) < 15:
                        continue
                    color = self.extract_jersey_color(frame, bbox)
                    if color.sum() > 0:
                        c_bgr = color.reshape(1, 1, 3).astype(np.uint8)
                        c_lab = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2LAB).reshape(
                            1, 3).astype(np.float32)
                        d0 = np.linalg.norm(c_lab - lab_centers[0])
                        d1 = np.linalg.norm(c_lab - lab_centers[1])
                        closer_to_c0 = d0 < d1
                        if home_is_c0:
                            team_map[int(row["track_id"])] = (
                                "Home" if closer_to_c0 else "Away")
                        else:
                            team_map[int(row["track_id"])] = (
                                "Away" if closer_to_c0 else "Home")

        # Apply to dataframe
        detections_df = detections_df.copy()
        detections_df["team"] = detections_df["track_id"].map(team_map).fillna("Unknown")

        return detections_df

    def swap_teams(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        """Swap Home/Away labels if user indicates the auto-assignment is wrong."""
        detections_df = detections_df.copy()
        swap = {"Home": "Away", "Away": "Home"}
        detections_df["team"] = detections_df["team"].map(lambda t: swap.get(t, t))
        if "Home" in self.team_colors and "Away" in self.team_colors:
            self.team_colors["Home"], self.team_colors["Away"] = (
                self.team_colors["Away"], self.team_colors["Home"]
            )
        return detections_df

    def get_team_color_rgb(self, team: str) -> tuple:
        """Return the team's dominant color as an RGB tuple for display."""
        if team not in self.team_colors:
            return (128, 128, 128)
        bgr = self.team_colors[team]
        # BGR → RGB
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))
