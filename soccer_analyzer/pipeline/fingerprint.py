"""Multi-modal fusion: jersey OCR + face + gait + cleat + height for player identification."""

import numpy as np
import pandas as pd
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from config import ID_AUTO_ASSIGN_THRESHOLD, ID_CONFIRMATION_THRESHOLD


class PlayerFingerprinter:
    """
    Multi-modal player identification engine.
    Combines jersey OCR, face Re-ID, gait, cleat color, height into unified ID.
    Weights optimized for wide-angle full-field recording (players at distance).
    """

    WEIGHTS = {
        "jersey_ocr": 0.30,
        "face": 0.15,
        "gait": 0.20,
        "cleat": 0.15,
        "height": 0.10,
        "hair": 0.05,
        "body_shape": 0.05,
    }

    def __init__(self, roster_db, face_reid, gait_analyzer, cleat_extractor, jersey_ocr=None):
        self.db = roster_db
        self.face = face_reid
        self.gait = gait_analyzer
        self.cleat = cleat_extractor
        self.jersey_ocr = jersey_ocr
        self.roster_heights = {}  # {player_id: relative_height}
        self.roster_cleat_colors = {}  # {player_id: hsv_dict}
        self.roster_gait_sigs = {}  # {player_id: gait_signature}
        self.roster_numbers = {}  # {jersey_number: player_id}

    def _load_roster_profiles(self):
        """Load all roster profiles from DB."""
        players = self.db.get_all_players()
        for p in players:
            pid = p["id"]
            if p["relative_height"] is not None:
                self.roster_heights[pid] = p["relative_height"]
            cleat = self.db.get_cleat_color(pid)
            if cleat:
                self.roster_cleat_colors[pid] = cleat
            gait = self.db.get_gait_signature(pid)
            if gait is not None:
                self.roster_gait_sigs[pid] = gait
            # Build jersey number → player_id mapping
            jersey_num = p.get("jersey_number")
            if jersey_num and jersey_num > 0:
                self.roster_numbers[jersey_num] = pid

    def identify_track(self, track_id: int, detections_df: pd.DataFrame,
                       frames: dict, face_results: dict = None,
                       cleat_profiles: dict = None,
                       track_heights: dict = None,
                       jersey_ocr_results: dict = None) -> dict:
        """
        For a given track_id, fuse all identification signals.

        Returns:
            {
                player_id: int or None,
                confidence: float 0-1,
                breakdown: dict showing contribution of each signal,
                needs_confirmation: bool
            }
        """
        track_data = detections_df[detections_df["track_id"] == track_id]
        if track_data.empty:
            return {"player_id": None, "confidence": 0.0,
                    "breakdown": {}, "needs_confirmation": False}

        players = self.db.get_all_players()
        player_scores = {p["id"]: {} for p in players}

        # 1. Jersey OCR scores (highest weight — number is near-certain ID)
        if jersey_ocr_results and track_id in jersey_ocr_results:
            ocr = jersey_ocr_results[track_id]
            detected_num = ocr.get("jersey_number")
            ocr_conf = ocr.get("confidence", 0.0)
            if detected_num is not None and detected_num in self.roster_numbers:
                matched_pid = self.roster_numbers[detected_num]
                for pid in player_scores:
                    if pid == matched_pid:
                        player_scores[pid]["jersey_ocr"] = ocr_conf
                    else:
                        player_scores[pid]["jersey_ocr"] = 0.0
            else:
                for pid in player_scores:
                    player_scores[pid]["jersey_ocr"] = 0.0
        else:
            for pid in player_scores:
                player_scores[pid]["jersey_ocr"] = 0.0

        # 2. Face scores
        if face_results and track_id in face_results:
            fr = face_results[track_id]
            if fr["player_id"] is not None:
                for pid in player_scores:
                    if pid == fr["player_id"]:
                        player_scores[pid]["face"] = fr["confidence"]
                    else:
                        player_scores[pid]["face"] = 0.0
            else:
                for pid in player_scores:
                    player_scores[pid]["face"] = 0.0
        else:
            for pid in player_scores:
                player_scores[pid]["face"] = 0.0

        # 3. Gait scores
        gait_sig = self.gait.extract_track_signature(track_data, frames)
        if gait_sig is not None:
            for pid in player_scores:
                if pid in self.roster_gait_sigs:
                    player_scores[pid]["gait"] = self.gait.similarity(
                        gait_sig, self.roster_gait_sigs[pid]
                    )
                else:
                    player_scores[pid]["gait"] = 0.0
        else:
            for pid in player_scores:
                player_scores[pid]["gait"] = 0.0

        # 4. Cleat color scores
        if cleat_profiles and track_id in cleat_profiles:
            track_cleat = cleat_profiles[track_id]
            for pid in player_scores:
                if pid in self.roster_cleat_colors:
                    player_scores[pid]["cleat"] = self.cleat.similarity(
                        track_cleat, self.roster_cleat_colors[pid]
                    )
                else:
                    player_scores[pid]["cleat"] = 0.0
        else:
            for pid in player_scores:
                player_scores[pid]["cleat"] = 0.0

        # 5. Height scores
        if track_heights and track_id in track_heights:
            track_height = track_heights[track_id]
            for pid in player_scores:
                if pid in self.roster_heights:
                    height_diff = abs(track_height - self.roster_heights[pid])
                    player_scores[pid]["height"] = max(0, 1.0 - height_diff * 3)
                else:
                    player_scores[pid]["height"] = 0.5  # neutral
        else:
            for pid in player_scores:
                player_scores[pid]["height"] = 0.0

        # 6. Hair (placeholder — would need hair color model)
        for pid in player_scores:
            player_scores[pid]["hair"] = 0.0

        # 7. Body shape (placeholder — width/height ratio as proxy)
        for pid in player_scores:
            player_scores[pid]["body_shape"] = 0.0

        # Weighted fusion
        final_scores = {}
        for pid, scores in player_scores.items():
            weighted_sum = 0.0
            total_weight = 0.0
            for signal, weight in self.WEIGHTS.items():
                if signal in scores and scores[signal] > 0:
                    weighted_sum += scores[signal] * weight
                    total_weight += weight

            # Normalize by available weight
            # Require at least one strong signal (jersey OCR, face, or gait) to produce a meaningful score
            has_real_signal = (scores.get("jersey_ocr", 0) > 0 or
                              scores.get("face", 0) > 0 or
                              scores.get("gait", 0) > 0)
            if total_weight > 0 and has_real_signal:
                final_scores[pid] = min(weighted_sum / total_weight, 1.0)
            else:
                final_scores[pid] = 0.0

        # Find best match
        if not final_scores:
            return {"player_id": None, "confidence": 0.0,
                    "breakdown": {}, "needs_confirmation": False}

        best_pid = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_pid]

        needs_confirmation = (ID_CONFIRMATION_THRESHOLD <= confidence < ID_AUTO_ASSIGN_THRESHOLD)

        return {
            "player_id": best_pid if confidence >= ID_CONFIRMATION_THRESHOLD else None,
            "confidence": confidence,
            "breakdown": player_scores.get(best_pid, {}),
            "needs_confirmation": needs_confirmation,
        }

    def identify_all_tracks(self, detections_df: pd.DataFrame, frames: dict,
                            progress_callback=None) -> tuple:
        """
        Run identification for every unique track_id.

        Returns:
            assignments: dict {track_id: {player_id, confidence, status}}
            pending_confirmations: list of {track_id, candidates, confidence}
        """
        self._load_roster_profiles()

        # Run jersey OCR in batch (if available)
        jersey_ocr_results = None
        if self.jersey_ocr is not None:
            jersey_ocr_results = self.jersey_ocr.batch_read(detections_df, frames)

        # Run face matching in batch
        face_results = self.face.batch_match_video(detections_df, frames)

        # Build cleat profiles
        cleat_profiles = self.cleat.build_cleat_profiles(detections_df, frames)

        # Get relative heights
        track_heights = self.get_relative_heights(detections_df)

        track_ids = detections_df["track_id"].unique()
        assignments = {}
        pending_confirmations = []

        for i, tid in enumerate(track_ids):
            result = self.identify_track(
                tid, detections_df, frames,
                face_results=face_results,
                cleat_profiles=cleat_profiles,
                track_heights=track_heights,
                jersey_ocr_results=jersey_ocr_results,
            )

            if result["confidence"] >= ID_AUTO_ASSIGN_THRESHOLD:
                assignments[tid] = {
                    "player_id": result["player_id"],
                    "confidence": result["confidence"],
                    "status": "auto_assigned",
                }
            elif result["needs_confirmation"]:
                assignments[tid] = {
                    "player_id": result["player_id"],
                    "confidence": result["confidence"],
                    "status": "needs_confirmation",
                }
                pending_confirmations.append({
                    "track_id": tid,
                    "candidates": self._get_top_candidates(
                        tid, detections_df, frames, face_results,
                        cleat_profiles, track_heights
                    ),
                    "confidence": result["confidence"],
                })
            else:
                assignments[tid] = {
                    "player_id": None,
                    "confidence": result["confidence"],
                    "status": "unknown",
                }

            if progress_callback:
                progress_callback(i + 1, len(track_ids))

        return assignments, pending_confirmations

    def _get_top_candidates(self, track_id, detections_df, frames,
                            face_results, cleat_profiles, track_heights) -> list:
        """Get top 3 candidate player IDs for a track."""
        track_data = detections_df[detections_df["track_id"] == track_id]
        players = self.db.get_all_players()

        scores = {}
        for p in players:
            pid = p["id"]
            # Quick scoring based on face result
            face_score = 0.0
            if face_results and track_id in face_results:
                if face_results[track_id]["player_id"] == pid:
                    face_score = face_results[track_id]["confidence"]
            scores[pid] = face_score

        # Sort and return top 3
        sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        return [{"player_id": pid, "score": score} for pid, score in sorted_players]

    def merge_track_ids(self, assignments: dict, detections_df: pd.DataFrame) -> pd.DataFrame:
        """
        After all track_ids are identified, merge tracks belonging to same player.
        Add player_id, player_name, jersey_number, id_confidence columns.
        """
        detections_df = detections_df.copy()

        # Create mapping columns
        players = {p["id"]: p for p in self.db.get_all_players()}

        player_ids = []
        player_names = []
        jersey_numbers = []
        confidences = []

        for _, row in detections_df.iterrows():
            tid = row["track_id"]
            if tid in assignments and assignments[tid]["player_id"] is not None:
                pid = assignments[tid]["player_id"]
                player = players.get(pid, {})
                player_ids.append(pid)
                player_names.append(player.get("name", f"Unknown_{tid}"))
                jersey_numbers.append(player.get("jersey_number", 0))
                confidences.append(assignments[tid]["confidence"])
            else:
                player_ids.append(None)
                player_names.append(f"Unknown_{tid}")
                jersey_numbers.append(0)
                confidences.append(0.0)

        detections_df["player_id"] = player_ids
        detections_df["player_name"] = player_names
        detections_df["jersey_number"] = jersey_numbers
        detections_df["id_confidence"] = confidences

        return detections_df

    def get_relative_heights(self, detections_df: pd.DataFrame) -> dict:
        """
        Estimate relative height of each track_id from bounding box heights.
        Normalize within team (0 = shortest, 1 = tallest).
        Use 75th percentile of bbox heights.
        """
        track_heights_raw = {}

        for tid in detections_df["track_id"].unique():
            track_data = detections_df[detections_df["track_id"] == tid]
            bbox_heights = track_data["bbox_y2"] - track_data["bbox_y1"]
            # Use 75th percentile to avoid crouching/jumping noise
            track_heights_raw[tid] = float(np.percentile(bbox_heights, 75))

        if not track_heights_raw:
            return {}

        # Normalize to 0-1
        heights = np.array(list(track_heights_raw.values()))
        min_h = heights.min()
        max_h = heights.max()
        range_h = max_h - min_h if max_h > min_h else 1.0

        track_heights = {}
        for tid, h in track_heights_raw.items():
            track_heights[tid] = (h - min_h) / range_h

        return track_heights
