"""InsightFace face recognition + matching for player identification."""

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import Counter
import insightface
from config import FACE_MATCH_THRESHOLD, MIN_BBOX_HEIGHT_FACE


def load_face_model():
    """Load InsightFace model (not cached — freed after use)."""
    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


class FaceReID:
    def __init__(self):
        self._app = None  # Lazy-loaded to avoid OOM when not needed
        self.roster_embeddings = {}  # {player_id: 512-dim np.array}

    @property
    def app(self):
        if self._app is None:
            self._app = load_face_model()
        return self._app

    def build_roster_embeddings(self, players: list, db) -> dict:
        """
        For each player in roster:
        1. Load their photo
        2. Detect face
        3. Extract 512-dim ArcFace embedding
        4. Store in self.roster_embeddings AND save to DB

        Returns: {player_id: embedding} for successful extractions
        """
        results = {}
        for player in players:
            player_id = player["id"]

            # Check if embedding already in DB
            existing = db.get_face_embedding(player_id)
            if existing is not None:
                self.roster_embeddings[player_id] = existing
                results[player_id] = existing
                continue

            # Try to extract from photo
            photo_path = player.get("photo_path")
            if not photo_path:
                continue

            try:
                img = cv2.imread(photo_path)
                if img is None:
                    continue

                faces = self.app.get(img)
                if len(faces) == 0:
                    continue

                # Use the largest face detected
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embedding = largest_face.embedding

                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)

                self.roster_embeddings[player_id] = embedding
                db.update_face_embedding(player_id, embedding)
                results[player_id] = embedding

            except Exception:
                continue

        return results

    def extract_face_embedding(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """
        Crop face region from top 35% of bounding box.
        Detect face within crop.
        Return 512-dim embedding or None if no clear face found.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bbox_height = y2 - y1

        # Only attempt if bbox is large enough
        if bbox_height < MIN_BBOX_HEIGHT_FACE:
            return None

        # Crop top 35% (face region)
        face_y2 = y1 + int(bbox_height * 0.35)
        face_crop = frame[max(0, y1):min(frame.shape[0], face_y2),
                          max(0, x1):min(frame.shape[1], x2)]

        if face_crop.size == 0:
            return None

        try:
            faces = self.app.get(face_crop)
            if len(faces) == 0:
                return None

            # Use highest confidence face
            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.det_score < 0.5:
                return None

            embedding = best_face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

        except Exception:
            return None

    def match(self, embedding: np.ndarray, threshold: float = FACE_MATCH_THRESHOLD) -> Tuple[Optional[int], float, dict]:
        """
        Compare embedding against all roster embeddings using cosine similarity.

        Returns:
            best_match_player_id: int or None
            confidence: float 0-1
            all_scores: dict {player_id: similarity_score}
        """
        if not self.roster_embeddings:
            return None, 0.0, {}

        all_scores = {}
        for player_id, roster_emb in self.roster_embeddings.items():
            similarity = float(np.dot(embedding, roster_emb))
            all_scores[player_id] = similarity

        if not all_scores:
            return None, 0.0, {}

        best_player_id = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_player_id]

        if best_score >= threshold:
            return best_player_id, best_score, all_scores
        else:
            return None, best_score, all_scores

    def batch_match_video(self, detections_df, frames: dict) -> dict:
        """
        Process all frames, attempt face match for each track_id.
        Aggregate matches per track_id (vote on most frequent match).

        Returns: {track_id: {player_id, confidence, face_match_count}}
        """
        # Skip entirely if no roster embeddings — avoids loading heavy model
        if not self.roster_embeddings:
            return {tid: {"player_id": None, "confidence": 0.0,
                          "face_match_count": 0, "total_attempts": 0}
                    for tid in detections_df["track_id"].unique()}

        track_matches = {}  # {track_id: [(player_id, confidence), ...]}

        track_ids = detections_df["track_id"].unique()

        for tid in track_ids:
            track_data = detections_df[detections_df["track_id"] == tid]
            matches = []

            # Sample frames for this track (every 5th detection)
            sample_indices = range(0, len(track_data), 5)

            for idx in sample_indices:
                if idx >= len(track_data):
                    break
                row = track_data.iloc[idx]
                frame_num = int(row["frame"])

                if frame_num not in frames:
                    continue

                frame = frames[frame_num]
                bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])

                embedding = self.extract_face_embedding(frame, bbox)
                if embedding is not None:
                    player_id, confidence, _ = self.match(embedding)
                    if player_id is not None:
                        matches.append((player_id, confidence))

            if matches:
                # Vote on most frequent player_id
                player_ids = [m[0] for m in matches]
                counter = Counter(player_ids)
                best_player_id, count = counter.most_common(1)[0]

                # Average confidence for the winning player
                winning_confidences = [m[1] for m in matches if m[0] == best_player_id]
                avg_confidence = np.mean(winning_confidences)

                track_matches[tid] = {
                    "player_id": best_player_id,
                    "confidence": float(avg_confidence),
                    "face_match_count": count,
                    "total_attempts": len(matches),
                }
            else:
                track_matches[tid] = {
                    "player_id": None,
                    "confidence": 0.0,
                    "face_match_count": 0,
                    "total_attempts": 0,
                }

        return track_matches
