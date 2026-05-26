"""Roster CRUD operations with SQLite."""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional
from config import DB_PATH, INITIAL_ROSTER, PHOTOS_DIR


class RosterDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._seed_roster()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                jersey_number INTEGER NOT NULL UNIQUE,
                photo_path TEXT,
                face_embedding BLOB,
                gait_signature BLOB,
                cleat_color_hsv TEXT,
                relative_height REAL,
                hair_description TEXT,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                field_length_m REAL,
                field_width_m REAL,
                src_points TEXT,
                dst_points TEXT,
                homography_matrix TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def _seed_roster(self):
        cursor = self.conn.execute("SELECT COUNT(*) FROM players")
        count = cursor.fetchone()[0]
        if count == 0:
            for player in INITIAL_ROSTER:
                photo_path = self._find_player_photo(player["jersey"], player["name"])
                self.conn.execute(
                    "INSERT INTO players (name, jersey_number, photo_path) VALUES (?, ?, ?)",
                    (player["name"], player["jersey"], photo_path)
                )
            self.conn.commit()
        else:
            # Update photo paths for existing players if photos were added
            self._update_photo_paths()

    def _find_player_photo(self, jersey: int, name: str) -> Optional[str]:
        """Find a player's photo by jersey number or name in the photos directory."""
        if not PHOTOS_DIR.exists():
            return None
        # Try patterns: "03.jpg", "03_ben_adam.jpg", "ben_adam.jpg", "Ben Adam.jpg"
        name_lower = name.lower().replace(" ", "_")
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidates = [
                PHOTOS_DIR / f"{jersey:02d}{ext}",
                PHOTOS_DIR / f"{jersey}{ext}",
                PHOTOS_DIR / f"{jersey:02d}_{name_lower}{ext}",
                PHOTOS_DIR / f"{name_lower}{ext}",
                PHOTOS_DIR / f"{name}{ext}",
            ]
            for path in candidates:
                if path.exists():
                    return str(path)
        return None

    def _update_photo_paths(self):
        """Update photo paths for players that don't have one yet."""
        cursor = self.conn.execute(
            "SELECT id, name, jersey_number, photo_path FROM players WHERE active = 1"
        )
        for row in cursor.fetchall():
            if not row[3]:  # no photo_path set
                photo = self._find_player_photo(row[2], row[1])
                if photo:
                    self.conn.execute(
                        "UPDATE players SET photo_path = ? WHERE id = ?",
                        (photo, row[0])
                    )
        self.conn.commit()

    def get_all_players(self):
        cursor = self.conn.execute(
            "SELECT * FROM players WHERE active = 1 ORDER BY jersey_number"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_player(self, player_id: int):
        cursor = self.conn.execute("SELECT * FROM players WHERE id = ?", (player_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_player_by_jersey(self, jersey_number: int):
        cursor = self.conn.execute(
            "SELECT * FROM players WHERE jersey_number = ? AND active = 1",
            (jersey_number,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def add_player(self, name: str, jersey_number: int, photo_path: Optional[str] = None,
                   hair_description: Optional[str] = None):
        self.conn.execute(
            """INSERT INTO players (name, jersey_number, photo_path, hair_description)
               VALUES (?, ?, ?, ?)""",
            (name, jersey_number, photo_path, hair_description)
        )
        self.conn.commit()

    def update_player(self, player_id: int, **kwargs):
        valid_fields = {
            "name", "jersey_number", "photo_path", "face_embedding",
            "gait_signature", "cleat_color_hsv", "relative_height",
            "hair_description", "active"
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values())
        values.append(player_id)

        self.conn.execute(
            f"UPDATE players SET {set_clause} WHERE id = ?", values
        )
        self.conn.commit()

    def update_face_embedding(self, player_id: int, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        self.conn.execute(
            "UPDATE players SET face_embedding = ? WHERE id = ?",
            (blob, player_id)
        )
        self.conn.commit()

    def get_face_embedding(self, player_id: int) -> Optional[np.ndarray]:
        cursor = self.conn.execute(
            "SELECT face_embedding FROM players WHERE id = ?", (player_id,)
        )
        row = cursor.fetchone()
        if row and row["face_embedding"]:
            return np.frombuffer(row["face_embedding"], dtype=np.float32)
        return None

    def get_all_face_embeddings(self) -> dict:
        players = self.get_all_players()
        embeddings = {}
        for p in players:
            if p["face_embedding"]:
                embeddings[p["id"]] = np.frombuffer(
                    p["face_embedding"], dtype=np.float32
                )
        return embeddings

    def update_gait_signature(self, player_id: int, signature: np.ndarray):
        blob = signature.astype(np.float32).tobytes()
        self.conn.execute(
            "UPDATE players SET gait_signature = ? WHERE id = ?",
            (blob, player_id)
        )
        self.conn.commit()

    def get_gait_signature(self, player_id: int) -> Optional[np.ndarray]:
        cursor = self.conn.execute(
            "SELECT gait_signature FROM players WHERE id = ?", (player_id,)
        )
        row = cursor.fetchone()
        if row and row["gait_signature"]:
            return np.frombuffer(row["gait_signature"], dtype=np.float32)
        return None

    def update_cleat_color(self, player_id: int, hsv: dict):
        self.conn.execute(
            "UPDATE players SET cleat_color_hsv = ? WHERE id = ?",
            (json.dumps(hsv), player_id)
        )
        self.conn.commit()

    def get_cleat_color(self, player_id: int) -> Optional[dict]:
        cursor = self.conn.execute(
            "SELECT cleat_color_hsv FROM players WHERE id = ?", (player_id,)
        )
        row = cursor.fetchone()
        if row and row["cleat_color_hsv"]:
            return json.loads(row["cleat_color_hsv"])
        return None

    def delete_player(self, player_id: int):
        self.conn.execute(
            "UPDATE players SET active = 0 WHERE id = ?", (player_id,)
        )
        self.conn.commit()

    # Field operations
    def save_field(self, name: str, field_length_m: float, field_width_m: float,
                   src_points: list, dst_points: list, homography_matrix: list):
        self.conn.execute(
            """INSERT OR REPLACE INTO fields
               (name, field_length_m, field_width_m, src_points, dst_points, homography_matrix)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, field_length_m, field_width_m,
             json.dumps(src_points), json.dumps(dst_points),
             json.dumps(homography_matrix))
        )
        self.conn.commit()

    def get_field(self, name: str) -> Optional[dict]:
        cursor = self.conn.execute("SELECT * FROM fields WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d["src_points"] = json.loads(d["src_points"]) if d["src_points"] else None
            d["dst_points"] = json.loads(d["dst_points"]) if d["dst_points"] else None
            d["homography_matrix"] = json.loads(d["homography_matrix"]) if d["homography_matrix"] else None
            return d
        return None

    def get_all_fields(self) -> list:
        cursor = self.conn.execute("SELECT * FROM fields ORDER BY name")
        fields = []
        for row in cursor.fetchall():
            d = dict(row)
            d["src_points"] = json.loads(d["src_points"]) if d["src_points"] else None
            d["dst_points"] = json.loads(d["dst_points"]) if d["dst_points"] else None
            d["homography_matrix"] = json.loads(d["homography_matrix"]) if d["homography_matrix"] else None
            fields.append(d)
        return fields

    def close(self):
        self.conn.close()
