"""Match history storage and retrieval."""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from config import DB_PATH


class MatchDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                opponent TEXT,
                field_id INTEGER,
                video_path TEXT,
                result TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS player_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER REFERENCES matches(id),
                player_id INTEGER,
                player_name TEXT,
                jersey_number INTEGER,
                team TEXT,
                distance_m REAL,
                top_speed_ms REAL,
                avg_speed_ms REAL,
                sprints_count INTEGER,
                sprint_distance_m REAL,
                pct_att_third REAL,
                pct_mid_third REAL,
                pct_def_third REAL,
                minutes_played REAL,
                passes_made INTEGER,
                passes_received INTEGER,
                identification_confidence REAL,
                stints INTEGER,
                avg_x REAL,
                avg_y REAL,
                positional_spread_m REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER REFERENCES matches(id),
                player_id INTEGER REFERENCES players(id),
                source TEXT DEFAULT 'playermaker',
                distance_km REAL,
                sprint_distance_m REAL,
                top_speed_kmh REAL,
                ball_touches INTEGER,
                touches_left INTEGER,
                touches_right INTEGER,
                first_touch_score REAL,
                time_on_ball_s REAL,
                release_time_s REAL,
                kick_power_kmh REAL,
                two_footed_pct REAL,
                weak_foot_pct REAL,
                accelerations INTEGER,
                decelerations INTEGER,
                direction_changes INTEGER,
                match_score REAL,
                raw_json TEXT,
                screenshot_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def save_match(self, date: str, opponent: str, field_id: Optional[int] = None,
                   video_path: Optional[str] = None, result: Optional[str] = None,
                   notes: Optional[str] = None) -> int:
        cursor = self.conn.execute(
            """INSERT INTO matches (date, opponent, field_id, video_path, result, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (date, opponent, field_id, video_path, result, notes)
        )
        self.conn.commit()
        return cursor.lastrowid

    def save_player_stats(self, match_id: int, stats_list: list):
        for stats in stats_list:
            self.conn.execute(
                """INSERT INTO player_match_stats
                   (match_id, player_id, player_name, jersey_number, team,
                    distance_m, top_speed_ms, avg_speed_ms, sprints_count,
                    sprint_distance_m, pct_att_third, pct_mid_third, pct_def_third,
                    minutes_played, passes_made, passes_received,
                    identification_confidence, stints, avg_x, avg_y, positional_spread_m)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (match_id, stats.get("player_id"), stats.get("name"),
                 stats.get("jersey_number"), stats.get("team"),
                 stats.get("distance_m", 0), stats.get("top_speed_ms", 0),
                 stats.get("avg_speed_ms", 0), stats.get("sprint_count", 0),
                 stats.get("sprint_distance_m", 0),
                 stats.get("pct_att_third", 0), stats.get("pct_mid_third", 0),
                 stats.get("pct_def_third", 0), stats.get("minutes_played", 0),
                 stats.get("passes_made", 0), stats.get("passes_received", 0),
                 stats.get("id_confidence", 0), stats.get("stints", 1),
                 stats.get("avg_x", 0), stats.get("avg_y", 0),
                 stats.get("positional_spread_m", 0))
            )
        self.conn.commit()

    def get_match(self, match_id: int) -> Optional[dict]:
        cursor = self.conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_matches(self) -> list:
        cursor = self.conn.execute("SELECT * FROM matches ORDER BY date DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_match_stats(self, match_id: int) -> pd.DataFrame:
        cursor = self.conn.execute(
            "SELECT * FROM player_match_stats WHERE match_id = ? ORDER BY team, jersey_number",
            (match_id,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_player_season_stats(self, player_id: int) -> pd.DataFrame:
        cursor = self.conn.execute(
            """SELECT pms.*, m.date, m.opponent
               FROM player_match_stats pms
               JOIN matches m ON pms.match_id = m.id
               WHERE pms.player_id = ?
               ORDER BY m.date""",
            (player_id,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_all_season_stats(self) -> pd.DataFrame:
        cursor = self.conn.execute(
            """SELECT pms.*, m.date, m.opponent
               FROM player_match_stats pms
               JOIN matches m ON pms.match_id = m.id
               ORDER BY m.date, pms.team, pms.jersey_number"""
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_playing_time_summary(self) -> pd.DataFrame:
        cursor = self.conn.execute(
            """SELECT pms.player_name, pms.jersey_number, m.date, m.opponent,
                      pms.minutes_played
               FROM player_match_stats pms
               JOIN matches m ON pms.match_id = m.id
               WHERE pms.team = 'Home'
               ORDER BY pms.player_name, m.date"""
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def delete_match(self, match_id: int):
        self.conn.execute("DELETE FROM sensor_data WHERE match_id = ?", (match_id,))
        self.conn.execute("DELETE FROM player_match_stats WHERE match_id = ?", (match_id,))
        self.conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))
        self.conn.commit()

    # --- Sensor Data (PlayerMaker, etc.) ---

    def save_sensor_data(self, match_id: int, player_id: int, data: dict,
                         source: str = "playermaker",
                         screenshot_path: str = None) -> int:
        raw_json = json.dumps(data)
        cursor = self.conn.execute(
            """INSERT INTO sensor_data
               (match_id, player_id, source,
                distance_km, sprint_distance_m, top_speed_kmh,
                ball_touches, touches_left, touches_right,
                first_touch_score, time_on_ball_s, release_time_s,
                kick_power_kmh, two_footed_pct, weak_foot_pct,
                accelerations, decelerations, direction_changes,
                match_score, raw_json, screenshot_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (match_id, player_id, source,
             data.get("distance_km"), data.get("sprint_distance_m"),
             data.get("top_speed_kmh"),
             data.get("ball_touches"), data.get("touches_left"),
             data.get("touches_right"),
             data.get("first_touch_score"), data.get("time_on_ball_s"),
             data.get("release_time_s"),
             data.get("kick_power_kmh"), data.get("two_footed_pct"),
             data.get("weak_foot_pct"),
             data.get("accelerations"), data.get("decelerations"),
             data.get("direction_changes"),
             data.get("match_score"), raw_json, screenshot_path)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_sensor_data(self, match_id: int, player_id: int,
                        source: str = "playermaker") -> Optional[dict]:
        cursor = self.conn.execute(
            """SELECT * FROM sensor_data
               WHERE match_id = ? AND player_id = ? AND source = ?
               ORDER BY id DESC LIMIT 1""",
            (match_id, player_id, source)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_player_sensor_history(self, player_id: int,
                                  source: str = "playermaker") -> pd.DataFrame:
        cursor = self.conn.execute(
            """SELECT sd.*, m.date, m.opponent
               FROM sensor_data sd
               JOIN matches m ON sd.match_id = m.id
               WHERE sd.player_id = ? AND sd.source = ?
               ORDER BY m.date""",
            (player_id, source)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def delete_sensor_data(self, sensor_id: int):
        self.conn.execute("DELETE FROM sensor_data WHERE id = ?", (sensor_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
