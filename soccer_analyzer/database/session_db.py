"""Training session database — stores session metadata, drill segments, and player metrics."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from config import DATA_DIR

SESSION_DB_PATH = DATA_DIR / "sessions.db"


class SessionDB:
    """CRUD for training sessions, drill segments, and per-player drill metrics."""

    def __init__(self, db_path: Path = SESSION_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                session_type TEXT NOT NULL CHECK(session_type IN ('game', 'scrimmage', 'drill')),
                video_path TEXT,
                calibration_path TEXT,
                field_length_m REAL,
                field_width_m REAL,
                flag_color TEXT,
                duration_s REAL,
                total_players INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS drill_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                segment_index INTEGER NOT NULL,
                start_frame INTEGER,
                end_frame INTEGER,
                start_time_s REAL,
                end_time_s REAL,
                duration_s REAL,
                drill_type TEXT,
                avg_intensity REAL,
                player_count INTEGER,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS player_session_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                segment_id INTEGER REFERENCES drill_segments(id) ON DELETE CASCADE,
                player_name TEXT,
                jersey_number INTEGER,
                track_id INTEGER,
                team TEXT,
                distance_m REAL,
                avg_speed_ms REAL,
                top_speed_ms REAL,
                sprint_count INTEGER,
                sprint_distance_m REAL,
                active_pct REAL,
                area_covered_m2 REAL,
                high_intensity_pct REAL,
                drill_type TEXT
            );

            CREATE TABLE IF NOT EXISTS session_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
                total_distance_m REAL,
                avg_team_speed_ms REAL,
                total_sprints INTEGER,
                total_drills INTEGER,
                high_intensity_pct REAL,
                formation TEXT,
                passes_total INTEGER,
                summary_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
            CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type);
            CREATE INDEX IF NOT EXISTS idx_drill_segments_session ON drill_segments(session_id);
            CREATE INDEX IF NOT EXISTS idx_player_metrics_session ON player_session_metrics(session_id);
        """)
        self.conn.commit()

    # ─── Sessions CRUD ─────────────────────────────────────────────────────────

    def create_session(
        self,
        date: str,
        session_type: str,
        video_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        field_length_m: Optional[float] = None,
        field_width_m: Optional[float] = None,
        flag_color: Optional[str] = None,
        duration_s: Optional[float] = None,
        total_players: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Create a new session record. Returns session ID."""
        cursor = self.conn.execute(
            """INSERT INTO sessions (date, session_type, video_path, calibration_path,
               field_length_m, field_width_m, flag_color, duration_s, total_players, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, session_type, video_path, calibration_path,
             field_length_m, field_width_m, flag_color, duration_s, total_players, notes),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_session(self, session_id: int) -> Optional[dict]:
        """Get a single session by ID."""
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, session_type: Optional[str] = None, limit: int = 50) -> List[dict]:
        """List sessions, optionally filtered by type."""
        if session_type:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE session_type = ? ORDER BY date DESC LIMIT ?",
                (session_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: int):
        """Delete session and all related data (cascades)."""
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    # ─── Drill Segments ────────────────────────────────────────────────────────

    def add_drill_segment(
        self,
        session_id: int,
        segment_index: int,
        start_frame: int,
        end_frame: int,
        start_time_s: float,
        end_time_s: float,
        duration_s: float,
        drill_type: str,
        avg_intensity: float = 0.0,
        player_count: int = 0,
        notes: Optional[str] = None,
    ) -> int:
        """Add a drill segment to a session. Returns segment ID."""
        cursor = self.conn.execute(
            """INSERT INTO drill_segments (session_id, segment_index, start_frame, end_frame,
               start_time_s, end_time_s, duration_s, drill_type, avg_intensity, player_count, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, segment_index, start_frame, end_frame,
             start_time_s, end_time_s, duration_s, drill_type, avg_intensity, player_count, notes),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_drill_segments(self, session_id: int) -> List[dict]:
        """Get all drill segments for a session."""
        rows = self.conn.execute(
            "SELECT * FROM drill_segments WHERE session_id = ? ORDER BY segment_index",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Player Metrics ────────────────────────────────────────────────────────

    def add_player_metrics(
        self,
        session_id: int,
        segment_id: Optional[int],
        player_name: Optional[str],
        jersey_number: Optional[int],
        track_id: Optional[int],
        team: Optional[str] = None,
        distance_m: float = 0.0,
        avg_speed_ms: float = 0.0,
        top_speed_ms: float = 0.0,
        sprint_count: int = 0,
        sprint_distance_m: float = 0.0,
        active_pct: float = 0.0,
        area_covered_m2: float = 0.0,
        high_intensity_pct: float = 0.0,
        drill_type: Optional[str] = None,
    ) -> int:
        """Add player metrics for a session/segment."""
        cursor = self.conn.execute(
            """INSERT INTO player_session_metrics (session_id, segment_id, player_name,
               jersey_number, track_id, team, distance_m, avg_speed_ms, top_speed_ms,
               sprint_count, sprint_distance_m, active_pct, area_covered_m2,
               high_intensity_pct, drill_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, segment_id, player_name, jersey_number, track_id, team,
             distance_m, avg_speed_ms, top_speed_ms, sprint_count, sprint_distance_m,
             active_pct, area_covered_m2, high_intensity_pct, drill_type),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_player_metrics(self, session_id: int, segment_id: Optional[int] = None) -> List[dict]:
        """Get player metrics for a session, optionally filtered by segment."""
        if segment_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM player_session_metrics WHERE session_id = ? AND segment_id = ?",
                (session_id, segment_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM player_session_metrics WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Session Summary ───────────────────────────────────────────────────────

    def save_summary(
        self,
        session_id: int,
        total_distance_m: float = 0.0,
        avg_team_speed_ms: float = 0.0,
        total_sprints: int = 0,
        total_drills: int = 0,
        high_intensity_pct: float = 0.0,
        formation: Optional[str] = None,
        passes_total: int = 0,
        summary_json: Optional[str] = None,
    ):
        """Upsert session summary."""
        self.conn.execute(
            """INSERT INTO session_summary (session_id, total_distance_m, avg_team_speed_ms,
               total_sprints, total_drills, high_intensity_pct, formation, passes_total, summary_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   total_distance_m = excluded.total_distance_m,
                   avg_team_speed_ms = excluded.avg_team_speed_ms,
                   total_sprints = excluded.total_sprints,
                   total_drills = excluded.total_drills,
                   high_intensity_pct = excluded.high_intensity_pct,
                   formation = excluded.formation,
                   passes_total = excluded.passes_total,
                   summary_json = excluded.summary_json""",
            (session_id, total_distance_m, avg_team_speed_ms, total_sprints,
             total_drills, high_intensity_pct, formation, passes_total, summary_json),
        )
        self.conn.commit()

    def get_summary(self, session_id: int) -> Optional[dict]:
        """Get session summary."""
        row = self.conn.execute(
            "SELECT * FROM session_summary WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    # ─── Aggregation Queries ───────────────────────────────────────────────────

    def get_player_history(self, jersey_number: int, limit: int = 20) -> List[dict]:
        """Get a player's metrics across recent sessions (for season progress)."""
        rows = self.conn.execute(
            """SELECT s.date, s.session_type, pm.distance_m, pm.avg_speed_ms,
                      pm.top_speed_ms, pm.sprint_count, pm.active_pct
               FROM player_session_metrics pm
               JOIN sessions s ON s.id = pm.session_id
               WHERE pm.jersey_number = ?
               ORDER BY s.date DESC LIMIT ?""",
            (jersey_number, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_comparison(self, session_type: str = "drill", limit: int = 10) -> List[dict]:
        """Compare recent sessions of the same type."""
        rows = self.conn.execute(
            """SELECT s.id, s.date, ss.total_distance_m, ss.avg_team_speed_ms,
                      ss.total_sprints, ss.total_drills, ss.high_intensity_pct
               FROM sessions s
               LEFT JOIN session_summary ss ON ss.session_id = s.id
               WHERE s.session_type = ?
               ORDER BY s.date DESC LIMIT ?""",
            (session_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
