# ⚽ U10 Soccer Analysis App — Complete Final Spec
### Prompt for Claude Opus 4 — End-to-End Build on MacBook

---

## CONTEXT & GOAL

I am a **data scientist and U10 soccer coach** on a MacBook (Apple Silicon M-series). I have:
- **iPhone 15 Pro Max** with **BallerCam** camera — records games in **180° Panoramic View**
- A **team roster of 16 players** with face photos and jersey numbers (stored as image files)
- Games are **7v7**, ~40 minutes, with **4-5 substitutes rotating every 5-7 minutes**
- All players wear **identical green jersey + shorts + socks** — only cleats differ per player

I want a **local Streamlit app** that:
1. Accepts a BallerCam Panoramic video
2. Automatically identifies every player by name using face recognition + multi-modal fingerprinting
3. Correctly re-identifies players after substitutions despite identical kits
4. Generates per-player AND team-level tactical analysis
5. Runs 100% locally — zero cloud, zero subscriptions, zero ongoing cost

---

## FULL FILE STRUCTURE

Build every file completely — no placeholders, no TODOs.

```
soccer_analyzer/
├── app.py                          # Main Streamlit entry point
├── config.py                       # Global constants and paths
├── requirements.txt
├── setup_instructions.md
│
├── data/
│   ├── roster.db                   # SQLite database (auto-created on first run)
│   ├── fields/                     # Saved homography calibrations per field
│   │   └── .gitkeep
│   └── seasons/                    # Historical match stats per season
│       └── .gitkeep
│
├── pipeline/
│   ├── __init__.py
│   ├── detector.py                 # YOLOv8 detection + ByteTrack tracking
│   ├── homography.py               # Pixel → real field coordinate mapping
│   ├── team_classifier.py          # KMeans jersey color → team assignment
│   ├── face_reid.py                # InsightFace face recognition + matching
│   ├── gait.py                     # MediaPipe pose → gait signature
│   ├── cleat.py                    # Cleat color extraction + matching
│   ├── fingerprint.py              # Multi-modal fusion: face+gait+cleat+height
│   ├── jersey_ocr.py               # Targeted jersey number classification
│   ├── passes.py                   # Pass inference from tracking data
│   ├── stats.py                    # Per-player stat calculations
│   └── formation.py                # Formation detection + labeling
│
├── database/
│   ├── __init__.py
│   ├── roster_db.py                # Roster CRUD operations
│   └── match_db.py                 # Match history storage + retrieval
│
├── visualization/
│   ├── __init__.py
│   ├── heatmaps.py                 # Per-player heatmap plots
│   ├── passing_network.py          # Passing network plots
│   ├── pitch_overview.py           # All players average positions
│   ├── timeline.py                 # Per-player time-in-zone timeline
│   ├── formation_plot.py           # Formation shape visualization
│   └── report.py                   # PDF report generator
│
└── pages/
    ├── roster_manager.py           # Streamlit page: manage players
    ├── field_calibration.py        # Streamlit page: homography setup
    ├── match_analysis.py           # Streamlit page: main analysis
    └── season_progress.py          # Streamlit page: season trends
```

---

## ENVIRONMENT & DEPENDENCIES

### `requirements.txt`
```
streamlit>=1.32.0
ultralytics>=8.0.0
supervision>=0.19.0
opencv-python>=4.9.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
mplsoccer>=1.3.0
scipy>=1.10.0
scikit-learn>=1.3.0
insightface>=0.7.3
mediapipe>=0.10.0
onnxruntime>=1.16.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
reportlab>=4.0.0
fpdf2>=2.7.0
sqlite3
```

### Apple Silicon Setup Note
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# PyTorch with MPS support (Apple Silicon)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# InsightFace requires cmake
brew install cmake
pip install insightface

# All other dependencies
pip install -r requirements.txt
```

### Device Detection (use in every model initialization)
```python
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"      # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = "cuda"     # NVIDIA GPU
else:
    DEVICE = "cpu"
```

---

## DATABASE DESIGN

### `database/roster_db.py`

SQLite tables:

```sql
-- Players table
CREATE TABLE players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    jersey_number INTEGER NOT NULL UNIQUE,
    photo_path TEXT,                    -- path to roster photo
    face_embedding BLOB,                -- InsightFace 512-dim float32 array
    gait_signature BLOB,                -- 128-dim gait embedding
    cleat_color_hsv TEXT,               -- JSON: {"h": 45, "s": 200, "v": 180}
    relative_height REAL,               -- normalized 0-1 within team
    hair_description TEXT,              -- "blonde short", "dark curly" etc
    active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fields table (saved homography calibrations)
CREATE TABLE fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,          -- e.g. "Riverside Park Field 2"
    field_length_m REAL,
    field_width_m REAL,
    src_points TEXT,                    -- JSON: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    dst_points TEXT,                    -- JSON: real-world meter coords
    homography_matrix TEXT,             -- JSON: 3x3 matrix
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    opponent TEXT,
    field_id INTEGER REFERENCES fields(id),
    video_path TEXT,
    result TEXT,                        -- "3-1 W", "0-2 L" etc
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player match stats table
CREATE TABLE player_match_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id),
    player_id INTEGER REFERENCES players(id),
    distance_m REAL,
    top_speed_ms REAL,
    avg_speed_ms REAL,
    sprints_count INTEGER,
    pct_att_third REAL,
    pct_mid_third REAL,
    pct_def_third REAL,
    minutes_played REAL,
    passes_made INTEGER,
    passes_received INTEGER,
    identification_confidence REAL,     -- avg confidence of ID across game
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## MODULE SPECS

---

### `pipeline/detector.py`

**Purpose:** Process video, detect all persons, track with consistent IDs per stint.

```python
class VideoDetector:
    def __init__(self, model_size="n", confidence=0.35, sample_rate=3):
        # model_size: "n"=nano(fast), "s"=small, "x"=xlarge(accurate)
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.tracker = sv.ByteTrack()
        self.confidence = confidence
        self.sample_rate = sample_rate

    def process(self, video_path, progress_callback=None):
        """
        Returns:
            detections_df: DataFrame with columns:
                frame | time_s | track_id | x_px | y_px |
                bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 |
                frame_h | frame_w | fps
            video_meta: dict with fps, total_frames, duration_s, frame_h, frame_w
        """
```

**Key behaviors:**
- Use `classes=[0]` (person only)
- Store full bounding box for downstream cropping (face region, cleat region, jersey region)
- Log `frame_h` and `frame_w` on first frame for homography
- Call `progress_callback(frame_id, total_frames, detections_so_far)` every 30 frames
- Cache model with `@st.cache_resource`
- Return raw detections — do NOT assign player names here (that's fingerprint.py's job)

---

### `pipeline/homography.py`

**Purpose:** Convert pixel coordinates to real field coordinates in meters.

```python
class FieldHomography:
    def __init__(self, field_length_m, field_width_m):
        self.H = None
        self.field_length = field_length_m
        self.field_width = field_width_m

    def calibrate_auto(self, frame_h, frame_w):
        """Estimate homography from video dimensions."""
        # 8% horizontal margin, 12% vertical margin
        # Accounts for typical BallerCam mid-field sideline placement

    def calibrate_manual(self, src_points, dst_points):
        """Compute homography from 4 manually clicked field points."""
        # src_points: 4 pixel coords clicked by user
        # dst_points: corresponding real-world meter coords
        # dst_points for U10 7v7: corners = (0,0),(50,0),(50,35),(0,35)

    def transform(self, x_px, y_px):
        """Convert single pixel point to field meters. Clamp to field bounds."""

    def transform_df(self, df):
        """Apply transform to entire detections DataFrame. Add x_field, y_field cols."""

    def save(self, field_name, db_connection):
        """Save calibration to fields table in SQLite."""

    def load(self, field_name, db_connection):
        """Load calibration from SQLite by field name."""

    def validate(self, df):
        """
        Check that transformed coordinates make sense.
        Warn if >20% of points fall outside field boundaries.
        Return validation_score 0-1 and warning message.
        """
```

**Homography save/load flow in the UI:**
1. User selects field from dropdown (populated from SQLite `fields` table)
2. If field exists → load saved homography → skip calibration
3. If new field → show calibration UI → save after completion
4. "Recalibrate" button always available to redo any field

---

### `pipeline/face_reid.py`

**Purpose:** Extract face embeddings from roster photos and match against faces detected in video frames.

```python
class FaceReID:
    def __init__(self):
        # Use InsightFace buffalo_l model — best accuracy, runs locally
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]  # MPS not yet supported by onnxruntime
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.roster_embeddings = {}  # {player_id: 512-dim np.array}

    def build_roster_embeddings(self, players):
        """
        For each player in roster:
        1. Load their photo
        2. Detect face
        3. Extract 512-dim ArcFace embedding
        4. Store in self.roster_embeddings AND save to DB
        Called once when roster is set up or player photo changes.
        """

    def extract_face_embedding(self, frame, bbox):
        """
        Crop face region from top 35% of bounding box.
        Detect face within crop.
        Return 512-dim embedding or None if no clear face found.
        """

    def match(self, embedding, threshold=0.4):
        """
        Compare embedding against all roster embeddings using cosine similarity.
        Returns:
            best_match_player_id: int or None
            confidence: float 0-1
            all_scores: dict {player_id: similarity_score}
        Threshold: similarity > 0.4 = same person (InsightFace standard)
        """

    def batch_match_video(self, detections_df, frames_dir):
        """
        Process all frames, attempt face match for each track_id.
        Aggregate matches per track_id (vote on most frequent match).
        Returns: {track_id: {player_id, confidence, face_match_count}}
        """
```

**Critical implementation notes:**
- Extract face from **top 30-35% of bounding box** — not the whole bbox
- Only attempt face matching when bounding box height > 60 pixels (player close enough)
- Aggregate votes: if track_id matches player_X in 15 out of 20 face detection attempts → assign player_X with high confidence
- Face matching fires **in parallel** with other fingerprinting — not sequentially

---

### `pipeline/gait.py`

**Purpose:** Build per-player gait signature from skeleton keypoint sequences.

```python
class GaitAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, frame, bbox):
        """
        Crop player region, run MediaPipe Pose.
        Return 33 normalized keypoint coords as flat array (66 values).
        Return None if pose not detected.
        """

    def build_gait_signature(self, keypoint_sequence):
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

    def similarity(self, sig1, sig2):
        """Cosine similarity between two gait signatures. Returns 0-1."""

    def build_roster_signatures(self, players, training_video_path=None):
        """
        Build gait signatures for roster players.
        If training_video_path provided: extract from warmup footage.
        Otherwise: build progressively during first game (first 5 minutes).
        Save to DB.
        """
```

**Implementation note:** Gait signatures need **at least 60 consecutive frames** of a player running to be reliable. Flag players with insufficient running footage as "gait signature pending."

---

### `pipeline/cleat.py`

**Purpose:** Extract cleat color as a player fingerprint signal.

```python
class CleatExtractor:
    def extract_cleat_color(self, frame, bbox):
        """
        Crop bottom 20% of bounding box (foot/cleat region).
        Convert to HSV.
        Use KMeans k=2 to find dominant non-grass color
        (grass is green HSV(35-85, 40+, 40+) — exclude it).
        Return dominant HSV values as dict {"h": x, "s": y, "v": z}.
        """

    def build_cleat_profiles(self, detections_df, frames):
        """
        For each track_id, aggregate cleat color samples across 50+ frames.
        Return stable cleat color per track_id.
        """

    def similarity(self, hsv1, hsv2):
        """
        Compare two cleat colors.
        Weight hue heavily, saturation medium, value low.
        Return 0-1 similarity.
        Note: black cleats (low saturation) match differently — handle separately.
        """

    def build_roster_profiles(self, players):
        """
        If roster has cleat color manually entered → use that.
        Otherwise: extract from first game automatically.
        Save to DB.
        """
```

---

### `pipeline/fingerprint.py`

**Purpose:** Fuse all identification signals into a single player identity decision.

```python
class PlayerFingerprinter:
    """
    Multi-modal player identification engine.
    Combines face Re-ID, gait, cleat color, height, hair into unified ID.
    """

    WEIGHTS = {
        "face":   0.50,   # Highest weight — most distinctive
        "gait":   0.25,   # High weight — very individual
        "cleat":  0.15,   # Medium weight — distinctive per player
        "height": 0.07,   # Low weight — similar across U10 age group
        "hair":   0.03,   # Low weight — changes seasonally
    }

    def __init__(self, roster_db, face_reid, gait_analyzer, cleat_extractor):
        self.db = roster_db
        self.face = face_reid
        self.gait = gait_analyzer
        self.cleat = cleat_extractor

    def identify_track(self, track_id, detections_df, frames):
        """
        For a given track_id:
        1. Attempt face matching → get face_scores per player
        2. Compare gait signature → get gait_scores per player
        3. Compare cleat color → get cleat_scores per player
        4. Compare relative height → get height_scores per player
        5. Weighted fusion of all scores
        6. Return best match + confidence

        Returns:
            player_id: int (matched roster player) or None
            confidence: float 0-1
            breakdown: dict showing contribution of each signal
            needs_confirmation: bool (True if confidence 0.5-0.8)
        """

    def identify_all_tracks(self, detections_df, frames, progress_callback=None):
        """
        Run identify_track for every unique track_id in detections_df.
        
        Auto-assign if confidence > 0.80
        Flag for confirmation if confidence 0.50 - 0.80
        Mark as unknown if confidence < 0.50
        
        Returns:
            assignments: dict {track_id: {player_id, confidence, status}}
            pending_confirmations: list of {track_id, candidates, crops}
        """

    def merge_track_ids(self, assignments, detections_df):
        """
        After all track_ids are identified:
        Merge all track_ids belonging to same player_id into unified timeline.
        
        Example: track_ids 3, 17, 24 all = Khalid Yaacoub #10
        → merge into single player record with combined stats
        
        Returns detections_df with added columns:
            player_id | player_name | jersey_number | id_confidence
        """

    def get_relative_heights(self, detections_df):
        """
        Estimate relative height of each track_id from bounding box heights.
        Normalize within team (0 = shortest, 1 = tallest).
        Use 75th percentile of bbox heights (avoids crouching/jumping noise).
        """
```

---

### `pipeline/jersey_ocr.py`

**Purpose:** Targeted jersey number recognition — knowing we're looking for specific numbers makes this tractable.

```python
class JerseyOCR:
    KNOWN_NUMBERS = [3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21]

    def __init__(self):
        # Use EasyOCR for number detection
        # pip install easyocr
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False)

    def extract_jersey_region(self, frame, bbox):
        """
        Crop chest/back region: middle 40% height, center 60% width of bbox.
        Return cropped image.
        """

    def read_number(self, jersey_crop):
        """
        Run EasyOCR on jersey crop.
        Filter results to only accept numbers in KNOWN_NUMBERS list.
        Return (number, confidence) or (None, 0) if no valid number found.
        """

    def batch_read(self, detections_df, frames, sample_every=15):
        """
        Attempt OCR on every Nth frame per track_id.
        Vote on most frequent valid number read.
        Return {track_id: {jersey_number, confidence, read_count}}
        
        This supplements face Re-ID — does NOT replace it.
        Used as confirmation signal in fingerprint fusion.
        """
```

**Note:** OCR is a supplementary signal only. Accuracy ~60-70% due to motion blur, occlusion, and small text. When OCR number matches face Re-ID assignment → confidence boost. When they conflict → flag for manual review.

---

### `pipeline/passes.py`

**Purpose:** Infer passes between identified players.

```python
class PassDetector:
    def detect_passes(self, detections_df, fps, field_length, field_width):
        """
        Algorithm:
        1. Smooth player positions with rolling window (reduce jitter)
        2. Estimate ball position as centroid of closest player cluster
        3. Detect "possession change events":
           - Player A has high velocity directed toward another player
           - Followed within 2 seconds by Player B gaining similar velocity
           - Both on same team
           - Distance between them at time of pass < 25m
        4. Filter: minimum 3m distance (not a tackle/dribble)
        
        Returns: list of {
            timestamp_s, passer_name, receiver_name, team,
            passer_pos, receiver_pos, pass_distance_m
        }
        """

    def build_pass_matrix(self, passes, players):
        """
        NxN matrix where N = number of players on one team.
        Value = number of passes between player i and player j.
        Return as pandas DataFrame with player names as index/columns.
        """
```

**Disclaimer to show in UI:** *"Passing network is AI-inferred from movement patterns without dedicated ball tracking. Directional patterns are reliable; exact pass counts may vary ±20%."*

---

### `pipeline/stats.py`

**Purpose:** Calculate per-player statistics from identified tracking data.

**Input:** detections_df with `player_name`, `jersey_number`, `x_field`, `y_field`, `time_s`

**Output columns per player:**

```python
{
    "jersey_number": int,
    "name": str,
    "team": str,                    # "Home" or "Away"
    "minutes_played": float,        # actual time on field
    "distance_m": float,            # total distance covered
    "top_speed_ms": float,          # 95th percentile speed
    "avg_speed_ms": float,
    "sprint_count": int,            # frames where speed > 4.0 m/s
    "sprint_distance_m": float,
    "pct_att_third": float,
    "pct_mid_third": float,
    "pct_def_third": float,
    "avg_x": float,                 # average field position
    "avg_y": float,
    "positional_spread_m": float,   # std dev of positions
    "passes_made": int,
    "passes_received": int,
    "id_confidence": float,         # avg identification confidence
    "stints": int,                  # number of times entered field
}
```

**Sprint threshold:** 4.0 m/s for U10 (realistic youth sprint speed)
**Speed noise filter:** cap at 10 m/s, use rolling median over 5 frames
**Minutes played:** calculated from actual frame timestamps, not game clock

---

### `pipeline/formation.py`

**Purpose:** Detect and label team formation at any point in the game.

```python
class FormationDetector:
    # U10 7v7 common formations
    FORMATIONS_7V7 = {
        "2-3-1": [(0.2, 0.3), (0.2, 0.7), (0.5, 0.2), (0.5, 0.5),
                  (0.5, 0.8), (0.75, 0.5), (0.9, 0.5)],
        "3-2-1": [...],
        "2-2-2": [...],
        "3-1-2": [...],
        "1-3-2": [...],
    }

    def detect_formation(self, team_positions):
        """
        Input: array of (x, y) positions for all players of one team
        1. Sort players by x position (defensive → attacking)
        2. Use KMeans to cluster into defensive/mid/attacking lines
        3. Count players per line
        4. Match to closest known formation template
        5. Return formation string e.g. "2-3-1" + confidence
        """

    def formation_over_time(self, detections_df, window_seconds=60):
        """
        Slide a 60-second window across the game.
        Detect formation in each window.
        Return timeline of formation changes.
        """

    def compactness_over_time(self, detections_df, window_seconds=30):
        """
        Calculate team compactness (average inter-player distance) over time.
        U10 teams famously bunch around the ball — this makes it quantifiable.
        Return time series of compactness values for both teams.
        """
```

---

## STREAMLIT APP LAYOUT

### `app.py` — Navigation

Use `st.navigation` with 4 pages:

```python
pages = {
    "Analysis": [
        st.Page("pages/match_analysis.py", title="Match Analysis", icon="⚽"),
        st.Page("pages/season_progress.py", title="Season Progress", icon="📈"),
    ],
    "Setup": [
        st.Page("pages/roster_manager.py", title="Roster Manager", icon="👥"),
        st.Page("pages/field_calibration.py", title="Field Setup", icon="🗺️"),
    ],
}
```

---

### `pages/roster_manager.py`

**Sections:**

**1. Current Roster Table**
Show all 16 players: photo thumbnail | name | jersey # | face embedding status | gait status | cleat color swatch

**2. Add / Edit Player**
- Name input
- Jersey number input
- Photo upload (accepts JPEG/PNG)
- On save: auto-extract face embedding using InsightFace → store in DB
- Cleat color picker (optional manual override)
- Hair description text field

**3. Batch Import**
- Upload ZIP of photos named `{jersey_number}_{name}.jpg` → auto-import all
- Show progress per player

**Pre-loaded roster (hard-code as default seed data):**
```python
INITIAL_ROSTER = [
    {"name": "Ben Adam",         "jersey": 3},
    {"name": "Vince Sharma",     "jersey": 5},
    {"name": "Maverick Cardoso", "jersey": 6},
    {"name": "Liam Gibala",      "jersey": 7},
    {"name": "Ben Hahn",         "jersey": 8},
    {"name": "Nolan Bowser",     "jersey": 9},
    {"name": "Khalid Yaacoub",   "jersey": 10},
    {"name": "Issa Hassoun",     "jersey": 11},
    {"name": "Liam Garland",     "jersey": 14},
    {"name": "Arian Rezaeian",   "jersey": 15},
    {"name": "Alexander Kerr",   "jersey": 16},
    {"name": "Jason Qian",       "jersey": 17},
    {"name": "David Shallvari",  "jersey": 18},
    {"name": "Jaedyn Duncan",    "jersey": 19},
    {"name": "Luca Perrotta",    "jersey": 20},
    {"name": "Gabriel Zaidan",   "jersey": 21},
]
```

---

### `pages/field_calibration.py`

**Flow:**

1. **Select or create field**
   - Dropdown of saved fields from SQLite
   - "➕ New Field" button

2. **If new field:**
   - Field name input (e.g. "Riverside Park Field 2")
   - Field length slider (30-70m, default 50)
   - Field width slider (20-50m, default 35)
   - Upload any video from that field OR use first frame of game video
   - Show first frame as interactive image
   - Instruction: *"Click the 4 corner points of the field in order: top-left → top-right → bottom-right → bottom-left"*
   - Use `st.plotly_chart` with click events for point selection
   - Show clicked points with numbered markers
   - "Save Field Calibration" button → writes to SQLite
   - "Test Calibration" button → overlays a grid on the frame showing field meter lines

3. **If existing field:**
   - Show saved calibration info + thumbnail
   - "Use This Field" button
   - "Recalibrate" button

---

### `pages/match_analysis.py`

**Step 1: Match Setup**
```
Date picker | Opponent name | Field selector | Match result (optional)
```

**Step 2: Video Upload**
```
Large dropzone: "Drop BallerCam Panoramic Video (MP4/MOV)"
Show: Duration | Resolution | FPS | Estimated processing time
```

**Step 3: Processing Settings (expandable)**
```
Sample rate: every N frames (default 3)
Detection confidence (default 0.35)
YOLOv8 model size: Nano/Small/Large (default Nano)
```

**Step 4: Analyze Button**
```
Large green "⚽ Analyze Match" button
```

**Step 5: Processing Progress**

Show 5 sequential progress stages:
```
[████████░░] Stage 1/5: Detecting & tracking players (frame X/Y)
[████████░░] Stage 2/5: Building player fingerprints
[████████░░] Stage 3/5: Identifying players
[████████░░] Stage 4/5: Calculating statistics
[████████░░] Stage 5/5: Generating visualizations
```

**Step 6: Confirmation Queue**

If any track IDs have confidence 0.50-0.80:
Show card for each: *"Who is this player?"*
- Small video crop GIF (3 seconds of the player running)
- Candidate list with confidence scores (top 3 candidates)
- Quick-select buttons: player name options + "Unknown"
- "Confirm All" button after reviewing

**Step 7: Results — 6 Tabs**

```
[📊 Overview] [👤 Players] [🔥 Heatmaps] [🔗 Passing] [🏟️ Formation] [⬇️ Export]
```

**Tab 1 — Overview:**
- 6 metric cards: Players Identified / Identification Confidence / Home Possession % / Away Possession % / Total Distance Home / Total Distance Away
- Full pitch average position map (both teams)
- Team compactness chart over time (line chart)
- Formation timeline (both teams, per 5-minute window)

**Tab 2 — Players:**
- Team A and Team B stats tables side by side
- Color coded: green = top performer, red = needs attention (least distance/least active)
- Columns: Name | # | Mins | Distance | Top Speed | Sprints | Att% | Mid% | Def% | Passes | ID Confidence
- Sortable by any column
- Click player row → expands to show their heatmap inline

**Tab 3 — Heatmaps:**
- Player selector dropdown (all identified players)
- Large heatmap of selected player
- Thumbnail grid: all players mini heatmaps (4 per row)
- Toggle: "Show unidentified players" (track IDs that couldn't be matched)

**Tab 4 — Passing:**
- Two columns: Home passing network | Away passing network
- Below: pass matrix table (who passed to whom)
- Top 5 passing combinations per team
- Disclaimer about inference accuracy

**Tab 5 — Formation:**
- Formation timeline bar chart (x=time, color=formation type)
- Compactness chart (how bunched team was over time)
- 4 formation snapshots: 0min / 10min / 20min / 30min
- Coaching note generator: *"Team played 2-3-1 for 70% of game. Compactness dropped in final 10 minutes."*

**Tab 6 — Export:**
- Download Stats CSV
- Download All Heatmaps (ZIP of PNGs)
- Download Passing Networks (ZIP)
- Download PDF Match Report (full auto-generated report)
- "Save to Season History" button → writes to match_db

---

### `pages/season_progress.py`

**Sections:**

**1. Season Summary**
- Games played, wins/losses, total distance per player across season

**2. Player Progress Charts**
- Select player from dropdown
- Line charts over all games: Distance covered / Top Speed / Sprint Count / Time in Attacking Third
- "Is this player improving?" trend indicator

**3. Team Trends**
- Compactness trend over season (are they staying more organized?)
- Formation consistency (which formation do they actually end up in?)
- Possession trend

**4. Playing Time Tracker**
- Bar chart: minutes played per player per game
- Helps ensure fair playing time across the squad

---

## PDF REPORT — `visualization/report.py`

Auto-generate a one-page PDF per match using ReportLab:

**Layout:**
```
[Team Logo Placeholder]  Match vs {Opponent} — {Date}
─────────────────────────────────────────────────
[Team A Stats Table]     [Team B Stats Table]
─────────────────────────────────────────────────
[Home Passing Network]   [Away Passing Network]
─────────────────────────────────────────────────
[Formation Timeline Chart — full width]
─────────────────────────────────────────────────
[Top 3 Player Heatmaps]
─────────────────────────────────────────────────
Coach Notes: ___________________________________
```

---

## VISUAL DESIGN

**Color Palette:**
```python
BG_PRIMARY    = "#0d1117"   # Page background
BG_CARD       = "#161b22"   # Cards, sidebar
BORDER        = "#30363d"   # Borders
TEXT_PRIMARY  = "#e6edf3"   # Main text
TEXT_MUTED    = "#8b949e"   # Secondary text
ACCENT_GREEN  = "#00c853"   # Buttons, highlights, metric values
TEAM_A_BLUE   = "#2196f3"   # Home team
TEAM_B_RED    = "#f44336"   # Away team
PITCH_GRASS   = "#1a2a1a"   # Pitch background
PITCH_LINES   = "#4a7a4a"   # Pitch markings
```

**Typography:**
```css
/* Import in st.markdown */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@300;400;500&display=swap');

Headers: Barlow Condensed 800, uppercase, letter-spacing 0.05em
Body: Barlow 400
Metrics: Barlow Condensed 800, color ACCENT_GREEN
```

**All pitch visualizations:**
- Background: `#1a2a1a` (dark grass)
- Line color: `#4a7a4a` (muted green)
- Team A: blue colormap for heatmaps
- Team B: red colormap for heatmaps
- Never use default white mplsoccer pitch

---

## PERFORMANCE TARGETS

| Video Length | Apple Silicon M1/M2 | Intel Mac |
|---|---|---|
| 20 min (half) | ~4-6 min | ~12-15 min |
| 40 min (full game) | ~8-12 min | ~20-25 min |

**Achieve via:**
- Default sample_rate=3 (every 3rd frame)
- YOLOv8 nano by default
- Face matching only when bbox height > 60px
- Gait extraction only when player detected for 10+ consecutive samples
- `@st.cache_resource` for all models
- `@st.cache_data` for processed results (keyed on video hash + settings)
- Run face Re-ID and gait extraction in parallel using `concurrent.futures.ThreadPoolExecutor`

---

## ERROR HANDLING

| Situation | UI Response |
|---|---|
| No players detected | "No players detected. Lower confidence threshold or check video brightness." |
| Only 1 team found | "Only one team detected. Check team color settings." |
| Face embedding fails for player | "Could not extract face for [Name]. Try a clearer photo facing the camera." |
| InsightFace model download needed | Show spinner: "Downloading face recognition model (first run, ~300MB)..." |
| Homography produces out-of-bounds coords | "Calibration may be off — X% of positions fell outside field boundaries. Consider recalibrating." |
| Video too short | "Video is under 5 minutes. Ensure this is a full game clip." |
| Low ID confidence overall | "Average identification confidence: X%. Consider running the roster setup with a warmup video for better gait signatures." |
| Player not in roster | "Unknown player detected (Track ID #X). Add them to the roster to include in analysis." |

---

## IMPORTANT IMPLEMENTATION NOTES FOR CLAUDE OPUS

1. **Build all files completely** — every function fully implemented, no stubs

2. **InsightFace first run** downloads ~300MB buffalo_l model to `~/.insightface/` — show a clear download progress message

3. **SQLite database** — auto-create all tables on first run if they don't exist. Never crash on missing DB.

4. **Roster pre-seeding** — on first app launch, if DB is empty, auto-insert the 16 players from `INITIAL_ROSTER` list above

5. **Homography orientation** — for mplsoccer custom pitch: x-axis = field length (0 → field_length_m), y-axis = field width (0 → field_width_m). Attacking direction = increasing x. Be consistent across ALL visualizations.

6. **Track ID vs Player ID** — never confuse these. `track_id` = ByteTrack temporary ID per stint. `player_id` = SQLite roster ID. The mapping between them is built by `fingerprint.py` and stored in session state.

7. **Substitution handling** — the same player will have MULTIPLE track_ids across a game. `merge_track_ids()` in fingerprint.py unifies them. All downstream stats and visualizations must use the merged player identity, not raw track_ids.

8. **mplsoccer figures** — always return `fig` object from plot functions, never call `plt.show()`. Use `st.pyplot(fig)` in Streamlit. Always call `plt.close(fig)` after displaying to prevent memory leaks.

9. **Session state** — store processed results in `st.session_state` so switching tabs doesn't re-run analysis. Key: `st.session_state["match_results"]`

10. **Cleat color for black cleats** — black cleats have low saturation in HSV. Handle as special case: if saturation < 30 across foot region → classify as "black cleats" regardless of hue.

11. **Apple Silicon MPS** — InsightFace/onnxruntime does not yet support MPS. Use `CPUExecutionProvider` for InsightFace. Use MPS for PyTorch/YOLO models only.

12. **The identification confidence shown to the coach** should be the AVERAGE confidence across all frames where the player was identified — not just the best single match. This gives an honest picture of overall tracking quality.

---

## `setup_instructions.md`

```markdown
# Setup Instructions — U10 Soccer Analyzer

## Requirements
- MacBook (Apple Silicon M1/M2/M3 recommended)
- Python 3.10 or 3.11
- ~2GB free disk space (models + database)
- BallerCam Panoramic View video exports

## Installation

### 1. Download the project
Place the soccer_analyzer/ folder anywhere on your Mac.

### 2. Create virtual environment
cd soccer_analyzer
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
brew install cmake          # Required for InsightFace
pip install -r requirements.txt

### 4. Run the app
streamlit run app.py

## First Launch Checklist
1. App opens at http://localhost:8501
2. Go to Roster Manager → 16 players are pre-loaded
3. Upload a photo for each player (team photo works — crop individually)
4. Go to Field Setup → calibrate your home field (2 minutes, done once)
5. You're ready to analyze games

## Getting Video From BallerCam
- After game: open BallerCam app → go to game replay
- Tap "Panoramic" view (NOT Smart View)
- Tap download → select HD quality
- AirDrop to MacBook
- Drag MP4 into the app uploader

## Recommended Analysis Settings (U10 7v7)
- Field: 50m × 35m
- Sample rate: 3 (process every 3rd frame)
- Confidence: 0.35
- Model: YOLOv8 Nano (fast) or Small (more accurate)

## Processing Time
- 40-minute game on M2 Mac: approximately 8-12 minutes
- Results are cached — switching tabs is instant after first analysis

## Season Workflow
1. Game day: set up BallerCam at mid-field, hit record
2. After game: AirDrop video to MacBook
3. Open app → Match Analysis → upload video → Analyze
4. Review tabs, confirm any flagged player IDs (usually 1-3 per game)
5. Save to Season History
6. Print PDF report for next training session
```
