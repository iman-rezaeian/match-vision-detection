# Setup Instructions — U10 Soccer Analyzer

## Requirements
- MacBook (Apple Silicon M1/M2/M3 recommended)
- Python 3.10 or 3.11
- ~2GB free disk space (models + database)
- BallerCam Panoramic View video exports

## Installation

### 1. Download the project
Place the `soccer_analyzer/` folder anywhere on your Mac.

### 2. Create virtual environment
```bash
cd soccer_analyzer
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
brew install cmake          # Required for InsightFace
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

## First Launch Checklist
1. App opens at http://localhost:8501
2. Go to **Roster Manager** → 16 players are pre-loaded
3. Upload a photo for each player (team photo works — crop individually)
4. Go to **Field Setup** → calibrate your home field (2 minutes, done once)
5. You're ready to analyze games!

## Getting Video From BallerCam
- After game: open BallerCam app → go to game replay
- Tap **"Panoramic"** view (NOT Smart View)
- Tap download → select HD quality
- AirDrop to MacBook
- Drag MP4 into the app uploader

## Recommended Analysis Settings (U10 7v7)
- **Field:** 50m × 35m
- **Sample rate:** 3 (process every 3rd frame)
- **Confidence:** 0.35
- **Model:** YOLOv8 Nano (fast) or Small (more accurate)

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

## Troubleshooting

### "No players detected"
- Lower the confidence threshold (try 0.25)
- Check video brightness — very dark or overexposed video reduces detection accuracy
- Ensure video is Panoramic mode (not Smart View)

### "Face embedding extraction failed"
- Use a clear front-facing photo of the player
- Ensure the face is well-lit and not obscured
- Photo should be at least 200×200 pixels

### InsightFace model download
- First run downloads ~300MB model to `~/.insightface/`
- Requires internet connection for first run only
- Subsequent runs are fully offline

### Slow processing
- Use YOLOv8 Nano (default) for fastest processing
- Increase sample rate to 5 (trades accuracy for speed)
- Close other resource-heavy applications during analysis

## Architecture Notes

### Device Usage
- **YOLO detection:** Uses CPU (MPS support varies)
- **InsightFace:** CPU only (onnxruntime doesn't support MPS yet)
- **MediaPipe:** CPU only
- **Data processing:** NumPy/Pandas on CPU

### Data Storage
- All data stored locally in `data/roster.db` (SQLite)
- No cloud services, no API keys, no subscriptions
- Videos are processed but not stored — only tracking data is kept
