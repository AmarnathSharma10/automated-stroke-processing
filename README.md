Badminton Dataset and Tools
===========================

A focused repository containing data, scripts, and experiments for badminton trajectory, stroke, and movement forecasting together with supporting tools and example projects (CoachAI, Movement Forecasting, Stroke Forecasting, etc.). This repo collects preprocessing, training, and evaluation utilities used in several badminton-related research and demo projects.

Key points
- Purpose: provide datasets, preprocessing, models, and example training/evaluation code for badminton trajectory and stroke forecasting tasks.
- Language: Python 3
- Minimal requirements are listed in `requirements.txt`.

Repository layout (top-level)
- `main.py` — top-level entry / example orchestration script.
- `requirements.txt` — Python dependencies.
- `yolov8n-pose.pt` — example pose model checkpoint (tracked in .gitignore exceptions).
- `CoachAI-Projects/` — CoachAI example code and environments.
- `Movement Forecasting/`, `Stroke Forecasting/`, `ShuttleSet/`, `RallyNet/`, `Shot Influence/` — research subprojects with datasets, models and training code.

main.py (concise)
- Purpose: extract 30-frame stroke clips from match videos, run YOLOv8 pose estimation on each frame, convert ankle pixel coordinates to real-world court coordinates (using homography), and save per-stroke pose, position, and label data.
- Inputs:
  - `match_index` (int) — index into `CoachAI-Projects/ShuttleSet/set/match.csv` (script's default run uses indices 41–50).
  - Required files: `CoachAI-Projects/ShuttleSet/set/match.csv`, `homography.csv`, per-match `set1.csv`/`set2.csv`/`set3.csv` under the match folder, and `yolov8n-pose.pt` in repo root.
  - External tools: `yt-dlp` (for downloading videos); typical Python deps (pandas, numpy, opencv-python, torch, ultralytics).
- Outputs:
  - `pkl-files-match-{match_index}.zip` — ZIP containing `poses_{match_index}.pkl` (list of (30,2,17,3) arrays), `positions_{match_index}.pkl` (list of (2,30,2) arrays), and `labels_{match_index}.pkl`.
  - Temporary artifacts (downloaded mp4, extracted frames, cloned `CoachAI-Projects`) are removed by the script; the zip is preserved.
- Key behavior notes:
  - Each stroke uses 30 frames (default window: 15 frames before hit, 14 after).
  - Pose handling: frames with fewer than 2 detections cause the stroke to be skipped; when ≥3 detections are present they are merged into exactly 2 players by horizontal centroid and joint-confidence.
  - Homography from `homography.csv` is used to map pixel ankle positions to real-world coordinates.
  - By default the script will clone `CoachAI-Projects` if missing and will delete the clone and generated PKLs after zipping (adjust `cleanup_generated_files` if you want to keep them).
- Quick run:

```bash
# default behaviour (as currently implemented in this repo)
python main.py
# or import and run in Python:
# from main import main; main(match_index=21, visualize=False)
```

Quickstart
1. Create and activate a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the main script (example):

```bash
python main.py
```

Notes
- Some subprojects include their own README files and environment specifications (e.g., `environment.yml`) — check each subfolder for details.
- Large model files (e.g., `*.pt`) and temporary artifacts are ignored via `.gitignore`.
- If you plan to reproduce experiments, ensure you have the required datasets and model checkpoints referenced by the project you are using.

Contributing
- Please open issues or pull requests. Keep changes small and include tests where appropriate.

Contact
- For questions about setup or experiments, open an issue in this repository.
