"""
Vehicle Speed Estimation + Counting from Fixed CCTV Video  (version 6)
------------------------------------------------------------------------
Folder-structure version. All paths are resolved relative to the project
root (one level above this file), so it doesn't matter where you run the
script from - it always finds data/, models/ and outputs/ correctly.

Project layout expected:

vehicle_speed_estimation/
├── data/videos/        <- put your input CCTV video(s) here
├── models/              <- put your trained best.pt here
├── outputs/videos/      <- annotated output videos are written here
├── outputs/csv/         <- per-vehicle speed logs (CSV) are written here
├── src/speed_estimation.py   <- this file
├── trackers/             <- tracker config (auto-created)
└── notebooks/            <- optional, for experiments/testing in .ipynb

NOTE: practice / estimation only. The pixel -> meter conversion is approximate.
"""

import csv
import datetime
import statistics
from pathlib import Path

import cv2
from ultralytics import YOLO
from collections import defaultdict

# ---------------- CONFIG ----------------
VIDEO_FILENAME = "Cars Moving On Road Stock Footage - Free Download.mp4"      # must exist inside data/videos/
MODEL_FILENAME = "yolov8.pt"             # must exist inside models/
RUN_NAME = "run1"                      # change this per run so outputs don't overwrite each other

LINE_A_RATIO = 0.38   # yellow line (near line, for speed only)
LINE_B_RATIO = 0.80   # green line (counting + far end of speed measurement)

REAL_DISTANCE_METERS = 20.0     # real road distance between the two lines (ESTIMATE THIS)
EXPECTED_MEDIAN_KMH = 100.0     # typical traffic speed here, used only for the calibration hint

IMGSZ = 1280
CONF = 0.20
DISPLAY_SECONDS = 3.0   # how long a speed / crossing stays in the on-screen logs
# -----------------------------------------

# ---------------- PROJECT PATHS (do not need editing) ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / VIDEO_FILENAME
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "outputs" / "videos" / f"{RUN_NAME}_annotated.mp4"
CSV_PATH = PROJECT_ROOT / "outputs" / "csv" / f"{RUN_NAME}_speeds.csv"
TRACKER_CFG = PROJECT_ROOT / "trackers" / "custom_bytetrack.yaml"

OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
TRACKER_CFG.parent.mkdir(parents=True, exist_ok=True)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}\nPut your video in data/videos/")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nPut best.pt in models/")

if not TRACKER_CFG.exists():
    TRACKER_CFG.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.20\n"
        "track_low_thresh: 0.05\n"
        "new_track_thresh: 0.20\n"
        "track_buffer: 60\n"
        "match_thresh: 0.85\n"
        "fuse_score: True\n"
    )
# ----------------------------------------------------------------------

model = YOLO(str(MODEL_PATH))
class_names = model.names

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

LINE_A_Y = int(height * LINE_A_RATIO)
LINE_B_Y = int(height * LINE_B_RATIO)
print(f"Video: {width}x{height} @ {fps:.2f} fps | line A y={LINE_A_Y}, line B y={LINE_B_Y}")

writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

track_data = defaultdict(lambda: {
    "prev_y": None, "prev_f": None,
    "A": None, "B": None,
    "speed": None, "cls": None,
    "n": 0, "counted": False,
})


def crossing_frame(prev_y, prev_f, cur_y, cur_f, line_y):
    if prev_y is None or prev_y == cur_y:
        return None
    if (prev_y - line_y) * (cur_y - line_y) <= 0:
        ratio = (line_y - prev_y) / (cur_y - prev_y)
        return prev_f + ratio * (cur_f - prev_f)
    return None


def draw_panel(frame, lines, x=15, y=15, line_h=26, pad=10, alpha=0.55):
    """Draw a semi-transparent box behind a list of text lines, starting at (x, y)."""
    if not lines:
        return y
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t in lines) + pad * 2
    h = line_h * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + pad, y + pad + (i + 1) * line_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return y + h  # bottom edge, so the next panel can be drawn below it


csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["vehicle_id", "class_name", "speed_kmh", "video_time_s", "logged_at"])

results_gen = model.track(source=str(VIDEO_PATH), tracker=str(TRACKER_CFG), persist=True,
                          stream=True, conf=CONF, imgsz=IMGSZ)

total_count = 0
class_counts = defaultdict(int)
display_id_map = {}   # track_id -> simple sequential ID (1, 2, 3, ...), assigned on crossing
next_display_id = 1
crossing_log = []   # (frame_idx, "text") for the recently-crossed list
speed_log = []       # (frame_idx, "text") for the recently-measured-speed list

for frame_idx, result in enumerate(results_gen):
    frame = result.orig_img.copy()

    cv2.line(frame, (0, LINE_A_Y), (width, LINE_A_Y), (0, 255, 255), 2)
    cv2.line(frame, (0, LINE_B_Y), (width, LINE_B_Y), (0, 255, 0), 2)

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        ids = result.boxes.id.int().cpu().tolist()
        clss = result.boxes.cls.int().cpu().tolist()

        for (x1, y1, x2, y2), track_id, cls_id in zip(boxes, ids, clss):
            d = track_data[track_id]
            d["cls"] = cls_id
            d["n"] += 1

            cx, cy = (x1 + x2) / 2, y2

            if d["A"] is None:
                d["A"] = crossing_frame(d["prev_y"], d["prev_f"], cy, frame_idx, LINE_A_Y)

            b_cross = None
            if d["B"] is None:
                b_cross = crossing_frame(d["prev_y"], d["prev_f"], cy, frame_idx, LINE_B_Y)
                if b_cross is not None:
                    d["B"] = b_cross

            # --- COUNTING: triggers once, the moment the green line is crossed ---
            if b_cross is not None and not d["counted"]:
                d["counted"] = True
                total_count += 1
                class_counts[cls_id] += 1
                if track_id not in display_id_map:
                    display_id_map[track_id] = next_display_id
                    next_display_id += 1
                crossing_log.append((frame_idx, f"#{display_id_map[track_id]} {class_names[cls_id]} crossed"))

            # --- SPEED: only if both lines were crossed ---
            if d["A"] is not None and d["B"] is not None and d["speed"] is None:
                frame_gap = abs(d["B"] - d["A"])
                if frame_gap > 0:
                    time_s = frame_gap / fps
                    d["speed"] = (REAL_DISTANCE_METERS / time_s) * 3.6
                    disp_id = display_id_map.get(track_id, "?")
                    speed_log.append((frame_idx, f"#{disp_id} {class_names[cls_id]}: {d['speed']:.0f} km/h"))
                    print(f"ID {disp_id!s:>3} {class_names[cls_id]:<10} speed={d['speed']:6.1f} km/h")

                    video_time_s = frame_idx / fps
                    csv_writer.writerow([
                        disp_id, class_names[cls_id], round(d["speed"], 1),
                        round(video_time_s, 2), datetime.datetime.now().isoformat(timespec="seconds"),
                    ])
                    csv_file.flush()   # write to disk immediately, in case the run is interrupted

            d["prev_y"], d["prev_f"] = cy, frame_idx

            name = class_names[cls_id]
            label = f"{name} {d['speed']:.0f} km/h" if d["speed"] is not None else f"{name} --"
            color = (0, 200, 0) if d["speed"] is not None else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(frame, label, (int(x1), max(int(y1) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---- top-left COUNT panel ----
    count_lines = [f"TOTAL: {total_count}"]
    for cid in sorted(class_counts):
        count_lines.append(f"  {class_names[cid]}: {class_counts[cid]}")
    bottom_y = draw_panel(frame, count_lines, x=15, y=15)

    max_age = DISPLAY_SECONDS * fps
    crossing_log = [(f, t) for f, t in crossing_log if frame_idx - f <= max_age]
    if crossing_log:
        draw_panel(frame, [t for _, t in crossing_log[-5:]], x=15, y=bottom_y + 10)

    # ---- speed log, top-right ----
    speed_log = [(f, t) for f, t in speed_log if frame_idx - f <= max_age]
    if speed_log:
        lines = speed_log[-5:]
        w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for _, t in lines) + 20
        draw_panel(frame, [t for _, t in lines], x=width - w - 15, y=15)

    writer.write(frame)

writer.release()
csv_file.close()
print("\nDone. Saved annotated video to:", OUTPUT_VIDEO_PATH)
print("Saved per-vehicle speed log to:", CSV_PATH)

print("\n--- Counting Summary ---")
print(f"Total vehicles counted: {total_count}")
for cid, cnt in sorted(class_counts.items()):
    print(f"  {class_names[cid]}: {cnt}")

# ---------------- CALIBRATION HELPER ----------------
speeds = [d["speed"] for d in track_data.values() if d["speed"] is not None]
if speeds:
    med = statistics.median(speeds)
    suggested = REAL_DISTANCE_METERS * EXPECTED_MEDIAN_KMH / med
    print("\n--- Calibration helper ---")
    print(f"Median measured speed: {med:.0f} km/h (with distance = {REAL_DISTANCE_METERS} m)")
    print(f"To make the median ~{EXPECTED_MEDIAN_KMH:.0f} km/h set REAL_DISTANCE_METERS = {suggested:.1f}")
