#!/usr/bin/env bash
# Download model weights from Google Drive into ./weights/
# Skips the download entirely if all required files are already present.
# Retries up to 3 times on network failure.
set -uo pipefail

FOLDER_ID="1joO7w1Am7B418SIqGBq90YipQl81FMzh"

REQUIRED_FILES=(
    "weights/players_detection/yolov8m.pt"
    "weights/ball_detection/TrackNet_best.pt"
    "weights/ball_detection/InpaintNet_best.pt"
    "weights/players_keypoints_detection/best.pt"
    "weights/court_keypoints_detection/best.pt"
)

mkdir -p weights/players_detection \
         weights/ball_detection \
         weights/players_keypoints_detection \
         weights/court_keypoints_detection

# Skip download if all weights already present (e.g. container restart)
all_present=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        all_present=false
        break
    fi
done

if $all_present; then
    echo "All weights already present — skipping download."
    find weights/ -name "*.pt" -exec ls -lh {} \;
    exit 0
fi

pip install --quiet gdown

# Retry folder download up to 3 times with exponential backoff
max_attempts=3
delay=10
success=false

for attempt in $(seq 1 $max_attempts); do
    echo "Downloading weights from Google Drive (attempt $attempt/$max_attempts)..."
    if gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" \
             -O weights/ --remaining-ok; then
        success=true
        break
    fi
    echo "Attempt $attempt failed. Waiting ${delay}s before retry..."
    sleep "$delay"
    delay=$((delay * 2))
done

if ! $success; then
    echo "ERROR: Failed to download weights after $max_attempts attempts." >&2
    echo "Check network connectivity and Google Drive folder permissions." >&2
    exit 1
fi

echo "Weights downloaded successfully."
find weights/ -name "*.pt" -exec ls -lh {} \;
