#!/usr/bin/env bash
# Download model weights from Google Drive into ./weights/
# Usage: bash scripts/download_weights.sh
set -euo pipefail

WEIGHTS_FOLDER_URL="https://drive.google.com/drive/folders/1joO7w1Am7B418SIqGBq90YipQl81FMzh"

echo "Creating weight directories..."
mkdir -p weights/players_detection \
         weights/ball_detection \
         weights/players_keypoints_detection \
         weights/court_keypoints_detection

echo "Installing gdown..."
pip install --quiet gdown

echo "Downloading weights from Google Drive..."
gdown --folder "${WEIGHTS_FOLDER_URL}" -O weights/ --remaining-ok

echo "Weights downloaded successfully."
ls -lh weights/players_detection/ \
        weights/ball_detection/ \
        weights/players_keypoints_detection/ \
        weights/court_keypoints_detection/ 2>/dev/null || true
