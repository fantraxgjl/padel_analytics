#!/usr/bin/env bash
# Download model weights from Google Drive into $WEIGHTS_DIR (default: ./weights).
# Skips the download entirely if all required files are already present.
# Retries up to 3 times on network failure.
#
# Environment variables:
#   WEIGHTS_DIR              - where weights live (default: ./weights)
#                              Set to your RunPod volume path, e.g. /runpod-volume/weights
#   SKIP_WEIGHTS_DOWNLOAD=1  - skip download entirely (trust mounted weights)
#
# Manual fallback: download https://drive.google.com/drive/folders/1joO7w1Am7B418SIqGBq90YipQl81FMzh
# and mount the extracted folder: docker run -v /path/to/weights:/app/weights ...
set -uo pipefail

FOLDER_ID="1joO7w1Am7B418SIqGBq90YipQl81FMzh"
WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"

REQUIRED_FILES=(
    "${WEIGHTS_DIR}/players_detection/yolov8m.pt"
    "${WEIGHTS_DIR}/ball_detection/TrackNet_best.pt"
    "${WEIGHTS_DIR}/ball_detection/InpaintNet_best.pt"
    "${WEIGHTS_DIR}/players_keypoints_detection/best.pt"
    "${WEIGHTS_DIR}/court_keypoints_detection/best.pt"
)

mkdir -p "${WEIGHTS_DIR}/players_detection" \
         "${WEIGHTS_DIR}/ball_detection" \
         "${WEIGHTS_DIR}/players_keypoints_detection" \
         "${WEIGHTS_DIR}/court_keypoints_detection"

# ---------------------------------------------------------------------------
# Rescue any .pt files that gdown placed in root-level subfolders on a
# previous (partial) run.  This makes the script idempotent: re-running it
# after a network failure won't re-download files that already exist on disk
# in the wrong location.
# ---------------------------------------------------------------------------
collect_stray_weights() {
    while IFS= read -r -d '' pt_file; do
        subdir=$(basename "$(dirname "$pt_file")")
        filename=$(basename "$pt_file")
        target="${WEIGHTS_DIR}/${subdir}/${filename}"
        mkdir -p "${WEIGHTS_DIR}/${subdir}"
        echo "  Collecting $pt_file -> $target"
        mv "$pt_file" "$target"
        rmdir "$(dirname "$pt_file")" 2>/dev/null || true
    done < <(find . -maxdepth 3 -name "*.pt" \
                    -not -path "./${WEIGHTS_DIR#./}/*" \
                    -not -path "*/vim/*" \
                    -print0 2>/dev/null)
}

collect_stray_weights

# ---------------------------------------------------------------------------

# Allow completely bypassing the download (e.g. weights mounted via a volume)
if [ "${SKIP_WEIGHTS_DOWNLOAD:-0}" = "1" ]; then
    echo "SKIP_WEIGHTS_DOWNLOAD=1 — skipping download, trusting mounted weights."
    find "${WEIGHTS_DIR}" -name "*.pt" -exec ls -lh {} \; 2>/dev/null || true
    exit 0
fi

# Skip download if all weights already present (e.g. mounted volume or container restart)
all_present=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        all_present=false
        break
    fi
done

if $all_present; then
    echo "All weights already present — skipping download."
    find "${WEIGHTS_DIR}" -name "*.pt" -exec ls -lh {} \;
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
             -O . --remaining-ok; then
        success=true
        break
    fi
    echo "Attempt $attempt failed. Waiting ${delay}s before retry..."
    sleep "$delay"
    delay=$((delay * 2))
done

if ! $success; then
    echo "ERROR: Failed to download weights after $max_attempts attempts." >&2
    echo "Manual fix: download the folder in your browser and mount it:" >&2
    echo "  https://drive.google.com/drive/folders/${FOLDER_ID}" >&2
    echo "  docker run -v /path/to/weights:/app/weights ..." >&2
    exit 1
fi

# Move any files gdown placed outside WEIGHTS_DIR into the correct subdirectory.
echo "Organising downloaded files into ${WEIGHTS_DIR}/ ..."
collect_stray_weights

# Verify all required files are actually present and non-empty
all_present=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        echo "ERROR: Expected weight file missing after download: $f" >&2
        all_present=false
    fi
done

if ! $all_present; then
    echo "" >&2
    echo "Files found on disk after download:" >&2
    find . -name "*.pt" -not -path "*/vim/*" -exec ls -lh {} \; 2>/dev/null || echo "  (none)" >&2
    echo "" >&2
    echo "Manual fix: download the folder in your browser and mount it:" >&2
    echo "  https://drive.google.com/drive/folders/${FOLDER_ID}" >&2
    echo "  docker run -v /path/to/weights:/app/weights ..." >&2
    exit 1
fi

echo "Weights downloaded successfully."
find "${WEIGHTS_DIR}" -name "*.pt" -exec ls -lh {} \;
