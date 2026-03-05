#!/bin/bash
# Creates symbolic links so the list files can find features.
# Run ONCE after cloning and downloading data.
# Usage: bash setup_data.sh /path/to/your/data/root
set -e
if [ -z "$1" ]; then
    echo "Usage: bash setup_data.sh /path/to/your/data/root"
    exit 1
fi
DATA_ROOT="$1"
PARENT_DIR="$(cd "$(dirname "$0")/.." \&\& pwd)"
echo "Linking data from: $DATA_ROOT"
echo "Into parent dir:   $PARENT_DIR"
for folder in XD-Violence-V0.2 ownExtractedV0.0.2 video_features I3D-Corrupted-Features-Fixed-3 I3D-Corrupted-Features-Fixed-5 I3D-VAL VGGISH-VAL; do
    SRC="$DATA_ROOT/$folder"
    DST="$PARENT_DIR/$folder"
    if [ -d "$SRC" ]; then
        ln -sfn "$SRC" "$DST"
        echo "  OK:   $folder"
    else
        echo "  SKIP: $folder (not found)"
    fi
done
echo ""
echo "Done. You can now run evaluation scripts."
