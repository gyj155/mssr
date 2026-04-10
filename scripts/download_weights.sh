#!/bin/bash
# Download model weights for MSSR
# Usage: bash scripts/download_weights.sh

set -e

# GroundingDINO weights
GDINO_DIR="src/models/GroundingDINO/weights"
GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

echo ""
echo "[1/3] Downloading GroundingDINO weights..."
mkdir -p "$GDINO_DIR"
if [ ! -f "$GDINO_DIR/groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress -O "$GDINO_DIR/groundingdino_swint_ogc.pth" "$GDINO_URL"
    echo "  GroundingDINO weights downloaded."
else
    echo "  GroundingDINO weights already exist, skipping."
fi

# SAM2.1 checkpoints
SAM2_DIR="src/models/sam2/checkpoints"
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

echo ""
echo "[2/3] Downloading SAM2.1 weights..."
mkdir -p "$SAM2_DIR"
if [ ! -f "$SAM2_DIR/sam2.1_hiera_large.pt" ]; then
    wget -q --show-progress -O "$SAM2_DIR/sam2.1_hiera_large.pt" "$SAM2_URL"
    echo "  SAM2.1 weights downloaded."
else
    echo "  SAM2.1 weights already exist, skipping."
fi

# VGGT checkpoints
VGGT_DIR="vggt/checkpoints"
VGGT_URL="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

echo ""
echo "[3/3] Downloading VGGT weights..."
mkdir -p "$VGGT_DIR"
if [ ! -f "$VGGT_DIR/model.pt" ]; then
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$VGGT_DIR/model.pt" "$VGGT_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$VGGT_DIR/model.pt" "$VGGT_URL"
    else
        echo "  ERROR: wget or curl not found. Please download manually from:"
        echo "  $VGGT_URL -> $VGGT_DIR/model.pt"
    fi
    echo "  VGGT weights downloaded."
else
    echo "  VGGT weights already exist, skipping."
fi

echo "Weight locations:"
echo "  GroundingDINO: $GDINO_DIR/groundingdino_swint_ogc.pth"
echo "  SAM2.1:        $SAM2_DIR/sam2.1_hiera_large.pt"
echo "  VGGT:          $VGGT_DIR/model.pt"
