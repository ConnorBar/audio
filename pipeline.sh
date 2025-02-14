#!/bin/bash
echo "Segmenting sentences..."
python -m scripts.segment-sentences

echo "Extracting features and exporting..."
python -m scripts.feature-extraction

echo "Done! You can start training now!"