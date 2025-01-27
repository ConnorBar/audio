#!/bin/bash
echo "Filtering out sentences..."
python -m scripts.single-word-filter

echo "Converting from mp3 to wav..."
python -m scripts.mp3towav

echo "Extracting features and exporting..."
python -m scripts.feature-extraction

echo "Done! You can run the training file now!"