#!/bin/bash
echo "Segmenting sentences..."
python -m scripts.segment-sentences

echo "Segmented sentences have been saved so you can exit safely here if you do not want to run the next step..."

echo "Extracting features and exporting..."
python -m scripts.process

echo "Features have been saved."