#!/bin/bash
# Main experiment for one model: independent vs. batch vs. sequential grading, 12 assignments x 60 texts,
# with and without criteria. Every block is resumable: rerunning this script continues where it stopped.
#   caffeinate -i ./run_main.sh gemma4:26b        (keep the laptop lid open)
# Progress: python3 status.py
cd "$(dirname "$0")"
MODEL=${1:-gemma4:26b}
THINK=${2:-off}
ALL="G6-1 G6-2 G6-3 G7-1 G7-2 G7-3 G8-1 G8-2 G8-3 G9-1 G9-2 G9-3"
B="python3 run_batch.py --tag main --model $MODEL --think $THINK --assignments $ALL"
S="python3 run_sequential.py --tag main --model $MODEL --think $THINK --assignments $ALL"

echo "=== [$(date '+%a %H:%M')] A1 independent, batch format: each text alone, 3 reps"
$B --designs single_code --reps 3 --num-ctx 8192
echo "=== [$(date '+%a %H:%M')] B  batch, random classes of 5/10/20/30, 3 splits"
$B --designs random5 random10 random20 random30 --splits 3 --num-ctx 16384
echo "=== [$(date '+%a %H:%M')] Db batch, class profiles weak/middle/strong, 6 classes each"
$B --designs profiles --splits 6 --num-ctx 16384
echo "=== [$(date '+%a %H:%M')] A2 independent, Gem format: one-turn conversations, 3 reps"
$S --designs single_code --reps 3 --num-ctx 16384
echo "=== [$(date "+%a %H:%M")] C  sequential, random classes of 5/10/20/30, 3 splits"
$S --designs random5 random10 random20 random30 --splits 3 --num-ctx 16384
echo "=== [$(date '+%a %H:%M')] Ds sequential, class profiles weak/middle/strong, 6 classes each"
$S --designs profiles --splits 6 --num-ctx 16384
echo "=== [$(date '+%a %H:%M')] all blocks finished for $MODEL"
