#!/bin/bash
# Focused commercial-model run via OpenRouter (variant 3): single reference (2 reps), random classes of 20
# (3 splits), weak/middle/strong classes of 20 (6 each); all 12 assignments, with and without criteria.
# Usage: ./run_or_main.sh <openrouter-model> <max-usd>
cd "$(dirname "$0")"
M=$1; CAP=$2
R="python3 run_openrouter.py --tag ormain --model $M --max-usd $CAP"
$R --designs single_code --reps 2
$R --designs random20 --splits 3
$R --designs profiles --splits 6
echo "=== finished $M"
