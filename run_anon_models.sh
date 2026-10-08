#!/bin/bash
# Full anonymous round (720 texts x with/without criteria x 3 reps) for the comparison models on inferno,
# one model at a time. Run with caffeinate so the Mac doesn't sleep: caffeinate -i ./run_anon_models.sh
cd "$(dirname "$0")"
HOST=http://localhost:11434
KEY=~/.ollama_gateway_key

run() {  # model think
  local tag
  tag=$(echo "$1" | tr ':/' '--')
  echo "=== $1 (think=$2) $(date '+%H:%M')"
  python3 run_ollama.py --model "$1" --think "$2" --workers 9 \
    --hosts "$HOST" --api-key-file "$KEY" --out "results/anon_inferno_${tag}_think-$2.jsonl"
}

run qwen3:30b-a3b off
echo "$(date '+%H:%M') all done"
