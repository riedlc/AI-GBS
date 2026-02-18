#!/usr/bin/env bash
# Run this on the server (in AI-GBS repo) to find why the Llama experiment stopped.
# Usage: ./scripts/check_last_error.sh   or   bash scripts/check_last_error.sh

set -e
RESULTS_DIR="${1:-results}"

echo "=== Checking for saved errors ==="
echo ""

# 1. Global fatal error (crash before or outside batch loop)
if [ -f "results/last_fatal_error.txt" ]; then
  echo "--- results/last_fatal_error.txt ---"
  cat results/last_fatal_error.txt
  echo ""
fi

# 2. Per-experiment last_error.txt (error during a batch)
for f in "$RESULTS_DIR"/*/last_error.txt; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    cat "$f"
    echo ""
  fi
done

# 3. Interruption state (e.g. Ctrl+C or crash – which batch we were at)
echo "=== Interruption state (which batch stopped) ==="
for f in "$RESULTS_DIR"/*/interruption_state.json; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    cat "$f"
    echo ""
  fi
done

# 4. Last batch in progress.json often has "error" in failed results
echo "=== Last batch results (look for 'error' in failed entries) ==="
for progress in "$RESULTS_DIR"/*/progress.json; do
  if [ -f "$progress" ]; then
    echo "--- $progress ---"
    python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))
batches = p.get('batches_completed', [])
if not batches: sys.exit(0)
last = batches[-1]
print('Batch', last.get('batch_num'), 'of', last.get('total_configs'), 'configs')
for r in last.get('results', []):
  if r.get('status') != 'success' and r.get('error'):
    err = r.get('error', '')[:300]
    print('  ', r.get('config'), '->', err)
" "$progress" 2>/dev/null || true
    echo ""
  fi
done

echo "Done."
