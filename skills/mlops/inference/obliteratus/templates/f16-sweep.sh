#!/usr/bin/env bash
#
# f16-sweep.sh — Run multiple orthogonalization configs on F16, evaluate each.
#
# For each config: orthogonalize F16 copy -> start llama-server -> compare.py (refusal + capability) -> kill server
# Best config gets quantized to Q4 for the final matched comparison.
#
# Prerequisites:
#   - orthogonalize_v3.py in the same directory
#   - compare.py (eval harness) in the same directory
#   - llama.cpp built (llama-server, llama-quantize)
#   - clean F16 model + refusal_mean.gguf (cvector-generator output)
#   - clean F16 baseline already benchmarked (capability + refusal)

set -euo pipefail

ABLATE_DIR="${ABLATE_DIR:-$HOME/qwable-eval/abliterate}"
LLAMA="${LLAMA:-$HOME/llama.cpp/build/bin}"
PORT="${PORT:-8099}"
RESULTS_DIR="$ABLATE_DIR/results_v2"
mkdir -p "$RESULTS_DIR"

F16_SRC="$ABLATE_DIR/qwable_f16.gguf"

# All 16 attention layers for Qwen3.5 hybrid (every 4th, 0-indexed)
ALL_ATTN="3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63"
# Mid-band 12 (skip first 2 and last 2 attention layers)
MID_ATTN="11,15,19,23,27,31,35,39,43,47,51,55"

start_server() {
  local model="$1" name="$2"
  "$LLAMA/llama-server" -m "$model" -ngl 99 -c 8192 --port $PORT \
    --jinja --no-context-shift > "$RESULTS_DIR/server_${name}.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then
      echo "  Server ready (${i}s) PID=$pid"
      echo "$pid"
      return 0
    fi
    sleep 1
  done
  echo "  FAILED: server timeout"
  kill $pid 2>/dev/null || true
  echo ""
  return 1
}

kill_server() {
  kill "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
  sleep 2
}

# ---- Configs: name|targets|layers|alpha ----
# NEVER include ssm_out.weight. ALWAYS use explicit layer lists.
CONFIGS=(
  "attn16_a1.0|attn_output.weight|${ALL_ATTN}|1.0"
  "attn16_a0.7|attn_output.weight|${ALL_ATTN}|0.7"
  "attn12_a1.0|attn_output.weight|${MID_ATTN}|1.0"
  "ffn_attn16_a1.0|ffn_down.weight|${ALL_ATTN}|1.0"
  "ffn_attn16_a0.7|ffn_down.weight|${ALL_ATTN}|0.7"
  "both_attn16_a1.0|attn_output.weight,ffn_down.weight|${ALL_ATTN}|1.0"
  "both_attn16_a0.7|attn_output.weight,ffn_down.weight|${ALL_ATTN}|0.7"
  "attn16_a0.5|attn_output.weight|${ALL_ATTN}|0.5"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name targets layers alpha <<< "$cfg"

  echo ""
  echo "============================================"
  echo "CONFIG: $name"
  echo "  targets=$targets  layers=$layers  alpha=$alpha"
  echo "============================================"

  OUT="$RESULTS_DIR/result_${name}.json"
  if [ -f "$OUT" ]; then
    echo "  Already done, skipping"
    continue
  fi

  F16_OUT="$RESULTS_DIR/qwable_${name}.gguf"
  echo "[1/3] Orthogonalizing..."
  python3 "$ABLATE_DIR/orthogonalize_v3.py" \
    --src "$F16_SRC" --targets "$targets" \
    --layers "$layers" --alpha "$alpha" \
    --out "$F16_OUT" 2>&1

  if [ ! -f "$F16_OUT" ]; then
    echo "FAILED"
    continue
  fi

  echo "[2/3] Starting server..."
  PID=$(start_server "$F16_OUT" "$name")
  if [ -z "$PID" ]; then
    rm -f "$F16_OUT"
    continue
  fi

  echo "[3/3] Evaluating..."
  python3 "$ABLATE_DIR/compare.py" \
    --url "http://127.0.0.1:$PORT" --tag "$name" \
    --mode both --out "$OUT" 2>&1

  kill_server "$PID"
  rm -f "$F16_OUT"
  echo "  Done: $name"
done

echo ""
echo "============================================"
echo "SWEEP COMPLETE — Results:"
echo "============================================"
for f in "$RESULTS_DIR"/result_*.json; do
  [ -f "$f" ] || continue
  python3 -c "
import json
d = json.load(open('$f'))
tag = d.get('tag', '?')
cap = d.get('capability', [])
ref = d.get('refusal', [])
parts = [tag]
if cap:
    acc = sum(r['correct'] for r in cap) / len(cap)
    parts.append(f'cap={acc*100:.1f}%')
if ref:
    rr = sum(r['refused'] for r in ref) / len(ref)
    parts.append(f'refuse={rr*100:.1f}%')
print('  ' + '  '.join(parts))
" 2>/dev/null || echo "  $(basename $f): error"
done
