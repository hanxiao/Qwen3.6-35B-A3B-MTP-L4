#!/bin/bash
# Honest decode benchmark for the running llama-server (assumes the optimized
# container/compose is up on :8080). Reproduces the numbers in the README.
#
#   - decode-only metric: timings.predicted_per_second
#   - NO input/output cache: each request sets "cache_prompt": false with a distinct prompt
#   - greedy (temperature 0) for stable, reproducible MTP acceptance
#
# Usage: ./bench.sh [host:port]   (default 127.0.0.1:8080)
set -u
HP="${1:-127.0.0.1:8080}"
URL="http://$HP/v1/chat/completions"

declare -A PROMPTS=(
  [prose]="Describe the architecture of a modern transformer language model in depth, covering attention, MoE, and positional encodings."
  [code]="Write a complete, well-documented Python LRU cache class using OrderedDict, with get and put, type hints and docstrings, then five assert-based unit tests."
  [json]="Generate a JSON array of 30 fictional employee records, each with id, first_name, last_name, email, department, salary, hire_date. Output only valid JSON."
  [repeat]="Write the integers from 1 to 120, one per line, formatted exactly as 'N: word' (e.g. 1: one)."
)

curl -s "http://$HP/health" | grep -q '"status":"ok"' || { echo "server not ready at $HP"; exit 1; }
# warmup
curl -s "$URL" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"warm up"}],"max_tokens":32,"temperature":0,"cache_prompt":false}' >/dev/null

for kind in prose code json repeat; do
  sum=0; n=0; as=0; an=0
  for r in 1 2 3; do
    R=$(curl -s "$URL" -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPTS[$kind]}\"}],\"max_tokens\":320,\"temperature\":0,\"cache_prompt\":false}")
    O=$(echo "$R" | python3 -c "import sys,json
t=json.load(sys.stdin)['timings']; dn=t.get('draft_n',0)
print(f\"{t['predicted_per_second']:.2f} {(t.get('draft_n_accepted',0)/dn) if dn else 0:.3f}\")")
    v=$(echo "$O"|awk '{print $1}'); a=$(echo "$O"|awk '{print $2}')
    sum=$(echo "$sum+$v"|bc -l); n=$((n+1)); as=$(echo "$as+$a"|bc -l); an=$((an+1))
  done
  printf "%-8s decode=%.2f tok/s   mtp_accept=%.3f\n" "$kind" "$(echo "scale=2;$sum/$n"|bc -l)" "$(echo "scale=3;$as/$an"|bc -l)"
done
