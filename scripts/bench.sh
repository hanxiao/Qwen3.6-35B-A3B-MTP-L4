#!/bin/bash
# Honest decode benchmark across 7 diverse workloads — reproduces the README's headline numbers.
#   - decode-only metric: timings.predicted_per_second (excludes prompt processing)
#   - NO input/output cache: each request sets "cache_prompt": false with a distinct prompt
#   - greedy (temperature 0) for stable, reproducible MTP acceptance
# Usage: ./bench.sh [host:port]   (default 127.0.0.1:8080) — point it at a running llama-server.
set -u
HP="${1:-127.0.0.1:8080}"; URL="http://$HP/v1/chat/completions"
curl -s "http://$HP/health" 2>/dev/null | grep -q ok || { echo "server not up at $HP"; exit 1; }
declare -A P=(
 [prose]="Write three detailed paragraphs explaining how photosynthesis works in plants, from light absorption to glucose production."
 [code]="Write a complete Python implementation of a thread-safe LRU cache with get/put, type hints, docstrings, and 5 unit tests."
 [json]="Output a JSON array of 25 fictional books, each with title, author, year, genre, isbn, pages. Valid JSON only."
 [chat]="I'm planning a 5-day trip to Tokyo in spring. Suggest a day-by-day itinerary with food recommendations and travel tips."
 [math]="Solve step by step: a train leaves city A at 60 mph, another leaves city B (300 miles away) at 40 mph toward A. When and where do they meet? Show all reasoning."
 [multi]="Translate the following into French, German, Spanish, and Japanese, then explain the key grammatical differences: 'The early bird catches the worm, but the second mouse gets the cheese.'"
 [summ]="Summarize the theory of general relativity, its key predictions, and experimental confirmations, in about 200 words for a general audience."
)
curl -s "$URL" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0,"cache_prompt":false}' >/dev/null
min=99999; minc=""
for k in prose code json chat math multi summ; do
  s=0; n=0; ac=0; an=0
  for r in 1 2; do
    R=$(curl -s "$URL" -H 'Content-Type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${P[$k]}\"}],\"max_tokens\":256,\"temperature\":0,\"cache_prompt\":false}")
    O=$(echo "$R" | python3 -c "import sys,json
try:
 t=json.load(sys.stdin)['timings']; dn=t.get('draft_n',0)
 print(f\"{t['predicted_per_second']:.2f} {(t.get('draft_n_accepted',0)/dn) if dn else 0:.3f}\")
except: print('ERR 0')" 2>/dev/null)
    v=$(echo "$O"|awk '{print $1}'); a=$(echo "$O"|awk '{print $2}')
    [[ "$v" =~ ^[0-9] ]] && { s=$(echo "$s+$v"|bc -l); n=$((n+1)); ac=$(echo "$ac+$a"|bc -l); an=$((an+1)); }
  done
  if [ $n -gt 0 ]; then avg=$(echo "scale=2;$s/$n"|bc -l); acc=$(echo "scale=3;$ac/$an"|bc -l)
    printf "  %-6s %7.2f tok/s  accept=%s\n" "$k" "$avg" "$acc"
    (( $(echo "$avg < $min"|bc -l) )) && { min=$avg; minc=$k; }
  fi
done
echo "  ----"
echo "  MIN = $min tok/s ($minc)"
