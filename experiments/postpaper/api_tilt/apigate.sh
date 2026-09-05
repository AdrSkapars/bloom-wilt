#!/bin/bash
# Recovery gate. Exit 0 only if the API answers 200 on N consecutive spaced probes.
#
# Why: at 05:53 a SINGLE probe returned 200 and both relaunched runs hit HTTP 412 within a
# minute. Eight probes immediately afterwards were 412/8. One success is not recovery --
# it can be a routing or cache artifact. Gate every relaunch on this.
#
#   bash experiments/postpaper/api_tilt/apigate.sh [n_probes] && <launch>
cd "$(dirname "$0")/../../.."
set -a; . ./.env.local; set +a
N="${1:-5}"
python -X utf8 - "$N" <<'PY'
import os, sys, time, requests
n = int(sys.argv[1])
key = os.environ["FIREWORKS_API_KEY"]
codes = []
for i in range(n):
    try:
        r = requests.post("https://api.fireworks.ai/inference/v1/completions",
                          json={"model": "accounts/fireworks/models/gpt-oss-120b",
                                "prompt": "hi", "max_tokens": 1},
                          headers={"Authorization": f"Bearer {key}"}, timeout=20)
        codes.append(r.status_code)
    except Exception as e:
        codes.append(type(e).__name__)
    if i < n - 1:
        time.sleep(3)
ok = sum(1 for c in codes if c == 200)
print(f"probes {codes}  -> {ok}/{n} healthy")
sys.exit(0 if ok == n else 1)
PY
