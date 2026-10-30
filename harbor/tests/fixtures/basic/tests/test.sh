#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
if [[ "$(cat /workspace/answer.txt)" == "harbor-sandfleet-ok" ]]; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
