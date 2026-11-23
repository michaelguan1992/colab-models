#!/usr/bin/env bash
# Runs on the Colab runtime. Idempotent: safe to re-run on an existing session.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /content/lm/logs /content/models /content/outputs

if [ ! -f /content/lm/.deps_ok ]; then
  python3 -m pip install -q "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27" pyyaml psutil \
    "huggingface_hub[hf_transfer]>=0.30"
  touch /content/lm/.deps_ok
fi

if ! command -v strata >/dev/null 2>&1; then
  curl -fsSL --connect-timeout 15 --max-time 120 \
    https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o /tmp/strata-install.sh
  sh /tmp/strata-install.sh
  # installer may put the binary in ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
fi

# (Re)start the controller detached from this ssh session.
pkill -f "python3 -m controller" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
setsid nohup python3 -m controller --config models.yaml --port 9000 ${LOAD:+--load "$LOAD"} \
  > /content/lm/logs/controller.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:9000/health >/dev/null && { echo "controller up"; exit 0; }
  sleep 1
done
echo "controller failed to start:" >&2
tail -n 50 /content/lm/logs/controller.log >&2
exit 1
