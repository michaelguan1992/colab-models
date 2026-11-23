#!/usr/bin/env bash
# Runs on the Colab runtime. Idempotent: safe to re-run on an existing session.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /content/lm/logs /content/models /content/outputs

if [ ! -f /content/lm/.deps_ok ]; then
  python3 -m pip install -q "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27" pyyaml psutil \
    "huggingface_hub>=1.0"
  touch /content/lm/.deps_ok
fi

# (Re)start the controller detached from this ssh session.
pkill -f "python3 -m controller" 2>/dev/null || true
# Reported by /health so the Mac can tell whether the running controller matches this checkout.
export LM_VERSION="${LM_VERSION:-$(find . -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)}"
setsid nohup python3 -m controller --config models.yaml --port "${LM_PORT:-9000}" ${LOAD:+--load "$LOAD"} \
  > /content/lm/logs/controller.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:${LM_PORT:-9000}/health >/dev/null && { echo "controller up"; exit 0; }
  sleep 1
done
echo "controller failed to start:" >&2
tail -n 50 /content/lm/logs/controller.log >&2
exit 1
