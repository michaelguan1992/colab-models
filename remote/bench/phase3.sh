#!/usr/bin/env bash
# Phase-3 work on the runtime, run outside the controller. Each step holds the runtime awake
# (a file in /content/lm/holds) while it runs, so the idle timers don't stop the session.
#   bash phase3.sh basalt        Basalt release + source tree + converter venv
#   bash phase3.sh mtp           Qwen drafter sources (range reads) + the tuned dense.bin
#   bash phase3.sh pika-bench    FP8 variants of the Pika benchmark (Pika must be downloaded)
#   bash phase3.sh qwen-convert  SC117 IQ3_XXS GGUF -> /content/qconv/qwen.basalt
#   bash phase3.sh qwen-upload   push the pack to the private HF repo
set -euo pipefail

BT=/content/bt
BASALT=$BT/basalt-v0.5.6-linux-x64
PY=$BASALT/.venv/bin/python
Q=/content/qconv
HOLDS=/content/lm/holds
PACK_REPO=mgresearch/colab-models-basalt

step=$1
mkdir -p "$HOLDS"
touch "$HOLDS/$step"
trap 'rm -f "$HOLDS/$step"' EXIT

case $step in
basalt)
  mkdir -p $BT && cd $BT
  if [ ! -x $PY ]; then
    curl -fsSL -o b.tgz https://github.com/jesdga95/basalt/releases/download/v0.5.6/basalt-v0.5.6-linux-x64.tar.gz
    echo "a1f986a2dc82bbeb8db251ee87147532cf01c1f1a3ad9d955683fc0445883866  b.tgz" | sha256sum -c
    tar xzf b.tgz && rm b.tgz
    (cd $BASALT && python3 -m venv --without-pip --system-site-packages .venv \
      && .venv/bin/python -m pip install -q -r requirements.txt)
  fi
  [ -d $BT/src ] || git clone -q --depth 1 --branch v0.5.6 https://github.com/jesdga95/basalt.git $BT/src
  ;;
mtp)
  cd $BT/src
  $PY tools/convert/mtp_fetch.py fetch --out $Q/mtp
  $PY tools/convert/mtp_fetch.py verify --out $Q/mtp
  hf download jesdga/Qwen3.8-Flash-Next-Basalt convert/mtp/rt/dense.bin --local-dir $Q/official
  ;;
pika-bench)
  cd /content/bench
  P=/content/lm/venvs/diffusers/bin/python
  $P pika_bench.py --tag fp8res --adapter official --fuse --fp8 --fp8-text --resident || echo "FAILED fp8res"
  $P pika_bench.py --tag fp8off --adapter official --fuse --fp8 --fp8-text || echo "FAILED fp8off"
  ;;
qwen-convert)
  # Explicit filenames make hf ignore --include, so the shards and the mmproj are two calls.
  hf download SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF --include 'IQ3_XXS/*' --local-dir $Q/gguf
  hf download SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF mmproj-Qwen3.8-Flash-Next-BF16.gguf --local-dir $Q/gguf
  shard1=$(ls $Q/gguf/IQ3_XXS/*00001-of-*.gguf)
  cd $BT/src
  $PY tools/convert/convert.py --gguf "$shard1" --mmproj $Q/gguf/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
    --out $Q/qwen.basalt --mtp-source $Q/mtp --mtp-dense $Q/official/convert/mtp/rt/dense.bin \
    --name "Qwen3.8-Flash-Next GSQ-RCO abliterated"
  ;;
qwen-upload)
  hf upload $PACK_REPO $Q/qwen.basalt Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.basalt \
    --commit-message "Qwen3.8-Flash-Next GSQ-RCO abliterated IQ3_XXS, Basalt v0.5.6 pack"
  ;;
*)
  echo "unknown step $step" >&2; exit 2
  ;;
esac
echo "STEPDONE $step"
