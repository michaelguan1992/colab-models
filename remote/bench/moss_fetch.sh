#!/usr/bin/env bash
# MOSS-Image SFT plus the step-distilled LoRAs compared by moss_bench.py.
set -euo pipefail
export HF_XET_HIGH_PERFORMANCE=1
cd /content/moss
hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir sft --exclude 'assets/*'
hf download lodestones/MOSS-Image-SFT-8-steps-dmd lora.safetensors lora_latest.safetensors --local-dir lode
hf download akhaliq/MOSS-Image-SFT-4-steps lora.safetensors lora_3step.safetensors --local-dir akh
pip install -q "torchao>=0.16"
echo DLDONE
