#!/usr/bin/env bash
# The MOSS-Image comparison: official 50-step CFG 3 against the step-distilled LoRAs, then FP8 text encoder.
cd /content/moss
B="python3 /content/moss/moss_bench.py"
$B --tag base50
$B --tag lode8 --lora lode/lora.safetensors --steps 8
$B --tag lode8l --lora lode/lora_latest.safetensors --strength 0.7 --steps 8
$B --tag akh4 --lora akh/lora.safetensors --steps 4
$B --tag akh3 --lora akh/lora_3step.safetensors --steps 3
$B --tag akh4fp8 --lora akh/lora.safetensors --steps 4 --fp8-text --resident
echo ALLDONE
