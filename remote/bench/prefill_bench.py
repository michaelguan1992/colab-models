"""Long-prompt read time through the controller: a passphrase buried at the start of ~N tokens of filler.

  uv run python remote/bench/prefill_bench.py --tag qwen-basalt --words 21000
Reports time to first token (prompt read plus one decode step) and whether the answer recalls the passphrase.
"""

import argparse
import json
import random
import time
import urllib.request

WORDS = ("river stone lantern harbor maple quiet engine orbit velvet canyon signal meadow copper whisper "
         "glacier tunnel saffron ember compass thunder").split()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--url", default="http://127.0.0.1:9000")
    p.add_argument("--model", default="qwen")
    p.add_argument("--words", type=int, default=21000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rnd = random.Random(a.seed)
    filler = " ".join(rnd.choice(WORDS) for _ in range(a.words))
    prompt = (f"The passphrase is AZURE-PELICAN-4471. Remember it.\n\n{filler}\n\n"
              "What was the passphrase given at the very start? Reply with the passphrase only.")
    body = {"model": a.model, "max_tokens": 40, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{a.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    usage = out.get("usage", {})
    answer = out["choices"][0]["message"].get("content") or ""
    print(json.dumps({"tag": a.tag, "prompt_tokens": usage.get("prompt_tokens"), "total_s": round(time.monotonic() - t0, 2),
                      "recalled": "AZURE-PELICAN-4471" in answer, "answer": answer.strip()[:80]}), flush=True)


if __name__ == "__main__":
    main()
