"""Same prompts through the controller's /v1/chat/completions: first-token latency, decode rate, answers.

Run on the Mac with the tunnel up:  uv run python remote/bench/chat_bench.py --tag qwen-basalt --out bench.jsonl
Decode rate counts streamed content and reasoning deltas over the time after the first one; the token count
comes from the final usage block when the engine sends one.
"""

import argparse
import json
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function that returns the longest palindromic substring of a string, with a short "
            "explanation of its time complexity.",
    "zh": "用三段话解释为什么天空是蓝色的，最后一段要提到日落为什么是红色的。",
    "math": "A train leaves at 9:40 and travels 288 km at 72 km/h, then waits 25 minutes, then travels 150 km at "
            "100 km/h. At what time does it arrive? Show the steps.",
    "long": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
}


def run(url: str, model: str, prompt: str, max_tokens: int, effort: str | None) -> dict:
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], "stream_options": {"include_usage": True}}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    first = None
    deltas, content, reasoning, usage = 0, [], [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                d = choice.get("delta", {})
                piece = d.get("content") or ""
                think = d.get("reasoning_content") or d.get("reasoning") or ""
                if piece or think:
                    first = first or time.monotonic()
                    deltas += 1
                    content.append(piece)
                    reasoning.append(think)
    end = time.monotonic()
    tokens = (usage or {}).get("completion_tokens") or deltas
    decode_s = end - (first or end)
    return {"ttft_s": round((first or end) - t0, 2), "total_s": round(end - t0, 1), "tokens": tokens,
            "tok_per_s": round((tokens - 1) / decode_s, 1) if decode_s > 0 else None,
            "reasoning_chars": len("".join(reasoning)), "answer": "".join(content)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--url", default="http://127.0.0.1:9000")
    p.add_argument("--model", default="qwen")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--effort", help="reasoning_effort to send, e.g. none")
    p.add_argument("--out", default="chat_bench.jsonl")
    a = p.parse_args()
    run(a.url, a.model, "Say hi.", 16, a.effort)  # warm-up: first request after load pays one-time costs
    with open(a.out, "a") as f:
        for name, prompt in PROMPTS.items():
            res = {"tag": a.tag, "prompt": name, **run(a.url, a.model, prompt, a.max_tokens, a.effort)}
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(json.dumps({k: v for k, v in res.items() if k != "answer"}), flush=True)


if __name__ == "__main__":
    main()
