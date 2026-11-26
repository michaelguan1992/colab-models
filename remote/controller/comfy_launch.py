"""Start ComfyUI on one model's weights with its preset workflows already saved.

The route's weights folder is laid out like ComfyUI's ``models/`` (diffusion_models/, text_encoders/, vae/,
loras/), so it is handed over as an extra model path. ``--workflow name=title`` copies the official template
``name`` out of the installed comfyui-workflow-templates package into the user's saved workflows as
``title``; the templates already name the files this route downloads. ``--swap old=new`` rewrites that text
in the copies, for a model that runs in an official template with other files or settings. Unknown arguments
go to ComfyUI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def model_paths_yaml(model_dir: Path) -> str:
    kinds = sorted(p.name for p in model_dir.iterdir() if p.is_dir() and p.name != "user" and not p.name.startswith("."))
    lines = ["lm:", f"  base_path: {model_dir}", "  is_default: true", *(f"  {k}: {k}/" for k in kinds)]
    return "\n".join(lines) + "\n"


def find_template(name: str, search: list[str]) -> Path | None:
    for root in search:
        base = Path(root)
        if not base.is_dir():
            continue
        for pkg in sorted(base.glob("comfyui_workflow_templates*")):
            hit = next(iter(sorted(pkg.rglob(f"{name}.json"))), None)
            if hit is not None:
                return hit
    return None


def install_workflows(specs: list[str], user_dir: Path, search: list[str], swaps: list[str] = ()) -> list[str]:
    """Copy each ``name=title`` template into the saved workflows; returns the names that were not found.

    Each ``old=new`` in ``swaps`` is replaced in the copied text (file names, a sampler, a sigma table).
    """
    dest = user_dir / "default" / "workflows"
    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    for spec in specs:
        name, _, title = spec.partition("=")
        src = find_template(name, search)
        if src is None:
            missing.append(name)
            continue
        target = dest / f"{title or name}.json"
        if not target.exists():  # keep the user's edits to a preset across restarts
            text = src.read_text(encoding="utf-8")
            for swap in swaps:
                old, _, new = swap.partition("=")
                text = text.replace(old, new)
            target.write_text(text, encoding="utf-8")
    return missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comfy", required=True, help="path to ComfyUI's main.py")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--workflow", action="append", default=[], metavar="TEMPLATE=TITLE")
    p.add_argument("--swap", action="append", default=[], metavar="OLD=NEW")
    args, passthrough = p.parse_known_args()

    comfy = Path(args.comfy).resolve()
    model_dir = Path(args.model_dir)
    paths = model_dir / "extra_model_paths.yaml"
    paths.write_text(model_paths_yaml(model_dir))
    user_dir = model_dir / "user"  # per model, so each UI lists only the presets its weights can run
    missing = install_workflows(args.workflow, user_dir, sys.path, args.swap)
    if missing:
        print(f"lm: workflow templates not found, skipped: {', '.join(missing)}", flush=True)
    os.chdir(comfy.parent)
    os.execv(sys.executable, [sys.executable, str(comfy), "--listen", "127.0.0.1", "--port", args.port,
                              "--output-directory", args.outputs_dir, "--extra-model-paths-config", str(paths),
                              "--user-directory", str(user_dir),
                              "--disable-auto-launch", *passthrough])


if __name__ == "__main__":
    main()
