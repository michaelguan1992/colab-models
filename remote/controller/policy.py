"""GPU scheduling rules for the Colab runtime.

Rules (user-defined):
  1. At most one LLM online at any time.
  2. The video model (Pika) may only be online while the online LLM uses
     less than ``VIDEO_LLM_LIMIT_GB`` of VRAM (or no LLM is online).

Pure functions only, so they can be unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VIDEO_LLM_LIMIT_GB = 20.0


@dataclass(frozen=True)
class Spec:
    id: str
    kind: str  # "llm" | "video"
    est_vram_gb: float


@dataclass(frozen=True)
class Online:
    id: str
    kind: str
    vram_gb: float | None = None  # measured; None = not measured yet


@dataclass
class Decision:
    ok: bool
    unload: list[str] = field(default_factory=list)
    reason: str = ""


def _llm_vram(o: Online, specs: dict[str, Spec]) -> float:
    """Measured VRAM if known, otherwise the registry estimate."""
    if o.vram_gb is not None:
        return o.vram_gb
    return specs[o.id].est_vram_gb if o.id in specs else float("inf")


def plan_load(target: Spec, online: list[Online], specs: dict[str, Spec], swap: bool = True) -> Decision:
    """Decide whether ``target`` may be loaded, and what must be unloaded first."""
    if any(o.id == target.id for o in online):
        return Decision(True, reason=f"{target.id} 已在线")

    llms = [o for o in online if o.kind == "llm"]
    videos = [o for o in online if o.kind == "video"]

    if target.kind == "llm":
        unload: list[str] = []
        if llms:
            if not swap:
                return Decision(False, reason=f"已有 LLM 在线（{', '.join(o.id for o in llms)}），同一时间只允许一个 LLM")
            unload = [o.id for o in llms]
        if videos and target.est_vram_gb >= VIDEO_LLM_LIMIT_GB:
            return Decision(
                False,
                reason=(
                    f"{target.id} 预计占用 {target.est_vram_gb:.0f}GB ≥ {VIDEO_LLM_LIMIT_GB:.0f}GB，"
                    f"视频模型 {', '.join(o.id for o in videos)} 在线时不允许；请先卸载视频模型"
                ),
            )
        return Decision(True, unload=unload)

    if target.kind == "video":
        for o in llms:
            used = _llm_vram(o, specs)
            if used >= VIDEO_LLM_LIMIT_GB:
                return Decision(
                    False,
                    reason=(
                        f"在线 LLM {o.id} 占用 {used:.1f}GB ≥ {VIDEO_LLM_LIMIT_GB:.0f}GB，"
                        f"视频模型不能上线；请先切换到更小的 LLM 或卸载 LLM"
                    ),
                )
        return Decision(True)

    return Decision(False, reason=f"未知模型类型: {target.kind}")


def violations(online: list[Online], specs: dict[str, Spec]) -> list[str]:
    """Rule breaches in the *current* state (e.g. an LLM grew past the limit after loading)."""
    out: list[str] = []
    llms = [o for o in online if o.kind == "llm"]
    if len(llms) > 1:
        out.append(f"同时有 {len(llms)} 个 LLM 在线: {', '.join(o.id for o in llms)}")
    if any(o.kind == "video" for o in online):
        for o in llms:
            if o.vram_gb is not None and o.vram_gb >= VIDEO_LLM_LIMIT_GB:
                out.append(f"视频模型在线，但 LLM {o.id} 实测占用 {o.vram_gb:.1f}GB ≥ {VIDEO_LLM_LIMIT_GB:.0f}GB")
    return out
