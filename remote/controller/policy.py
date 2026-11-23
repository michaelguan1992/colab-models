"""GPU scheduling for the Colab runtime: rules as data, evaluated as invariants on the post-load online set.

Two rule primitives (registry ``rules:``) plus the always-on total-VRAM fit:
  - ``MaxOnline(group, limit, swap)``: at most ``limit`` members of ``group`` online. With ``swap`` a
    load unloads same-group members to make room; otherwise it is refused.
  - ``RequiresBelow(group, other, vram_gb)``: while any ``group`` member is online, every online
    ``other`` member must use less than ``vram_gb`` (measured if known, else the registry estimate).

``plan_load`` decides everything (rules, VRAM, disk, what to unload and what weights to delete)
before any side effect, so a refused load leaves the runtime untouched. Pure functions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

GiB = 1024**3


@dataclass(frozen=True)
class Spec:
    id: str
    group: str
    est_vram_gb: float


@dataclass(frozen=True)
class Online:
    id: str
    group: str
    vram_gb: float | None = None  # measured; None = not measured yet


@dataclass(frozen=True)
class MaxOnline:
    group: str
    limit: int
    swap: bool


@dataclass(frozen=True)
class RequiresBelow:
    group: str
    other: str
    vram_gb: float


Rule = MaxOnline | RequiresBelow


@dataclass
class Decision:
    ok: bool
    unload: list[str] = field(default_factory=list)
    reason: str = ""


def _vram(o: Online, specs: Mapping[str, Spec]) -> tuple[float, bool]:
    """(GB, measured): measured VRAM if known, otherwise the registry estimate."""
    if o.vram_gb is not None:
        return o.vram_gb, True
    return (specs[o.id].est_vram_gb if o.id in specs else float("inf")), False


def violations(online: Sequence[Online], specs: Mapping[str, Spec], rules: Sequence[Rule]) -> list[str]:
    """Every rule breach in ``online``; the same engine checks a planned set and the live one."""
    out: list[str] = []
    for r in rules:
        members = [o for o in online if o.group == r.group]
        if isinstance(r, MaxOnline):
            if len(members) > r.limit:
                out.append(f"{r.group} 组最多同时在线 {r.limit} 个，现有 {len(members)} 个: "
                           f"{', '.join(o.id for o in members)}")
        elif members:
            names = "、".join(o.id for o in members)
            for o in online:
                if o.group != r.other:
                    continue
                gb, measured = _vram(o, specs)
                if gb >= r.vram_gb:
                    out.append(f"{r.group} 组的 {names} 在线时，{r.other} 组的 {o.id} 必须低于 {r.vram_gb:.0f}GB，"
                               f"但它{'实测占用' if measured else '预计占用'} {gb:.1f}GB；"
                               f"请先卸载 {o.id} 或 {names}")
    return out


@dataclass(frozen=True)
class DiskView:
    free_bytes: int
    weights_bytes: Mapping[str, int]  # on-disk weights per registry model id (target's included)
    need_bytes: int                   # what the target still has to write
    reserve_bytes: int
    inflight_bytes: int               # what other in-progress downloads still have to write


@dataclass(frozen=True)
class Plan:
    ok: bool
    reason: str = ""
    unload: list[str] = field(default_factory=list)
    evict: list[str] = field(default_factory=list)  # models whose on-disk weights get deleted


def plan_rules(target: Spec, online: Sequence[Online], specs: Mapping[str, Spec],
               rules: Sequence[Rule], swap: bool) -> Decision:
    if any(o.id == target.id for o in online):
        return Decision(True, reason=f"{target.id} 已在线")

    unload: list[str] = []
    for r in rules:
        if not isinstance(r, MaxOnline) or r.group != target.group:
            continue
        members = [o for o in online if o.group == r.group and o.id not in unload]
        excess = len(members) + 1 - r.limit
        if excess <= 0:
            continue
        if not (swap and r.swap):
            return Decision(False, reason=(
                f"已有 {r.group} 组的 {', '.join(o.id for o in members)} 在线，同组最多 {r.limit} 个"
                f"{'，且本次不允许换出' if r.swap else '，规则不允许换出'}"))
        unload += [o.id for o in members[:excess]]

    post = [o for o in online if o.id not in unload] + [Online(target.id, target.group)]
    bad = violations(post, {**specs, target.id: target}, rules)
    return Decision(False, reason="；".join(bad)) if bad else Decision(True, unload=unload)


def plan_load(target: Spec, online: Sequence[Online], specs: Mapping[str, Spec], rules: Sequence[Rule], *,
              total_gb: float | None, disk: DiskView, margin_gb: float = 0.0, swap: bool = True) -> Plan:
    """Decide whether ``target`` may be loaded, what to unload first and which weights to delete.

    ``total_gb`` is the assigned GPU's VRAM; ``None`` (unknown) skips the VRAM check. When
    not ok, ``unload`` and ``evict`` are empty.
    """
    d = plan_rules(target, online, specs, rules, swap)
    if not d.ok or any(o.id == target.id for o in online):
        return Plan(d.ok, d.reason)
    staying = [o for o in online if o.id not in d.unload]
    if total_gb is not None:
        need = sum(_vram(o, specs)[0] for o in staying) + target.est_vram_gb
        budget = total_gb - margin_gb
        if need > budget:
            names = "、".join(o.id for o in staying)
            return Plan(False, reason=(
                f"显存不够：{target.id} 预计 {target.est_vram_gb:.0f}GB"
                + (f" + 在线的 {names}" if staying else "")
                + f" ≈ {need:.0f}GB，超过本机可用 {budget:.0f}GB（{total_gb:.0f}GB − 余量 {margin_gb:.0f}GB）"))

    required = disk.need_bytes + disk.reserve_bytes + disk.inflight_bytes
    free = disk.free_bytes
    staying_ids = {o.id for o in staying}
    # Models being unloaded are only evicted if the space is actually needed.
    candidates = sorted(((size, m) for m, size in disk.weights_bytes.items()
                         if size > 0 and m != target.id and m not in staying_ids), reverse=True)
    evict: list[str] = []
    for size, m in candidates:
        if free >= required:
            break
        evict.append(m)
        free += size
    if free < required:
        extra = f"，另有其他下载还需写入 {disk.inflight_bytes / GiB:.0f}GB" if disk.inflight_bytes else ""
        return Plan(False, reason=(
            f"磁盘不够：{target.id} 还需约 {disk.need_bytes / GiB:.0f}GB（含 {disk.reserve_bytes / GiB:.0f}GB 余量{extra}），"
            f"只剩 {disk.free_bytes / GiB:.0f}GB，删除可删的离线权重后也只有 {free / GiB:.0f}GB；请先卸载其他模型"))
    return Plan(True, unload=list(d.unload), evict=evict)
