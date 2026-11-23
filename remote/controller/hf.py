"""HuggingFace access preflight: fail before any eviction or download when the repo is unreachable."""

from __future__ import annotations

import logging

log = logging.getLogger("lm.controller")

ACCESS_ERRORS = {"GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError"}
ACCESS_STATUS = {401, 403, 404}


def classify_hf_error(repo: str, exc: BaseException) -> str | None:
    """A refusal reason for auth/not-found errors, ``None`` for anything else (network, 5xx)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if {c.__name__ for c in type(exc).__mro__} & ACCESS_ERRORS or status in ACCESS_STATUS:
        return (f"无法访问 HuggingFace 仓库 {repo}（受限、私有或不存在）；"
                f"如果是受限仓库，请设置 HF_TOKEN 后重试")
    return None


def hf_preflight(repo: str, revision: str | None = None) -> str | None:
    """Returns a refusal reason, or ``None`` to proceed (including when the check itself fails)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub 未安装，跳过 %s 的访问检查", repo)
        return None
    try:
        HfApi().model_info(repo, revision=revision)  # token=None: taken from HF_TOKEN / cached login
    except Exception as e:
        reason = classify_hf_error(repo, e)
        if reason:
            return reason
        log.warning("检查 %s 访问权限时出错（继续加载）: %s", repo, e)
    return None
