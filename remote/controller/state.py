"""Remote state file: which engines this controller started, so a restarted controller can reap them."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("lm.controller")


class StateFile:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError as e:
            log.warning("无法写入状态文件 %s: %s", self.path, e)

    def set(self, model_id: str, entry: dict[str, Any]) -> None:
        data = self.read()
        data[model_id] = entry
        self._write(data)

    def remove(self, model_id: str) -> None:
        data = self.read()
        if data.pop(model_id, None) is not None:
            self._write(data)

    def clear(self) -> None:
        if self.path.exists():
            self._write({})

