from __future__ import annotations

from typing import Any

from src.hardware import detect_hardware


class CoreMLBackend:
    def __init__(self) -> None:
        self.hw = detect_hardware()
        self.available = self.hw.supports_coreml

    def load_model(self, model_name: str) -> Any:
        if not self.available:
            raise RuntimeError("CoreML backend not available on this hardware")
        return {"model": model_name, "backend": "coreml"}


__all__ = ["CoreMLBackend"]
