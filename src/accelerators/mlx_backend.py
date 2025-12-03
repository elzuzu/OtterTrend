from __future__ import annotations

from typing import Any

from src.hardware import detect_hardware


class MLXBackend:
    def __init__(self) -> None:
        self.hw = detect_hardware()
        self.available = self.hw.supports_mlx

    def load_model(self, model_name: str) -> Any:
        if not self.available:
            raise RuntimeError("MLX backend not available on this hardware")
        return {"model": model_name, "backend": "mlx"}


__all__ = ["MLXBackend"]
