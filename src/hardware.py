from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareInfo:
    os: str
    machine: str
    processor: str
    supports_mlx: bool
    supports_coreml: bool


def detect_hardware() -> HardwareInfo:
    machine = platform.machine()
    os_name = platform.system()
    processor = platform.processor()
    supports_mlx = os_name == "Darwin" and machine in {"arm64", "aarch64"}
    supports_coreml = supports_mlx
    return HardwareInfo(
        os=os_name,
        machine=machine,
        processor=processor,
        supports_mlx=bool(supports_mlx),
        supports_coreml=bool(supports_coreml),
    )


__all__ = ["detect_hardware", "HardwareInfo"]
