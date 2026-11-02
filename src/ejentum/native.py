"""
Ejentum Native Rust Acceleration Layer.
Loads ejentum_rs PyO3 extension when available with automatic fallback to pure Python.
Copyright 2026 Ejentum AI. Apache-2.0 License.
"""

from typing import Any, Dict, List, Optional

try:
    import ejentum_rs
    HAS_NATIVE_EXTENSION = True
except ImportError:
    ejentum_rs = None
    HAS_NATIVE_EXTENSION = False


def is_native_available() -> bool:
    """Returns True if the PyO3 native Rust extension is compiled and available."""
    return HAS_NATIVE_EXTENSION


def get_native_version() -> Optional[str]:
    """Returns the native Rust engine version if available."""
    if HAS_NATIVE_EXTENSION and ejentum_rs is not None:
        return ejentum_rs.get_engine_version()
    return None


def fast_iou(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float) -> float:
    """Zero-overhead IoU computation using native Rust accelerator."""
    if HAS_NATIVE_EXTENSION and ejentum_rs is not None:
        return ejentum_rs.compute_iou(x1, y1, x2, y2, x3, y3, x4, y4)

    # Pure Python fallback
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = max(0.0, (x2 - x1) * (y2 - y1))
    a2 = max(0.0, (x4 - x3) * (y4 - y3))
    union = a1 + a2 - inter
    return (inter / union) if union > 0.0 else 0.0


def fast_directional(
    x1: float, y1: float, x2: float, y2: float,
    predicate: str,
    x3: float, y3: float, x4: float, y4: float
) -> bool:
    """Evaluates directional positioning using native Rust accelerator."""
    if HAS_NATIVE_EXTENSION and ejentum_rs is not None:
        return ejentum_rs.check_directional(x1, y1, x2, y2, predicate, x3, y3, x4, y4)

    p = predicate.lower()
    if p == "above":
        return y2 <= y3
    elif p == "below":
        return y1 >= y4
    elif p == "left":
        return x2 <= x3
    elif p == "right":
        return x1 >= x4
    return False
