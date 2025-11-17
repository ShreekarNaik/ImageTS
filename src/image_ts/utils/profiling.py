"""Simple profiling helpers."""
from __future__ import annotations

import contextlib
import time
from typing import Iterator, Callable


@contextlib.contextmanager
def time_block(label: str, callback: Callable[[str, float], None]) -> Iterator[None]:
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    callback(label, duration)
