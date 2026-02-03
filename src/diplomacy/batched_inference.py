from __future__ import annotations

import os
from typing import Callable, Iterable, List, Sequence, TypeVar

from .state import GameState
from .types import Power

T = TypeVar("T")

BatchValueFn = Callable[[Sequence[GameState], Power], Sequence[float]]


def get_batch_size(default: int = 16) -> int:
    value = os.getenv("INFERENCE_BATCH_SIZE")
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def chunked_batches(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def run_batched_values(
    batch_fn: BatchValueFn,
    states: Sequence[GameState],
    power: Power,
    *,
    batch_size: int | None = None,
) -> List[float]:
    resolved_batch_size = batch_size or get_batch_size()
    values: List[float] = []
    for batch in chunked_batches(states, resolved_batch_size):
        values.extend(batch_fn(batch, power))
    return values


__all__ = ["BatchValueFn", "get_batch_size", "run_batched_values"]
