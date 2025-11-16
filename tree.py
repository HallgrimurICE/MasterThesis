"""Lightweight compatibility shim for dm-tree.

The original project depends on DeepMind's ``dm-tree`` package, which provides
``map_structure``-style helpers similar to ``jax.tree_util``.  Shipping another
external dependency makes it harder to run the scripts in constrained
environments, so we provide the tiny subset that this repository actually uses.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from jax import tree_util


Tree = Any


def flatten(structure: Tree) -> list[Any]:
    """Returns the leaves of ``structure`` as a flat Python list."""
    leaves, _ = tree_util.tree_flatten(structure)
    return list(leaves)


def unflatten_as(structure: Tree, flat_sequence: Iterable[Any]) -> Tree:
    """Rebuilds a pytree with ``structure``\'s shape from ``flat_sequence``."""
    _, treedef = tree_util.tree_flatten(structure)
    return tree_util.tree_unflatten(treedef, list(flat_sequence))


def map_structure(func: Callable[..., Any], *structures: Tree) -> Tree:
    """Applies ``func`` to the leaves of ``structures`` like ``dm-tree`` does."""
    return tree_util.tree_map(func, *structures)
