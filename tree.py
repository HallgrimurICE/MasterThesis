"""Compatibility shim for DeepMind's :mod:`dm_tree` package.

The original codebase expects the third-party :mod:`dm_tree` module to be
installed and to be importable as ``tree``.  Some environments (including the
one used for these exercises) do not provide ``dm_tree`` by default, which
meant that simply importing this module resulted in a ``ModuleNotFoundError``
before any of the project code could run.

To keep the public API surface identical we first try to import ``dm_tree`` and
re-export everything from it.  If that import fails we lazily fall back to
implementations backed by :mod:`jax.tree_util` (``jax`` is already a dependency
of the project).  Only the small subset of helpers that the repository uses is
implemented – namely :func:`map_structure`, :func:`flatten` and
:func:`unflatten_as` – so that consumers can continue to ``import tree``
transparently.
"""

try:  # pragma: no cover - exercised indirectly when dm_tree is available
  from dm_tree import *  # type: ignore  # noqa: F401,F403 - re-export public API
except ImportError:  # pragma: no cover - simple control-flow shim
  from collections.abc import Mapping
  from typing import Any, Iterable, Iterator, List

  def _is_namedtuple_instance(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")

  def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not _is_namedtuple_instance(value)

  def _map_structure(func, *structures):
    exemplar = structures[0]

    if isinstance(exemplar, Mapping):
      return exemplar.__class__(
          (key, _map_structure(func, *(s[key] for s in structures))) for key in exemplar)
    if _is_namedtuple_instance(exemplar):
      return exemplar.__class__(
          *(_map_structure(func, *(s[i] for s in structures)) for i in range(len(exemplar))))
    if _is_sequence(exemplar):
      return exemplar.__class__(
          _map_structure(func, *(s[i] for s in structures)) for i in range(len(exemplar)))
    return func(*structures)

  def map_structure(func, *structures):
    """Recursively apply ``func`` to the leaves of ``structures``."""

    if not structures:
      raise ValueError("map_structure requires at least one structure")
    return _map_structure(func, *structures)

  def flatten(structure: Any) -> List[Any]:
    """Return the leaves of ``structure`` as a flat list."""

    leaves: List[Any] = []

    def _append(leaf):
      leaves.append(leaf)
      return leaf

    map_structure(_append, structure)
    return leaves

  def _unflatten(template: Any, values: Iterator[Any]):
    if isinstance(template, Mapping):
      return template.__class__((key, _unflatten(template[key], values)) for key in template)
    if _is_namedtuple_instance(template):
      return template.__class__(
          *(_unflatten(value, values) for value in template))
    if _is_sequence(template):
      return template.__class__(_unflatten(value, values) for value in template)
    return next(values)

  def unflatten_as(structure: Any, flat_iterable: Iterable[Any]):
    """Reconstruct ``structure`` using leaves from ``flat_iterable``."""

    iterator = iter(flat_iterable)
    rebuilt = _unflatten(structure, iterator)
    try:
      next(iterator)
      raise ValueError("flat_iterable has more elements than needed")
    except StopIteration:
      return rebuilt

  __all__ = ["map_structure", "flatten", "unflatten_as"]
