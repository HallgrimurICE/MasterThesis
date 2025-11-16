"""Lightweight fallback implementation of the ``dm-tree`` API used in the repo.

This project historically relied on DeepMind's ``dm-tree`` package which
provides a ``tree`` module with helpers such as ``map_structure``.  Installing
``dm-tree`` can be troublesome on some platforms (and pip expects the hyphenated
name ``dm-tree``, not ``dm_tree``).  To make the codebase runnable without the
third-party dependency we provide a tiny subset of the API that covers all usages
in this repository.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Iterable, Iterator, MutableSequence

try:  # pragma: no cover - exercised when dm-tree is available
    import dm_tree as _dm_tree
except ModuleNotFoundError:  # pragma: no cover - the fallback itself is tested separately

    def _is_namedtuple(value: Any) -> bool:
        return isinstance(value, tuple) and hasattr(value, "_fields")

    def _is_tree(value: Any) -> bool:
        return isinstance(value, (list, tuple, Mapping)) and not isinstance(value, (str, bytes))

    def _assert_same_structure(first: Any, other: Any) -> None:
        if _is_tree(first) != _is_tree(other):
            raise TypeError("Structures do not match.")
        if not _is_tree(first):
            return
        if isinstance(first, Mapping):
            if first.keys() != other.keys():
                raise TypeError("Mapping keys differ between structures.")
            for key in first:
                _assert_same_structure(first[key], other[key])
        elif _is_namedtuple(first):
            if type(first) is not type(other) or len(first) != len(other):
                raise TypeError("Namedtuple structures differ.")
            for idx in range(len(first)):
                _assert_same_structure(first[idx], other[idx])
        elif isinstance(first, tuple):
            if len(first) != len(other):
                raise TypeError("Tuple structures differ.")
            for idx in range(len(first)):
                _assert_same_structure(first[idx], other[idx])
        elif isinstance(first, list):
            if len(first) != len(other):
                raise TypeError("List structures differ.")
            for idx in range(len(first)):
                _assert_same_structure(first[idx], other[idx])

    def _map_structure(fn: Callable[..., Any], *structures: Any) -> Any:
        if not structures:
            raise ValueError("map_structure requires at least one structure.")
        first = structures[0]
        for other in structures[1:]:
            _assert_same_structure(first, other)
        if not _is_tree(first):
            return fn(*structures)
        if isinstance(first, Mapping):
            return type(first)((key, _map_structure(fn, *(structure[key] for structure in structures))) for key in first)
        if _is_namedtuple(first):
            return type(first)(*(_map_structure(fn, *(structure[idx] for structure in structures)) for idx in range(len(first))))
        if isinstance(first, tuple):
            return type(first)(_map_structure(fn, *(structure[idx] for structure in structures)) for idx in range(len(first)))
        if isinstance(first, list):
            return [_map_structure(fn, *(structure[idx] for structure in structures)) for idx in range(len(first))]
        raise TypeError(f"Unsupported structure type: {type(first)!r}")

    def map_structure(fn: Callable[..., Any], *structures: Any, **_: Any) -> Any:
        """Apply ``fn`` to each leaf across identically structured trees."""
        return _map_structure(fn, *structures)

    def flatten(structure: Any, *, leaves: MutableSequence[Any] | None = None) -> list[Any]:
        """Return leaves in traversal order matching ``dm-tree.flatten``."""
        if leaves is None:
            leaves = []
        if _is_tree(structure):
            if isinstance(structure, Mapping):
                for key in structure:
                    flatten(structure[key], leaves=leaves)
            elif isinstance(structure, tuple):
                for value in structure:
                    flatten(value, leaves=leaves)
            elif isinstance(structure, list):
                for value in structure:
                    flatten(value, leaves=leaves)
        else:
            leaves.append(structure)
        return leaves

    def unflatten_as(structure: Any, flat_sequence: Iterable[Any]) -> Any:
        iterator: Iterator[Any] = iter(flat_sequence)

        def _unflatten(template: Any) -> Any:
            if not _is_tree(template):
                try:
                    return next(iterator)
                except StopIteration as exc:  # pragma: no cover - sanity guard
                    raise ValueError("flat_sequence was exhausted during unflatten.") from exc
            if isinstance(template, Mapping):
                return type(template)((key, _unflatten(value)) for key, value in template.items())
            if _is_namedtuple(template):
                return type(template)(*(_unflatten(value) for value in template))
            if isinstance(template, tuple):
                return type(template)(_unflatten(value) for value in template)
            if isinstance(template, list):
                return [_unflatten(value) for value in template]
            raise TypeError(f"Unsupported structure type: {type(template)!r}")

        result = _unflatten(structure)
        try:
            next(iterator)
            raise ValueError("flat_sequence has more elements than the template contains.")
        except StopIteration:
            return result

    __all__ = ["map_structure", "flatten", "unflatten_as"]

else:  # pragma: no cover - executed when dm-tree is installed
    map_structure = _dm_tree.map_structure
    flatten = _dm_tree.flatten
    unflatten_as = _dm_tree.unflatten_as
    __all__ = _dm_tree.__all__
