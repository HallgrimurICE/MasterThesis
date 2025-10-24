from __future__ import annotations

from typing import Dict, Iterable, List, Set, Union

try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore

    NX_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback used when networkx missing
    NX_AVAILABLE = False

    class _FallbackGraph:
        """Minimal undirected graph supporting the operations used in tests."""

        def __init__(self):
            self._adjacency: Dict[str, Set[str]] = {}
            self._node_attrs: Dict[str, Dict[str, Union[bool, str]]] = {}

        def add_node(self, node: str, **attrs: Union[bool, str]) -> None:
            self._adjacency.setdefault(node, set())
            if attrs:
                existing = self._node_attrs.setdefault(node, {})
                existing.update(attrs)

        def add_edge(self, u: str, v: str) -> None:
            self.add_node(u)
            self.add_node(v)
            self._adjacency[u].add(v)
            self._adjacency[v].add(u)

        def neighbors(self, node: str) -> Iterable[str]:
            return iter(self._adjacency.get(node, set()))

        def nodes(self, data: bool = False) -> List[Union[str, tuple[str, Dict[str, Union[bool, str]]]]]:
            if data:
                return [
                    (node, dict(self._node_attrs.get(node, {})))
                    for node in self._adjacency
                ]
            return list(self._adjacency)

        def __getitem__(self, node: str) -> Set[str]:
            return self._adjacency.get(node, set())

    class _FallbackNX:
        Graph = _FallbackGraph

        def __getattr__(self, name: str):  # pragma: no cover - diagnostics only
            raise RuntimeError(
                "networkx is required for visualization features and advanced graph "
                f"operations (attempted to access '{name}')."
            )

    nx = _FallbackNX()  # type: ignore

__all__ = ["nx", "NX_AVAILABLE"]
