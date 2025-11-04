"""Agent wrapper that drives the DeepMind SL policy inside the local engine."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..deepmind import build_observation
from ..deepmind.observation import _POWER_TO_INDEX
from ..orders import hold, move, support_hold, support_move
from ..state import GameState
from ..types import Order, OrderType, Power, Unit
from .base import Agent


@dataclass(frozen=True)
class _ActionInfo:
    """Metadata describing how a Diplomacy action maps back to the local model."""

    order_type: int
    origin: Optional[str]
    target: Optional[str]
    third: Optional[str]


class DeepMindSLAgent(Agent):
    """Wrap the released SL policy from *No Press Diplomacy* as a local agent.

    The agent lazily loads the DeepMind network implementation together with the
    published SL parameters.  On each turn the local ``GameState`` is converted
    into a DeepMind observation/legals bundle, the policy is queried for the
    controlled power, and the resulting unit actions are mapped back into
    ``Order`` objects understood by the lightweight adjudicator.

    Notes:
        * Only move phases are supported.  Retreat phases currently fall back to
          the deterministic "hold" behaviour from :class:`Agent`.
        * The DeepMind release depends on ``numpy``, ``dill``, ``jax`` and
          ``dm-haiku``.  These are not vendored with the repository, so the
          caller must install them before instantiating the agent.
        * The published weights must be downloaded separately
          (``sl_params.npz`` from the project README) and supplied through the
          ``weights_path`` parameter.
    """

    def __init__(
        self,
        power: Power,
        *,
        weights_path: str | Path,
        temperature: float = 0.2,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(power)
        self._weights_path = Path(weights_path)
        self._temperature = temperature
        self._rng_seed = rng_seed

        self._policy = None
        self._np = None
        self._action_utils = None
        self._action_lookup: Dict[Tuple[int, Optional[str], Optional[str], Optional[str]], List[int]] = {}
        self._action_metadata: Dict[int, _ActionInfo] = {}
        self._province_lookup: Dict[int, str] = {}

        self._last_actions: MutableMapping[str, List[int]] = {power_name: [] for power_name in _POWER_TO_INDEX}

    # ------------------------------------------------------------------
    # Agent hooks
    # ------------------------------------------------------------------
    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        if state.phase.name.endswith("RETREAT"):
            return []

        policy = self._ensure_policy_loaded()
        if round_index == 0:
            policy.reset()
            for key in self._last_actions:
                self._last_actions[key] = []

        legal_arrays, action_map = self._build_legal_actions(state)
        observation = build_observation(state, last_actions=self._flatten_last_actions())

        slots_list = [self._power_index(str(self.power))]
        policy_actions, _ = policy.actions(slots_list, observation, legal_arrays)
        chosen = policy_actions[0]

        orders: List[Order] = []
        for action in chosen:
            mapped = action_map.get(action)
            if mapped is not None:
                orders.append(mapped)

        self._last_actions[str(self.power)] = list(chosen)
        return orders

    # ------------------------------------------------------------------
    # Policy/bootstrap helpers
    # ------------------------------------------------------------------
    def _ensure_policy_loaded(self):
        if self._policy is not None:
            return self._policy

        # Import-heavy dependencies are deferred so that environments without
        # the DeepMind stack can still import the package.
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "DeepMindSLAgent requires the 'numpy' package; install the"
                " DeepMind dependencies to enable the agent."
            ) from exc

        try:  # pragma: no cover - executed when optional deps present
            from diplomacy.environment import action_list, action_utils, province_order
            from diplomacy_main.network import config, network_policy, parameter_provider
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "DeepMind modules are unavailable. Ensure the diplomacy-main"
                " checkout is present and its dependencies are installed."
            ) from exc

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Unable to locate DeepMind weights at {self._weights_path}."
                " Download sl_params.npz from the release README and point"
                " weights_path at that file."
            )

        with io.open(self._weights_path, "rb") as handle:
            provider = parameter_provider.ParameterProvider(handle)

        network_info = config.get_config()
        handler = parameter_provider.SequenceNetworkHandler(
            network_cls=network_info.network_class,
            network_config=network_info.network_kwargs,
            rng_seed=self._rng_seed,
            parameter_provider=provider,
        )

        policy = network_policy.Policy(
            network_handler=handler,
            num_players=len(_POWER_TO_INDEX),
            temperature=self._temperature,
            calculate_all_policies=True,
        )

        self._policy = policy
        self._np = np
        self._action_utils = action_utils
        self._province_lookup = self._build_province_lookup(province_order)
        self._action_lookup, self._action_metadata = self._build_action_tables(
            action_list.POSSIBLE_ACTIONS, action_utils
        )
        return self._policy

    def _build_legal_actions(self, state: GameState) -> Tuple[List["np.ndarray"], Dict[int, Order]]:  # type: ignore[name-defined]
        np = self._np
        if np is None or self._action_utils is None:
            raise RuntimeError("DeepMind policy was not initialised correctly.")

        per_power: Dict[str, List[int]] = {power: [] for power in _POWER_TO_INDEX}
        action_to_order: Dict[int, Order] = {}

        for unit in state.units.values():
            if unit.power != self.power:
                continue
            for order in self._candidate_orders(state, unit):
                encoded = self._encode_order(order)
                if encoded is None:
                    continue
                per_power[str(unit.power)].append(encoded)
                action_to_order[encoded] = order

        arrays: List["np.ndarray"] = []
        for power_name in _POWER_TO_INDEX:
            actions = per_power.get(power_name)
            if actions:
                arrays.append(np.asarray(actions, dtype=np.int64))
            else:
                arrays.append(np.zeros((0,), dtype=np.int64))
        return arrays, action_to_order

    def _candidate_orders(self, state: GameState, unit: Unit) -> Iterator[Order]:
        yield hold(unit)
        for destination in state.legal_moves_from(unit.loc):
            yield move(unit, destination)

        # Restrict supports to friendly units; the simplified engine does not
        # currently model cross-power support nor convoy chains.
        for friend in state.units.values():
            if friend.power != unit.power or friend.loc == unit.loc:
                continue
            if friend.loc in state.graph.neighbors(unit.loc):
                yield support_hold(unit, friend.loc)
            for move_target in state.legal_moves_from(friend.loc):
                if move_target in state.graph.neighbors(unit.loc):
                    yield support_move(unit, friend.loc, move_target)

    def _encode_order(self, order: Order) -> Optional[int]:
        dm_type = self._dm_order_code(order)
        if dm_type is None:
            return None
        key = (dm_type, order.unit.loc, order.target, order.support_unit_loc)
        candidates = self._action_lookup.get(key)
        if not candidates:
            return None
        return candidates[0]

    # ------------------------------------------------------------------
    # Mapping utilities
    # ------------------------------------------------------------------
    def _build_province_lookup(self, province_order_module) -> Dict[int, str]:
        mapping = province_order_module.province_name_to_id(
            province_order_module.MapMDF.BICOASTAL_MAP
        )
        return {idx: name.split("/")[0] for name, idx in mapping.items()}

    def _build_action_tables(
        self,
        possible_actions: Iterable[int],
        action_utils_module,
    ) -> Tuple[
        Dict[Tuple[int, Optional[str], Optional[str], Optional[str]], List[int]],
        Dict[int, _ActionInfo],
    ]:
        lookup: Dict[Tuple[int, Optional[str], Optional[str], Optional[str]], List[int]] = {}
        metadata: Dict[int, _ActionInfo] = {}
        for action in possible_actions:
            order_type, origin, target, third = action_utils_module.action_breakdown(action)
            origin_name = self._province_lookup.get(origin[0]) if origin else None
            target_name = self._province_lookup.get(target[0]) if target else None
            third_name = self._province_lookup.get(third[0]) if third else None
            key = (order_type, origin_name, target_name, third_name)
            lookup.setdefault(key, []).append(action)
            metadata[action] = _ActionInfo(
                order_type=order_type,
                origin=origin_name,
                target=target_name,
                third=third_name,
            )
        return lookup, metadata

    def _dm_order_code(self, order: Order) -> Optional[int]:
        action_utils = self._action_utils
        if action_utils is None:
            return None
        if order.type == OrderType.HOLD:
            return action_utils.HOLD
        if order.type == OrderType.MOVE:
            return action_utils.MOVE_TO
        if order.type == OrderType.SUPPORT:
            if order.support_target:
                return action_utils.SUPPORT_MOVE_TO
            return action_utils.SUPPORT_HOLD
        if order.type == OrderType.RETREAT:
            return action_utils.RETREAT_TO
        return None

    def _flatten_last_actions(self) -> List[int]:
        flattened: List[int] = []
        for power_name in _POWER_TO_INDEX:
            flattened.extend(self._last_actions.get(power_name, []))
        return flattened

    def _power_index(self, name: str) -> int:
        try:
            return _POWER_TO_INDEX[name]
        except KeyError as exc:
            raise ValueError(f"Unknown power for DeepMind agent: {name}") from exc


__all__ = ["DeepMindSLAgent"]

