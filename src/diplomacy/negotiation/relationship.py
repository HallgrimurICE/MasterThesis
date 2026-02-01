from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

from ..state import GameState
from ..types import Power
from .contracts import Contract
from .peace import build_peace_contract
from .simulation import PolicyFn, StepFn, ValueFn, estimate_expected_values


@dataclass(frozen=True)
class DealEvaluation:
    partner: Power
    self_delta: float
    partner_delta: float


class RelationshipAwareNegotiator:
    """Relationship-aware negotiation helper based on Eq. 16/17 of the ToM paper."""

    def __init__(
        self,
        power: Power,
        *,
        gamma: float = 0.1,
        min_relationship: float = 0.0,
        max_relationship: float = 2.0,
        proposal_threshold: float = 0.0,
        proposal_bias: float = 0.05,
        partner_utility_weight: float = 0.5,
        log_relationships: bool = False,
    ) -> None:
        self.power = power
        self.gamma = gamma
        self.min_relationship = min_relationship
        self.max_relationship = max_relationship
        self.proposal_threshold = proposal_threshold
        self.proposal_bias = proposal_bias
        self.partner_utility_weight = partner_utility_weight
        self.log_relationships = log_relationships
        self.relationships: Dict[Power, float] = {}
        self._last_value_deltas: Dict[Power, float] = {}

    def reset_relationships(self, powers: Iterable[Power]) -> None:
        self.relationships = {p: 1.0 for p in powers if p != self.power}

    def relationship_for(self, other: Power) -> float:
        return self.relationships.get(other, 1.0)

    def score_deal(self, deal: DealEvaluation) -> float:
        relationship = self.relationship_for(deal.partner)
        return deal.self_delta + relationship * self.partner_utility_weight * deal.partner_delta

    def rank_deals(self, deals: Sequence[DealEvaluation]) -> Sequence[DealEvaluation]:
        return sorted(deals, key=self.score_deal, reverse=True)

    def propose_partners(
        self,
        state: GameState,
        *,
        powers: Sequence[Power],
        legal_actions: Mapping[Power, Sequence[int]],
        policy_fns: Mapping[Power, PolicyFn],
        value_fn: ValueFn,
        step_fn: StepFn,
        rollouts: int = 4,
        tom_depth: int = 1,
    ) -> Set[Power]:
        self._last_value_deltas = {}
        baseline_values = estimate_expected_values(
            state,
            target_powers=powers,
            policy_fns=policy_fns,
            value_fn=value_fn,
            step_fn=step_fn,
            legal_actions=legal_actions,
            restricted_actions=None,
            rollouts=rollouts,
        )
        baseline = baseline_values.get(self.power, 0.0)

        proposals: Set[Power] = set()
        for other in powers:
            if other == self.power:
                continue
            contract = build_peace_contract(
                state,
                self.power,
                other,
                legal_actions[self.power],
                legal_actions[other],
            )
            restrictions = {
                self.power: tuple(contract.allowed_i),
                other: tuple(contract.allowed_j),
            }
            deal_values = estimate_expected_values(
                state,
                target_powers=powers,
                policy_fns=policy_fns,
                value_fn=value_fn,
                step_fn=step_fn,
                legal_actions=legal_actions,
                restricted_actions=restrictions,
                rollouts=rollouts,
            )

            if tom_depth <= 0:
                proposals.add(other)
                continue

            deal_value = deal_values.get(self.power, baseline)
            self_delta = deal_value - baseline
            other_baseline = baseline_values.get(other, 0.0)
            other_delta = deal_values.get(other, other_baseline) - other_baseline
            self._last_value_deltas[other] = self_delta

            if tom_depth >= 2 and other_delta <= 0:
                continue

            deal_score = self.score_deal(
                DealEvaluation(partner=other, self_delta=self_delta, partner_delta=other_delta)
            )
            relationship = self.relationship_for(other)
            threshold = self.proposal_threshold + (1.0 - relationship) * self.proposal_bias
            if deal_score <= threshold:
                continue
            proposals.add(other)
        return proposals

    def update_relationships(
        self,
        *,
        proposals: Mapping[Power, Set[Power]],
        contracts: Sequence[Contract],
        value_deltas: Optional[Mapping[Power, float]] = None,
    ) -> None:
        if not self.relationships:
            return

        accepted_partners = {
            contract.player_j
            for contract in contracts
            if contract.player_i == self.power
        } | {
            contract.player_i
            for contract in contracts
            if contract.player_j == self.power
        }
        proposed_partners = proposals.get(self.power, set())

        resolved_deltas = value_deltas or self._last_value_deltas
        for other in self.relationships:
            if resolved_deltas and other in resolved_deltas:
                delta = resolved_deltas[other]
            elif other in accepted_partners:
                delta = 1.0
            elif other in proposed_partners:
                delta = -1.0
            else:
                delta = 0.0
            updated = self.relationships[other] + self.gamma * delta
            self.relationships[other] = self._clamp(updated)

        if self.log_relationships:
            formatted = ", ".join(
                f"{other}:{self.relationships[other]:.2f}" for other in sorted(self.relationships, key=str)
            )
            print(f"[relationships] {self.power} -> {{{formatted}}}")

    def _clamp(self, value: float) -> float:
        return max(self.min_relationship, min(self.max_relationship, value))


__all__ = ["RelationshipAwareNegotiator", "DealEvaluation"]
