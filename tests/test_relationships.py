import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:  # noqa: E402
    from diplomacy.negotiation.relationship import DealEvaluation, RelationshipAwareNegotiator
    from diplomacy.types import Power
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    if exc.name == "numpy":
        raise unittest.SkipTest("numpy is required for diplomacy tests") from exc
    raise


def test_relationships_initialize_to_one() -> None:
    negotiator = RelationshipAwareNegotiator(Power("France"))
    negotiator.reset_relationships([Power("France"), Power("Germany"), Power("Italy")])
    assert negotiator.relationships == {
        Power("Germany"): 1.0,
        Power("Italy"): 1.0,
    }


def test_relationship_update_clamped() -> None:
    negotiator = RelationshipAwareNegotiator(
        Power("France"),
        gamma=1.0,
        min_relationship=0.0,
        max_relationship=2.0,
    )
    negotiator.reset_relationships([Power("France"), Power("Germany"), Power("Italy")])
    negotiator.update_relationships(
        proposals={Power("France"): set()},
        contracts=[],
        value_deltas={Power("Germany"): 1.5, Power("Italy"): -2.0},
    )
    assert negotiator.relationships[Power("Germany")] == 2.0
    assert negotiator.relationships[Power("Italy")] == 0.0


def test_relationship_bias_changes_ranking() -> None:
    negotiator = RelationshipAwareNegotiator(Power("France"), partner_utility_weight=1.0)
    negotiator.reset_relationships([Power("France"), Power("Germany"), Power("Italy")])
    negotiator.relationships[Power("Germany")] = 2.0
    negotiator.relationships[Power("Italy")] = 0.5

    deals = [
        DealEvaluation(partner=Power("Germany"), self_delta=0.1, partner_delta=0.4),
        DealEvaluation(partner=Power("Italy"), self_delta=0.1, partner_delta=0.4),
    ]
    ranked = negotiator.rank_deals(deals)
    assert ranked[0].partner == Power("Germany")
