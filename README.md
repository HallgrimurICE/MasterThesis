# MasterThesis

## Relationship-aware negotiator

This repo includes a relationship-aware RSS negotiator helper (`RelationshipAwareNegotiator`) that maintains directed
per-opponent scores `U_i->j`, initializes them to 1, updates them each round with a configurable `gamma` signal, and
clamps them to a bounded range. The negotiator uses those relationship values to bias which peace proposals are sent
and to rank candidate deals via partner-utility weighting. The implementation follows the “Relationship Based Baseline
Negotiator agents” description in *Theory_of_mind_in_the_game_of_Diplomacy-3.pdf*, Section 3.2 (Eq. 16–17). See
`src/diplomacy/negotiation/relationship.py` for details.

To enable relationship-aware negotiation in the demos, pass `use_relationships=True` (and optionally
`relationship_gamma` / `log_relationships`) to `run_standard_board_br_vs_neg` or `run_standard_board_mixed_tom_demo`.
