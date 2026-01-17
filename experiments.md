# ToM Experiment Plan (Compute-Limited)

This document captures the proposed experiment list for evaluating ToM depth (especially ToM2)
under a strict compute budget.

## Fixed settings (all experiments)

- **Rounds:** 50 (or 75 if stable)
- **Budgets:** `n_rollouts = 1`, `k_candidates = 1`, `action_rollouts = 1`
- **Seeds:** same schedule across conditions
- **Evaluation point:** consistent (after adjustments)

## Metrics (report for every experiment)

- **Final supply centers per power** (mean ± std over games)
- **Win rate / top-1 rate** (winner-defined or “most SCs”)
- **Optional:** SC trajectory at rounds 10/20/30/40/50

## A) Self-play — does ToM help when everyone has it?

**Exp A1 — Baseline self-play (control)**
- All powers: ToM1 (baseline negotiator assumption)

**Exp A2 — ToM2 self-play**
- All powers: ToM2

**Claim supported if:** A2 shows higher average SC growth stability / fewer collapses or higher decisive outcomes vs A1.

**Why needed:** Demonstrates ToM2 isn’t only “exploiting ToM1.”

## B) Cross-play — does ToM2 beat ToM1?

**Exp B1 — 1×ToM2 vs 6×ToM1**
- One focal power (e.g., Turkey) as ToM2
- All others ToM1
- Rotate focal power across 2–3 powers if possible

**Exp B2 — 1×ToM1 vs 6×ToM2 (reverse)**
- Same setup, but focal power is ToM1, others ToM2

**Claim supported if:** ToM2 focal power gets higher final SC than ToM1 focal power under symmetric conditions.

**Rationale:** Same search budget, only ToM depth differs.

## C) Dose-response — is there a monotonic trend with ToM depth?

**Exp C1 — Mixed depths in the same game**

Run three conditions:
- 2 powers ToM2, 5 powers ToM1
- 4 powers ToM2, 3 powers ToM1
- 6 powers ToM2, 1 power ToM1

**Claim supported if:** ToM2 powers collectively dominate SC share increasingly as ToM2 population share rises.

## D) Long-horizon negotiation benefit — does ToM2 reduce wasted proposals?

**Exp D1 — Proposal acceptance rate (ToM1 vs ToM2)**

For A1 and A2, log:
- Proposals sent
- Proposals accepted (mutual)
- Accepted deal rate per round

**Claim supported if:** ToM2 has higher acceptance rate or higher accepted-deal quality (e.g., accepted deals per round).

**Rationale:** Tests whether ToM2 better filters proposals against opponent BATNA.
