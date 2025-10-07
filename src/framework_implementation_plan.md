# Diplomacy Theory-of-Mind Framework Implementation Plan

## 1. Objectives and Scope
- Reproduce DeepMind's honest baseline negotiator stack (SBR, SVR, STAVE, MBDS) as the mechanical core for a negotiation-capable Diplomacy agent so that we can extend it with theory-of-mind and relationship reasoning proposed in the report.【F:src/report_sentences.txt†L2-L99】
- Support higher-order Theory of Mind (ToM) reasoning that filters proposals using opponents' fallback values and recursive BATNA updates, enabling agents to anticipate how deals will be evaluated by different opponent orders.【F:src/report_sentences.txt†L193-L237】
- Layer a relationship-aware value model that updates interpersonal scores after every round and adjusts state evaluation to capture alliance dynamics, betrayal incentives, and sanctioning behavior.【F:src/report_sentences.txt†L260-L356】【F:src/report_sentences.txt†L373-L399】

## 2. System Architecture Overview
- **Environment layer**: Map the simplified Diplomacy graph, province ownership, unit state, and order resolution rules needed for simultaneous moves and support interactions.【F:src/report_sentences.txt†L34-L53】 Use this as the canonical state representation shared by all modules.
- **Simulation & evaluation layer**: Implement SBR, SVR, and STAVE primitives that can roll out trajectories against policy models, compute expected values, and evaluate candidate contracts under move restrictions.【F:src/report_sentences.txt†L72-L99】
- **Negotiation protocols**: Encode the MBDS proposal/choose pipeline with BATNA refinement, contract sampling, Nash bargaining scoring, and action selection under agreed restrictions.【F:src/report_sentences.txt†L102-L188】
- **Reasoning extensions**: Add ToM^k scoring and recursive BATNA estimation, plus the relationship score updater and relationship-aware value function to bias both proposal ranking and action selection.【F:src/report_sentences.txt†L193-L290】
- **Experiment harness**: Provide a game runner capable of pairing different agent configurations, logging deals, relationships, and outcomes to reproduce the case studies and future experiments described in the report.【F:src/report_sentences.txt†L373-L399】

## 3. Implementation Roadmap

### Phase 1 – Reconstruct the Baseline Environment & Data Model
- **Task P1.1 – Board abstraction scaffolding (codex-agent-env-1)**: Implement immutable data classes for provinces, adjacencies, and initial unit placements, plus conversion helpers to align with any adjudicator coordinates.【F:src/report_sentences.txt†L34-L41】 Deliverables: board module with unit tests covering move adjacency, supply center ownership, and setup serialization.
- **Task P1.2 – Order resolution kernel (codex-agent-env-2)**: Build simultaneous move resolution covering supports, bounces, and convoy outcomes per Figure 3, returning post-adjudication state deltas.【F:src/report_sentences.txt†L42-L53】 Deliverables: resolver service with replayable test cases for each rule interaction.
- **Task P1.3 – State feature interface (codex-agent-env-3)**: Define observation tensors/feature dicts exposing province control, unit positions, and relationship placeholders, ensuring compatibility with SBR/STAVE evaluators. Deliverables: typed schema objects, serialization helpers, and documentation of tensor shapes.
- **Task P1.4 – Policy/value adapter shims (codex-agent-env-4)**: Integrate pretrained No-Press models or create deterministic stubs that emit policy logits and value estimates consumed during rollouts.【F:src/report_sentences.txt†L65-L76】 Deliverables: adapter layer with dependency injection hooks and smoke tests verifying outputs flow into STAVE.

### Phase 2 – Baseline Negotiator Mechanics (ToM¹)
- **Task P2.1 – MBDS contract sampler (codex-agent-neg-1)**: Generate bilateral action restriction proposals using policy-driven sampling and evaluate expected returns via STAVE rollouts.【F:src/report_sentences.txt†L116-L188】 Deliverables: sampler API with configurable proposal batch size K and regression tests on sampled contract diversity.
- **Task P2.2 – Nash scoring engine (codex-agent-neg-2)**: Implement Eq. (11) Nash Bargaining Score computation, fallback enforcement, and choose-phase selection consistent with SBR restrictions.【F:src/report_sentences.txt†L182-L193】 Deliverables: scoring module plus fixtures verifying edge cases (e.g., symmetric payoffs, fallback dominance).
- **Task P2.3 – Negotiated action executor (codex-agent-neg-3)**: Route accepted contracts into the resolver, enforce restricted move sets, and gracefully fallback to unrestricted SBR sampling when negotiations fail.【F:src/report_sentences.txt†L87-L95】 Deliverables: integration tests showing executed deals alter sampled orders versus fallback behavior.
- **Task P2.4 – Negotiation telemetry (codex-agent-neg-4)**: Instrument proposal generation, NBS components, and choose-phase outcomes for observability, validating baseline symmetry when K and α are large.【F:src/report_sentences.txt†L181-L189】 Deliverables: logging/metrics layer with sample dashboards or structured logs.

### Phase 3 – Higher-Order Theory of Mind (ToMᵏ)
- **Task P3.1 – Recursive BATNA engine (codex-agent-tom-1)**: Extend bargaining simulation to iterate over agent pairs, recursively simulate counterpart deals, and damp fallback updates for convergence.【F:src/report_sentences.txt†L102-L226】 Deliverables: recursion controller with convergence tests and profiling of iteration costs.
- **Task P3.2 – ToM gating & scoring (codex-agent-tom-2)**: Encode H^(k) gating and Score^(k) evaluation so proposals under an opponent's ToM^(k−1) fallback are discarded before NBS ranking.【F:src/report_sentences.txt†L200-L227】 Deliverables: scoring utilities with parameterized unit tests spanning varying opponent depths.
- **Task P3.3 – Depth configuration & guardrails (codex-agent-tom-3)**: Expose per-agent ToM depth k, support mixed populations, and implement safeguards against pessimistic equilibria highlighted in the thesis.【F:src/report_sentences.txt†L228-L238】 Deliverables: configuration schema, runtime validation, and scenario tests demonstrating heterogeneous k interactions.
- **Task P3.4 – Scenario reenactment harness (codex-agent-tom-4)**: Reproduce Figure 4 ToM² case study, capturing BATNA trajectories and verifying agents forego greedy moves to sustain acceptable deals.【F:src/report_sentences.txt†L239-L356】 Deliverables: scripted scenario with automated assertions and visualization-ready logs.

### Phase 4 – Relationship-Aware Extensions
- **Task P4.1 – Relationship tensor store (codex-agent-rel-1)**: Maintain directed Uᵢ→ⱼ scores initialized at 1 and persisted across phases within the state representation.【F:src/report_sentences.txt†L260-L266】 Deliverables: tensor data structure, serialization, and reset utilities.
- **Task P4.2 – Relationship updater (codex-agent-rel-2)**: Apply Eq. (16) after each phase using STAVE deltas and prior state values, exposing tunable λ sensitivity for experimentation.【F:src/report_sentences.txt†L267-L289】 Deliverables: update function with calibration tests sweeping λ.
- **Task P4.3 – Value augmentation pipeline (codex-agent-rel-3)**: Compute relationship-aware state value Vᵁ per Eq. (17) and inject it into proposal ranking, SBR evaluations, and telemetry outputs.【F:src/report_sentences.txt†L278-L285】 Deliverables: modified value adapters, integration tests, and logging of relationship contributions.
- **Task P4.4 – Behavioral scenario suite (codex-agent-rel-4)**: Simulate stalemate and alliance case studies (Figures 5–6) to confirm cooperative breakthroughs, reciprocity, and betrayal dynamics.【F:src/report_sentences.txt†L300-L370】 Deliverables: scripted simulations with acceptance criteria tied to relationship trends.

### Phase 5 – Evaluation & Experimentation
- **Task P5.1 – Metrics & analytics stack (codex-agent-exp-1)**: Capture deal acceptance rate, relationship trajectories, alliance longevity, and win probabilities as core evaluation metrics.【F:src/report_sentences.txt†L386-L399】 Deliverables: metrics module, storage schema, and sample analysis notebook.
- **Task P5.2 – Hyperparameter tooling (codex-agent-exp-2)**: Implement grid/Bayesian sweep utilities to explore λ, proposal batch size K, and ToM depth k.【F:src/report_sentences.txt†L389-L391】 Deliverables: experiment runner with configuration templates and example sweeps.
- **Task P5.3 – Tournament harness (codex-agent-exp-3)**: Run league-style evaluations across baseline, ToM-only, and relationship-aware agents to quantify incremental gains and detect instability when k is excessive.【F:src/report_sentences.txt†L236-L399】 Deliverables: tournament scripts, statistical summaries, and regression baselines.
- **Task P5.4 – Reporting pipeline (codex-agent-exp-4)**: Produce visualizations of relationship graphs, outcome statistics, and qualitative highlights matching thesis expectations.【F:src/report_sentences.txt†L373-L399】 Deliverables: reporting templates (e.g., notebooks or dashboards) and publication-ready figures.

### Phase 6 – Stretch Goals
- **Task P6.1 – Adaptive ToM depth (codex-agent-stretch-1)**: Prototype mechanisms that adjust k mid-game to avoid overestimating opponent sophistication.【F:src/report_sentences.txt†L236-L381】 Deliverables: adaptive policy module and experiments quantifying stability improvements.
- **Task P6.2 – Multi-party treaty generalization (codex-agent-stretch-2)**: Extend MBDS and ToM reasoning to support n-agent treaties aligned with future work directions.【F:src/report_sentences.txt†L391-L399】 Deliverables: generalized negotiation schema and proof-of-concept multi-party scenarios.
- **Task P6.3 – Human-in-the-loop interface (codex-agent-stretch-3)**: Design modular hooks for human communication channels to integrate language-enabled agents without disrupting the negotiation core. Deliverables: interface specification and prototype adapter.

## 4. Open Questions & Dependencies
- Availability of pretrained No-Press policy/value networks or requirements to substitute simplified evaluators until models are integrated.【F:src/report_sentences.txt†L65-L76】
- Selection of λ (relationship learning rate), K (proposal batch size), and damping factors that ensure numerical stability without extensive tuning; plan for experimentation harness early.【F:src/report_sentences.txt†L181-L289】【F:src/report_sentences.txt†L386-L391】
- Requirement for adjudicator parity with existing Diplomacy simulators if we later cross-check against external baselines.

## 5. Immediate Next Steps
1. Stand up the simplified board representation and unit adjudication tests (Phase 1).【F:src/report_sentences.txt†L34-L53】
2. Scaffold the MBDS pipeline with mock policy/value outputs to validate contract generation and NBS ranking before layering ToM/relationship logic.【F:src/report_sentences.txt†L116-L205】
3. Define data schemas for logging BATNA iterations and relationship updates to facilitate later analysis and visualization.【F:src/report_sentences.txt†L193-L399】
