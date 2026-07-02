# AI Monte Carlo Testing Scope

## Purpose

This folder is for testing and documenting the AI-assisted scenario generation experiment outside the production app.

No app code, backend code, frontend code, prompts inside the app, or planner logic should be changed as part of this work. Everything in this testing effort stays inside:

```text
C:\Users\abqua\Desktop\HRI app\ai-montecarlo tetsing
```

## System Context

The planning app has an AI-assisted scenario generation flow.

The app sends Gemini a structured context that includes:

- user business intent
- selected goal
- forecast periods
- default discount ladders
- slab structure for 12-ML and 18-ML
- planner context
- model coefficients and elasticity information
- price and margin profile
- user constraints

Gemini does not directly create the final scenario set.

Gemini creates exactly 3 scenario families. Each family contains parameters such as:

- family name
- priority_weight
- base_min
- base_max
- gap_min
- gap_max
- month_pattern
- month_shift_strength
- size_bias_12
- size_bias_18
- volatility
- anchor_weights

After this, the backend uses those families to generate many Monte Carlo-style discount scenarios. Those final scenarios are evaluated later on revenue, profit, volume, investment, and CTS.

## Testing Stages

### Stage 1: Gemini Intent Understanding and Family Creation

Current stage: active.

This stage validates only the first layer:

```text
User intent
-> Gemini interpretation
-> Gemini-created scenario family parameters
```

The goal is to check whether Gemini converts business intent into directionally appropriate family parameters.

This stage does not test:

- Monte Carlo sampling quality
- final scenario quality
- revenue optimization
- profit optimization
- volume optimization
- investment or CTS performance
- backend repair or enforcement logic
- frontend behavior
- app code correctness

Stage 1 asks:

Does Gemini understand the business intent and create scenario families that are directionally aligned with it?

Examples:

- If the intent is 12-ML growth, family parameters should bias movement toward 12-ML.
- If the intent is 18-ML growth, family parameters should bias movement toward 18-ML.
- If the intent is margin protection, families should be more conservative and avoid deep discounting.
- If the intent is volume growth, families can be more aggressive and responsive.
- If the intent is low disruption, families should use narrow ranges, lower volatility, and stable month patterns.
- If the intent includes hard constraints, families should not directionally contradict those constraints.

### Stage 2: Family-to-Monte-Carlo Sampling Quality

Future stage.

This stage will validate whether the backend's Monte Carlo-style sampling from Gemini-created families produces diverse, valid, non-degenerate discount scenarios.

This stage is not active yet.

### Stage 3: Business Metric Evaluation Quality

Future stage.

This stage will validate the final evaluated scenario outputs across business metrics:

- revenue
- profit
- volume
- investment
- CTS

This stage is not active yet.

### Stage 4: End-to-End Scenario Recommendation Quality

Future stage.

This stage will evaluate whether the complete flow produces scenarios that are useful for planner decision-making.

This stage is not active yet.

## Stage 1 Test Design

For Stage 1, each test case should contain:

- test case ID
- user intent category
- exact user prompt or business ask
- selected goal
- relevant constraints, if any
- Gemini output families
- expected directional behavior
- reviewer judgment
- notes

Recommended judgment labels:

- aligned
- partially aligned
- not aligned
- unclear

## Stage 1 Evaluation Criteria

Review Gemini family parameters against the intent.

For 12-ML growth:

- size_bias_12 should generally be stronger than size_bias_18.
- family names and ranges should indicate 12-ML movement.
- the family should not rely mainly on 18-ML movement.

For 18-ML growth:

- size_bias_18 should generally be stronger than size_bias_12.
- family names and ranges should indicate 18-ML movement.

For margin or profit protection:

- base_min and base_max should stay moderate.
- volatility should be low to moderate.
- month_pattern should avoid aggressive upward escalation unless explicitly requested.
- family names should reflect conservative or margin-safe behavior.

For volume growth:

- base ranges can be wider or higher.
- volatility can be moderate.
- month_pattern can be up, pulse, or wave depending on the ask.

For low disruption:

- base_min and base_max should be close together.
- gap_min and gap_max should be narrow.
- volatility should be low.
- month_pattern should usually be flat or gently varying.

For constraint-driven planning:

- family direction should not contradict hard slab, pack, or discount constraints.
- constraints should influence the family style even though backend enforcement happens later.

## Out of Scope for Current Stage

Do not use Stage 1 to judge:

- whether sampled scenarios are good
- whether final scenarios beat baseline
- whether revenue/profit improves
- whether Monte Carlo generated enough diversity
- whether backend repaired invalid values correctly
- whether the app UI presents results correctly
- whether the implementation should be changed

## Current Status

We are starting with Stage 1: Gemini intent understanding and family creation.

The first deliverable is a clear test record showing whether Gemini-created families are directionally aligned with the user intent before any Monte Carlo or business metric evaluation is considered.
