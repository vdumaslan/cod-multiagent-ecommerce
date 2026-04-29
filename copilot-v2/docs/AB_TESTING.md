# A/B Testing: Version A vs Version B

This document describes the A/B testing infrastructure added to the multi-agent BI dashboard, including the design rationale, implementation, and how to interpret results.

---

## Overview

The spec requires: *"Conduct rigorous A/B testing in authentic workflow environments to measure productivity gains, user satisfaction, and overall system effectiveness."*

The test compares two versions of the decision-making experience:

| | Version A | Version B |
|---|---|---|
| **Description** | No AI — manual decisions | Full AI system |
| **Query form** | Plain text + category only | Full form with objective, horizon, constraints, suggested goals, match preview |
| **After submit** | Plain product list (retrieval only) | Pipeline → Advocate/Critic debate → Judge → ranked plans |
| **User task** | Assign action + price change per product manually | Review AI-ranked plans and choose one |
| **AI involvement** | None | Query rewrite, retrieval, sentiment, pricing, inventory agents, LLM debate |

**What we measure:**
- **Time to decision** — how long from results appearing to the user committing to a decision
- **Confidence** — self-reported 1–5 rating after each decision
- **Abandonment** — whether the user rejected all results without deciding

---

## Files

| File | Purpose |
|---|---|
| `app/ab.py` | Variant assignment and event logger |
| `app/api/app.py` | `/ab/variant/{owner_id}` and `/ab/event` endpoints |
| `app/api/schemas.py` | `ABEventRequest` schema |
| `app/ui/src/App.jsx` | Toggle, Version A view, timer, confidence prompt, event firing |

---

## Variant Assignment

Variants are assigned deterministically by hashing `owner_id`:

```python
digest = int(hashlib.md5(owner_id.encode()).hexdigest(), 16)
variant = ("A", "B")[digest % 2]
```

The same owner always gets the same variant. No state is stored — the hash is recomputed on each request.

In the current demo setup, `owner_id` is always `store_00`, so the assigned variant is fixed. For a real multi-user deployment, each store/user would get a consistent assignment.

The toggle in the UI header allows manual override for demos and evaluation — a user can flip between versions to experience both sides.

---

## Event Logging

Events are appended as JSONL to `runs/ab_events/events.jsonl`. Each line is one event:

```json
{"ts": "2026-04-29T21:45:00Z", "owner_id": "store_00", "variant": "B", "run_id": "run_20260429_214655_store_00", "event": "session_start", "metadata": {"mode": "B"}}
{"ts": "2026-04-29T21:47:12Z", "owner_id": "store_00", "variant": "B", "run_id": "run_20260429_214655_store_00", "event": "decision_made", "metadata": {"mode": "B", "time_to_decision_s": 132, "plan_id": "B07WSRSM6N"}}
{"ts": "2026-04-29T21:47:14Z", "owner_id": "store_00", "variant": "B", "run_id": "run_20260429_214655_store_00", "event": "confidence_rated", "metadata": {"rating": 4, "mode": "B"}}
```

### Event types

| Event | When fired | Key metadata |
|---|---|---|
| `session_start` | On app load, after variant assigned | `mode` |
| `decision_made` | On "Submit Decision" (A) or "Choose a plan" (B) | `time_to_decision_s`, `mode`, `decisions` or `plan_id` |
| `confidence_rated` | After confidence prompt answered | `rating` (1–5), `mode` |
| `abandoned` | On "Reject all" in Version B results without choosing | `time_to_decision_s`, `mode` |

---

## Timer

The timer measures time from when results are first shown to when the user makes a decision. It does **not** include time spent on the query form.

- **Version A**: starts when the product list renders
- **Version B**: starts when the debate view loads (after pipeline completes)

The timer is displayed live in the header (`⏱ 0:32`) so evaluators can see it ticking during a demo.

---

## Version A Flow

1. User sees a simple form: text area + category/subcategory + "Find Products"
2. On submit, calls `/retrieval/preview` using the category as the retrieval query (e.g. `"Canisters products"`)
3. Results show a plain list: product title, category, match score
4. For each product, user picks an action (Reprice / Restock / Promote / Hold / Investigate) and optionally a price change %
5. "Submit Decision" fires `decision_made` event, shows confidence prompt
6. After rating, resets to query form

No agents, no debate, no ranked plans, no rationale. The user has only the product name, category, and match score to go on.

---

## Version B Flow

1. User sees the full query form with all controls
2. On submit, runs the full pipeline: query rewrite → retrieval → enrichment (sentiment, pricing, inventory) → Advocate/Critic debate → Judge
3. Results show ranked plans with action type, pricing delta, sentiment, inventory status, LLM rationale bullets, and risk bullets
4. User can add context, run more debate rounds, or move on to Judge
5. "Choose a plan" fires `decision_made` event, shows confidence prompt
6. After rating, saves and resets

---

## Metrics and Analysis

After collecting events, compute per-variant:

```
acceptance_rate  = decision_made events / session_start events
mean_time        = mean(time_to_decision_s) for decision_made events
mean_confidence  = mean(rating) for confidence_rated events
abandonment_rate = abandoned events / session_start events
```

Example comparison table:

| Metric | Version A | Version B |
|---|---|---|
| Acceptance rate | ? | ? |
| Mean time to decision (s) | ? | ? |
| Mean confidence rating | ? | ? |
| Abandonment rate | ? | ? |

Version B should show higher confidence and lower abandonment if the AI assistance is effective. Time to decision may be longer in Version B due to the debate step, but confidence should compensate.

---

## Query Tips for Testing

The system works best with specific product-oriented goals. Strategy-heavy goals often trigger clarifying questions from the query rewrite LLM.

**Works well:**
- `increase revenue for beauty canisters`
- `reduce returns for kitchen organizers`
- `boost sales for bathroom storage`

**Triggers clarifying questions:**
- `increase revenue` (too vague)
- `Grow revenue for All Beauty while controlling inventory risk and returns` (too strategy-heavy)

For Version B demos, pick a subcategory first, then use a short goal with a concrete product term.
