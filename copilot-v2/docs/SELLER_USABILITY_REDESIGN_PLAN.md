# Seller-Usable Results Redesign Plan (Copilot v2)

## Context
The current results UI is **technically correct** (it reflects retrieval/enrichment/debate outputs) but is often **not seller-usable** because it surfaces raw model signals instead of guiding a decision. This document proposes a seller-facing redesign grounded in **what the system actually produces today** (runtime fields, cached artifacts, and constraints).

### Goal
Transform the presentation from:

- **signals → agent output → field dump**

to:

- **situation → decision → explanation → risk / constraints → evidence (optional)**

## Observed Issues (Current UI)

### 1) Duplicate product text (Action title vs Retrieval snippet)
- The “action title” repeats the product document/title, and the “retrieval snippet” also shows the same product document fields.
- Because retrieval evidence is derived from the product document, rendering both blocks by default causes repetition and cognitive overload.

### 2) No real decision signal (top actions collapse to “reprice”)
- Seeing Rank #1/#2/#3 all as “reprice” does not answer: **what exactly should I do, and why this product first?**

### 3) Price deltas lack business meaning
- Showing `+6.08%` without clear direction/intent (raise vs lower) and without constraint framing or expected effect makes the recommendation hard to trust.

### 4) Raw features shown instead of conclusions
- The UI exposes internal fields (e.g., `p_neg`, `retrieval_score`, cache source) that are meaningful for debugging but not for seller decisions.

### 5) Contradictory signals aren’t resolved
- Example pattern: “inventory risk” appears alongside an action that would increase demand, without an explicit conflict resolution message.

### 6) “Suggested (playbook)” vs “Final (judge)” inconsistency breaks trust
- When baseline and final disagree, sellers interpret it as a bug unless the override is explained.

### 7) No portfolio-level summary
- Sellers need a top-level “what should I do overall?” view (counts, priority, risk), not only per-product cards.

### 8) UI overload
- Long product names and large blocks of repeated detail produce high cognitive load.

## Redesign Principles (Seller-Facing)

### A) Decision-first, evidence-second
Each product card should lead with:
- **Action**
- **Why** (plain language, 2–3 bullets)
- **Risk / constraints**
- **Confidence**

Evidence and raw signals should be **collapsed** by default.

### B) Convert numeric signals to plain-language claims
Replace raw values (e.g., `p_neg=0.12`) with interpretations (e.g., “Low negative feedback → safer to increase price”).

### C) Resolve conflicts before display
If signals conflict, the UI should show the resolved policy outcome (e.g., “Inventory low → avoid decreasing price; HOLD recommended”).

### D) Be honest about impact (dataset limits)
We do not have COGS/margins; therefore, “profit ↑” cannot be estimated directly. Use **proxy impact indicators** (risk, confidence) based on available evidence.

## Implementation-Feasible Plan (Grounded in Current Fields)

This plan assumes the current per-action structure includes (as produced by `copilot-v2/app/pipeline.py` and related agents):
- `product_id`
- `action_type` (e.g., `reprice`, `hold`, `investigate`, `promote`, `restock`)
- `recommended_price_change_pct`
- `evidence.retrieval_score` (+ optional retrieval snippet text)
- `pricing.source`, `pricing.price_missing`, `pricing.near_bound`, `pricing.large_delta`
- `sentiment.n_reviews`, `sentiment.p_neg` (and p_pos/p_neu)
- `inventory.stock_status`, `inventory.risk_flag`
- `signals.total_returns` (and other basic numeric signals)
- `llm_rationale_bullets`, `llm_risk_bullets` (currently deterministic/grounded where possible)

### 1) Remove duplication: separate “Product header” from “Retrieval evidence”
**Default card header**
- Product short name (not full retrieval document)
- Action (explicit, see section 2)
- Priority + confidence

**Collapsed “Evidence” section**
- Retrieval snippet: 1–2 lines + “Retrieval similarity: XX%”
- Raw fields (optional): SKU/product_id, cache source, raw sentiment numbers, etc.

### 2) Make the action explicit (direction + magnitude)
Rendering rule using `action_type` and `recommended_price_change_pct`:
- If `action_type == "reprice"`:
  - pct > 0 → **Increase price by X%**
  - pct < 0 → **Decrease price by X%**
  - pct == 0 → **Hold price (0%)**
- If `action_type == "investigate"` → **Investigate before changing price**
- If `action_type == "hold"` → **Hold**
- If `action_type == "restock"` → **Restock**
- If `action_type == "promote"` → **Promote / increase visibility**

### 3) Generate “Why” bullets in plain language (2–3 max)
Derive from current evidence:
- **Relevance**: `evidence.retrieval_score`
  - “High match to your goal (retrieval similarity ~80%).”
- **Sentiment**: `sentiment.p_neg`, `sentiment.n_reviews`
  - `n_reviews == 0`: “Low review coverage; treat sentiment as low-confidence.”
  - `p_neg >= 0.30`: “High negative feedback; avoid aggressive price increases.”
  - else: “Low negative feedback; safer to adjust price.”
- **Inventory**: `inventory.stock_status`, `inventory.risk_flag`
  - “Inventory risk (low stock / stockout risk); avoid demand-increasing actions.”
- **Returns**: `signals.total_returns`
  - “Elevated returns; prioritize investigation/quality checks.”
- **Pricing reliability**: `pricing.source`, `pricing.near_bound`, `pricing.large_delta`, `pricing.price_missing`
  - “Pricing unavailable (missing price in cache).”
  - “Suggested delta near policy bound; treat as lower-trust.”

Note: we already generate deterministic rationale bullets in the pipeline; the primary change is to **show fewer bullets** and **translate them into seller language**.

### 4) Conflict resolution (policy-first explanation)
Before displaying the action, apply deterministic conflict messaging:
- Inventory risk (`low_stock`/`stockout_risk`/`risk_flag=true`) should explicitly constrain repricing direction:
  - “Inventory low → avoid decreasing price; HOLD or small increase only.”
- Reduce-returns objective + high negativity (`do_not_raise_if_p_neg_above`) should constrain increases:
  - “Negative feedback high → price increases blocked; recommend HOLD/investigate.”

### 5) Fix baseline vs judge trust gap
If “Suggested (playbook)” differs from “Final (judge)”:
- Show only **Final recommendation** by default.
- Add one line when overridden:
  - “Overrode baseline due to: inventory risk / returns risk / sentiment constraint.”

### 6) Add portfolio-level summary (top of page)
Key design principle: **separate portfolio summary from top recommendations**.

- **Portfolio Summary** should describe the *overall situation* and must be computed on a broader, less biased set than the top-3 cards.
- **Top Recommendations** (cards) should remain focused and actionable (e.g., top-3 final ranked actions).

#### Recommended scope (feasible now): top-K enriched candidates (K = 20–50)
Instead of computing the summary from the displayed top-3 actions, compute it from the **top-K retrieval/enriched candidates** (before final ranking/judging). This is feasible because:
- Retrieval already returns top-k candidates (FAISS search).
- Enrichment is deterministic and uses cached signals (pricing/sentiment/inventory caches), so expanding from 3 to 20–50 adds modest overhead.

Compute summary statistics on the top-K enriched set:
- **Action distribution (proxy)**: use `suggested_action` (playbook) plus constraint guardrails (inventory/sentiment) to estimate hold/investigate/reprice/promote/restock rates.
- **Risk signals**: % high negative sentiment, % inventory risk, % high returns, % pricing missing.
- **Strategy sentence** (rule-based template aligned to objective):
  - Revenue/profit: “Most products are stable; focus on selective price increases on healthy inventory SKUs and investigate high-return items.”
  - Avoid stockouts: “Inventory risk is present; avoid demand-increasing actions and prioritize HOLD/restock candidates.”
  - Reduce returns: “Quality risk dominates; prioritize investigation and avoid price increases when negative feedback is elevated.”

#### Option (even better when available): seller-defined portfolio summary
If the seller can provide a portfolio list (e.g., category/subcategory selection, uploaded SKU list, or account inventory), compute the portfolio summary over **all portfolio products** rather than top-K retrieved candidates.

### 7) “Expected impact” as proxies (honest given missing margins)
Because margins/COGS are missing, provide proxy impact chips:
- **Demand risk** (inventory risk + sentiment)
- **Customer risk** (returns + negativity)
- **Confidence** (retrieval score + review coverage + pricing availability)

Avoid presenting dollar profit estimates.

### 8) Progressive disclosure to reduce cognitive load
Default view: Action + 2–3 why bullets + risk/confidence.
Expandable: evidence + raw numbers + debug.

## Evidence: Why this is consistent with our dataset/artifacts
- Retrieval evidence is the product document; showing it twice causes duplication.
- Pricing/sentiment are cached and surfaced as summary signals; sellers should see interpretations, not raw feature values.
- Inventory cache is derived from synthetic upstream tables, so distributions can become low-variance; the UI must compensate by emphasizing constraints and uncertainty.
- We lack cost/margin; “profit impact” must be framed as proxy-based rather than direct optimization.

## Suggested Execution Order (Highest ROI First)
1. **Decision-first card header + explicit action rendering** (remove duplication)
2. **Portfolio summary** based on top-K (20–50) enriched candidates (not top-3)
3. **Conflict resolution messaging** (policy constraints made explicit)
4. **Plain-language “Why” bullets with caps** (2–3)
5. **Baseline vs judge override explanation**
6. **Progressive disclosure + debug accordion**

