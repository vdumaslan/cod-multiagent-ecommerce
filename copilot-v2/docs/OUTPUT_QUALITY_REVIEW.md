# Output Quality Review (Root Causes + Dataset Evidence)

This note documents why the current system’s outputs often feel generic / impractical for a small-business owner, based on direct inspection of:

- runtime code in `copilot-v2/app/`
- UI code in `copilot-v2/app/ui/`
- artifacts under `copilot-v2/artifacts/` for snapshot `38710839ca6e1009`

It also includes concrete fixes and what to pass to retrieval vs debate LLM agents to materially improve usefulness.

---

## Why the output quality is poor (root causes in this codebase)

### 1) Inventory risk signal is effectively non-informative

- The inventory cache produced by `src/copilot_v2/scripts/precompute/build_inventory_cache.py` can become degenerate (nearly all items in the same class) if the rule thresholds do not match the data distribution.
- In our local artifacts, the generated cache classified **all 50,000** products as `healthy` with `risk_flag=false` for every row (details in the dataset section below).
- Result: the debate LLMs almost never see meaningful inventory tension (stockout risk vs overstock), so plans skew toward generic repricing.

**Where this happens**
- Cache build rules: `copilot-v2/src/copilot_v2/scripts/precompute/build_inventory_cache.py`
- Runtime consumption: `copilot-v2/app/agents/inventory_agent.py` + enrichment in `copilot-v2/app/pipeline.py`

**Solutions**
- Before assuming the entire cache is “broken”, verify whether the **`stockout_risk`** rule can ever fire (cases where `on_hand_units <= safety_stock_units`). If that count is near-zero too, the current ruleset is effectively unusable for decision-making.
- Replace fixed thresholds with dataset-aware rules (percentile-based thresholds) or days-of-cover (if you can derive units sold/day).
- Add a QA check in precompute: if distribution is >95% one class, mark cache “unreliable” and/or auto-tune thresholds.

---

### 2) Pricing cache is heavily saturated at the policy bounds (±10%)

- The pricing cache (`predicted_price_change_pct`) is concentrated near ±10%, meaning the system frequently recommends extreme repricing.
- This pushes the Advocate to repeat “+10%” / “-10%” behavior because it is literally being fed those deltas as the primary “action”.

**Where this happens**
- Runtime enrichment uses pricing cache directly: `copilot-v2/app/pipeline.py` (`recommended_price_change_pct`)
- Pricing cache used by: `copilot-v2/app/agents/pricing_agent.py`

**Solutions**
- Avoid linear shrinkage alone (it can just move the pileup from ±10% to a new “effective bound”).
- Prefer a **soft-cap** transform at cache-build time to eliminate hard clipping and reduce boundary pileups, e.g.:
  - \( \mathrm{softcap}(x) = b \cdot \tanh(x / b) \) where \(b = \text{policy\_bound}\)
- Include a “near_bound=true/false” flag in the LLM payload so debate agents treat bound-hit outputs skeptically.
- Add richer business signals so pricing isn’t the only lever (see “payload contracts” section).

---

### 3) Debate agents are grounded on too little business context

The debate payload is intentionally slimmed, but what it currently contains is insufficient for practical decisions:

- `product_id`
- pricing delta (often bound-saturated)
- sentiment probabilities
- inventory status (currently degenerate)
- retrieval score + a short doc excerpt

Missing inputs that are commonly required for actionable plans:

- current `price` (and missingness markers)
- category/subcategory constraints
- relative price position (e.g., price percentile in subcategory)
- review recency / “what changed recently”
- explicit objective function (profit vs revenue vs liquidation vs reduce returns)
- operational constraints (max # SKUs, don’t raise price when \(p_{neg}\) high, etc.)

**Where this happens**
- Slimming is done in prompts: `copilot-v2/app/agents/orchestrator/advocate.py` and `critic.py`
- Candidate enrichment is currently minimal and sets `margin_pct=0.0`: `copilot-v2/app/pipeline.py`

**Solutions**
- Pass structured, decision-ready features from `products.parquet` and `product_signals.parquet` to debate agents (see “What to pass” section).
- Introduce an explicit objective + constraints schema in the API/UI, not only `max_abs_price_change_pct`.

**Hard blocker: margin / COGS**
- `signals.margin_pct` is currently hardcoded to `0.0` in `copilot-v2/app/pipeline.py`, but this is not merely a wiring gap: the snapshot artifacts do not include cost/COGS fields.
- As a result, the system cannot compute true profit or margin impact from repricing without adding a new data source (or an assumption model) for costs.

---

### 4) Judge prompt embeds a Python dict string, not JSON (serialization bug)

In `copilot-v2/app/agents/orchestrator/judge.py` the Judge is given:

- `INPUT_JSON:\n{payload}`

where `payload` is a Python dict, not `json.dumps(payload)`.
This increases the likelihood of narrative / invalid outputs and increases reliance on retry/fallback behavior.

**Where this happens**
- `copilot-v2/app/agents/orchestrator/judge.py` (`build_messages`)

**Solutions**
- Serialize the payload via `json.dumps(payload)` (matching Advocate/Critic behavior).
- Add a unit test that prompts are valid JSON strings where expected.

**Observed run evidence (saved artifacts)**
- Example run: `copilot-v2/artifacts/runs/38710839ca6e1009/run_20260426_061542_store_00_judge/8_debate_judge.json`
  - The judge’s first response was non-JSON (narrative) and failed parsing (`judge_parse_error`), requiring a retry that produced valid JSON.
  - The narrative response referenced product IDs that were not in the validated judge output, a practical example of why the judge prompt/payload must be corrected and why retry/fallback rates should be monitored.

---

### 5) UI scoring/labels can make outputs feel “made up”

In `copilot-v2/app/ui/src/App.jsx`:

- “Confidence” is derived mainly from `retrieval_score * 100` (clamped), not a calibrated confidence.
- “Impact score” is a transform of confidence.
- “Risk” is driven largely by sentiment because inventory is degenerate.

Even if backend improves, presenting these as “Impact/Confidence” can erode trust.

**Where this happens**
- `copilot-v2/app/ui/src/App.jsx` (`buildPlansFromRanked`)
- display: `copilot-v2/app/ui/src/components/ResultsView.jsx`

**Solutions**
- Rename confidence to what it is (e.g., “Retrieval similarity”).
- Compute risk from meaningful, validated signals (inventory + returns + sentiment) once inventory is fixed.
- Add an “evidence panel” (why selected, what signals drove the decision).
- Surface the **retrieval snippet** that already exists at `evidence.points[0].text` (computed in `copilot-v2/app/pipeline.py` but not shown in the UI). This is the easiest “why this SKU?” improvement.
- Guard the “mock plans” fallback: the UI currently falls back to hardcoded mock plans if the real results array is empty. Those mock plans can mask pipeline failures during demos. Prefer an explicit “No results / API degraded” state over showing mock data without warning.

---

## What your actual dataset says (and what’s hurting output quality)

All findings below are from `copilot-v2/artifacts/` for snapshot `38710839ca6e1009`.

### Retrieval artifacts

**Index meta**
- Location: `copilot-v2/artifacts/indexes/38710839ca6e1009/dense/intfloat_e5-large-v2/index_meta.json`
- Encoder: `intfloat/e5-large-v2`, `max_seq_length=384`, prefixes enabled.
- Corpus size: **50,000** documents.

**Corpus table**
- `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/retrieval_corpus.parquet`
- Columns: `product_id`, `product_document`, `category`, `subcategory`, `price`, `avg_rating`

**Data issues impacting usefulness**
- `price` is missing for **~29.5%** of products (so price-based reasoning often collapses).
- Some `product_document` rows are placeholders like “PRODUCT NOT AVAILABLE”, which can pollute candidate sets for broad queries.
- Retrieval score separation for generic goals can be weak (scores cluster tightly), so downstream agents see “many similar candidates”.

**Solutions**
- Pre-filter bad docs (placeholder, too short).
- Add structured tags into retrieval docs (price buckets, rating buckets, recency buckets) so retrieval is more discriminative.
- Allow query-time filters (category/price bands) to reduce ambiguity.
- Add a minimum retrieval-score threshold in runtime (see Fix F below) so low-confidence candidates do not flow into debate.

**Observed run evidence (saved artifacts)**
- Example run: `copilot-v2/artifacts/runs/38710839ca6e1009/run_20260426_064309_store_00/1_retrieval.json`
  - Top-10 retrieval scores ranged from **0.75966 → 0.76749** (spread **0.00783**), indicating weak separation for broad goals.

---

### Products and signals (high-value structured features currently underused)

**Products**
- `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/products.parquet` (50,000 rows, 16 cols)
- Includes `title`, `brand`, `category`, `subcategory`, `description`, `price`, `avg_rating`, `rating_count`, `product_document`, review aggregates.

**Product signals**
- `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/product_signals.parquet` (50,000 rows, 15 cols)
- Includes `price_percentile_in_subcategory`, `subcategory_mean_price`, `rating_vs_subcategory_mean`, `days_since_last_review`, etc.

**Manifest / schema note (avoid confusion)**
- The “main” snapshot manifest `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/manifest.json` may not reflect the enriched/in-place derived columns.
- The authoritative post-enrichment column list is in `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/manifest_derived_enrich.json` (evidence that `enrich_snapshot_derived.py` was run with `--write-inplace`).

**Missing price nuance**
- `price` is missing for **~29.5%** of the catalog.
- For those products, derived signals like `price_percentile_in_subcategory` and `subcategory_mean_price` exist as columns but are often **NaN**.
- The debate payload should include a boolean `price_missing` (or `price_present`) so LLMs do not fabricate price-based rationale when price is unknown.

**Problem**
- These structured columns exist and are decision-relevant, but are not carried through the runtime enrichment → debate payload in a structured way.

**Solutions**
- Merge `product_signals.parquet` features into the enriched candidate payload for debate agents.

---

### Sentiment cache

- Cache path: `copilot-v2/artifacts/caches/38710839ca6e1009/sentiment/sentiment_cache.parquet`
- Coverage: 50,000 rows.

**Observations**
- `n_reviews` is relatively narrow (often near 15–24), which makes “confidence” from sentiment alone questionable.
- Only a small minority of items have very high negative probability; sentiment alone will not differentiate most candidates.

**Solutions**
- Add review recency (last 30/90 days) and theme extraction from negative reviews (even simple keywords) for explainability.
- Pass “missingness” signals: if `n_reviews` is low, treat sentiment as low-confidence.

---

### Inventory cache (degenerate distribution)

- Cache path: `copilot-v2/artifacts/caches/38710839ca6e1009/inventory/inventory_cache.parquet`
- Observed distribution: **100%** `healthy` (50,000 rows), **0** `risk_flag=true`.

**Why**
- The classifier in `build_inventory_cache.py` uses fixed thresholds that did not trigger on this dataset:
  - `available_to_sell` never <= 5
  - `mean_daily_revenue` never <= 1
  - `total_returns` never >= 10

**Solutions**
- Replace thresholds with percentile-based or dataset-derived thresholds.
- Add a precompute QA check that fails loudly or auto-tunes thresholds if distribution collapses.
- As part of debugging, explicitly compute how many rows satisfy `on_hand_units <= safety_stock_units` (the only way `stockout_risk` can fire under the current rule set). If this is ~0, inventory will never influence decisions.

---

### Pricing cache (bound saturation)

- Cache path: `copilot-v2/artifacts/caches/38710839ca6e1009/pricing/pricing_cache.parquet`
- Rows: 35,259 (not full 50,000 coverage; missing products fall back to 0.0 in runtime).

**Observed**
- Large fraction of predictions are near ±10% (policy bound), making recommendations feel unrealistic.

**Solutions**
- Replace hard clipping with a smooth soft-cap transform (e.g., tanh) to avoid boundary pileups.
- Add a calibration layer and include confidence/uncertainty or “near_bound” flags.
- Improve coverage (why ~14,741 products are missing from the pricing cache should be diagnosed).

---

## What should be passed to retrieval vs to chain-of-debate LLM agents

### Retrieval (e5-large-v2)

**Goal**
- Find relevant candidate products for the user’s goal with good separation and controllable filters.

**Recommended retrieval document fields**
- Stable identity + catalog descriptors: `title`, `brand`, `category`, `subcategory`
- A short description snippet (truncated)
- Structured tags appended as text:
  - `price_bucket`, `rating_bucket`, `rating_count_bucket`
  - `review_recency_bucket` (derived from `days_since_last_review`)
- Avoid placeholder docs (e.g., “PRODUCT NOT AVAILABLE”).

**Recommended query inputs**
- Owner goal text
- optional filters (category, price band, exclude low reviews) that reduce ambiguity.

---

### Debate LLMs (Advocate/Critic/Judge)

**Goal**
- Choose and justify a *small number of actions* grounded in measurable business signals, under explicit constraints/objectives.

**Recommended per-candidate feature bundle**
- Identity: `product_id`, `title`, `brand`, `category`, `subcategory`
- Pricing context:
  - `price` (+ `price_missing`)
  - `price_percentile_in_subcategory`, `subcategory_mean_price`, `subcategory_median_price`
  - pricing model output + flags: `predicted_delta`, `near_bound`, `pricing_cache_found`
- Review/sentiment context:
  - `n_reviews`, `p_pos`, `p_neu`, `p_neg`
  - `days_since_last_review`, `recent_review_ratio_90d`
  - (optional) extracted negative themes (top 3)
- Inventory/ops:
  - `available_to_sell`, `on_hand_units`, `safety_stock_units`
  - `mean_daily_revenue`, `total_returns`
  - `stock_status` and a meaningful `risk_flag`
- Evidence:
  - `retrieval_score` + 1 short snippet (not the full doc blob)

**Action space note (important for usefulness)**
- The output validator already accepts `hold`, `promote`, `investigate`, `restock` (see `copilot-v2/app/llm.py`), so no schema changes are required.
- However, the runtime pipeline currently hardcodes `action_type="reprice"` for every enriched candidate (see `_enrich()` in `copilot-v2/app/pipeline.py`), and the Advocate’s few-shot example only demonstrates `reprice`. Together, these strongly bias the debate toward repricing-only outputs.
- Recommended fix: add a lightweight playbook field during enrichment (e.g., `suggested_action`) derived from signals and pass it to Advocate/Critic/Judge. This is effectively blocked until inventory classification is non-degenerate (otherwise rules like low_stock → hold never trigger).

**Recommended shared context**
- Objective: profit / revenue / liquidation / reduce returns
- Constraints: max SKUs changed, max abs price delta, no increases when \(p_{neg}\) high, etc.

---

## Recommended next steps (non-exhaustive)

1) Fix Judge prompt serialization (`judge.py` should JSON-serialize payload).
2) Fix inventory cache rule thresholds (or make them dataset-aware) and add QA checks.
3) Calibrate pricing cache outputs (and diagnose coverage gaps).
4) Enrich debate payload with structured features from `products.parquet` and `product_signals.parquet`.
5) Update UI to present metrics honestly (rename “confidence”, add evidence, avoid fake-looking scores).

**Additional fixes (no cache rebuild required)**
- F) **Retrieval minimum-score threshold**: filter out candidates below a configured minimum similarity score. If no candidates remain, return “no strong matches” instead of running debate on weak matches.
- G) **Query reformulation before retrieval**: rewrite broad business goals into product-oriented retrieval queries (rule-based first; optional small LLM call later). This should run between UI submission and `retrieval_agent.retrieve()` in `pipeline.py`.

---

## Implementation status (this repo / local run)

This section tracks what has been implemented so far, and what we observed when re-running the system after each change.

### Implemented

- **A — Judge payload JSON serialization**
  - Status: implemented in `copilot-v2/app/agents/orchestrator/judge.py` (Judge now receives `json.dumps(payload)`).
  - Observed: judge still sometimes emits narrative first and requires a retry; however, the payload correctness bug is fixed.

- **F — Retrieval minimum-score threshold**
  - Status: implemented in `copilot-v2/app/agents/retrieval_agent.py` + wired via `COPILOT_RETRIEVAL_MIN_SCORE` in `copilot-v2/app/pipeline.py`.
  - Observed: with `COPILOT_RETRIEVAL_MIN_SCORE=0.80`, broad goals (e.g., “how to increase revenue”) can legitimately return **0** candidates (preferred vs debating weak matches).

- **G — Query reformulation before retrieval**
  - Status: implemented in `copilot-v2/app/pipeline.py` (writes `0_query_rewrite.json` per run and exposes `trace.query_rewrite`).
  - Observed:
    - For vague goals (e.g., “how to increase revenue”), the rewriter returns a **clarifying question** and an empty `retrieval_query`, allowing retrieval to correctly return “no strong matches”.
    - For semi-specific goals (e.g., “increase revenue for bed risers”), the rewriter produces a concrete `retrieval_query` and retrieval scores can clear the min-score threshold (e.g. top score ~0.85).

- **B — Inventory thresholds (dataset-aware)**
  - Status: implemented in `copilot-v2/src/copilot_v2/scripts/precompute/build_inventory_cache.py` and inventory cache rebuilt.
  - Observed:
    - Inventory cache distribution is no longer degenerate (example: ~44k `healthy`, ~5k `low_stock`, ~630 `overstocked`).
    - `risk_flag` now fires for thousands of products, and pipeline candidates can surface `low_stock`/risk items in `enriched_candidates`.
    - `stockout_risk` remains non-firing for this dataset because `on_hand_units <= safety_stock_units` never occurs (share ~0).

- **C — Pricing soft-cap (tanh)**
  - Status: implemented in `copilot-v2/src/copilot_v2/scripts/precompute/build_pricing_cache.py` and pricing cache rebuilt.
  - Observed:
    - Previously, >80% of predictions were within \(|pct|\ge 9.5\%\) (hard-bound pileup).
    - After soft-cap, predictions no longer pile up at ±10% and the observed range compressed smoothly (example: min ≈ -7.76, max ≈ 7.69 on the rebuilt cache).
    - Remaining issue: predictions are still somewhat “peaky” (e.g., many items clustering around ~7–8%). This is now a model/calibration characteristic (not a clipping artifact). Track it as a follow-up:
      - add and pass a `near_bound`-style flag for “large magnitude” deltas (relative to policy bound), and/or
      - add a calibration/shrinkage layer based on additional evidence signals.

- **D — Action space / playbook wiring**
  - Status: implemented (runtime only; no cache rebuild).
  - What changed:
    - `copilot-v2/app/pipeline.py`: adds `suggested_action` per candidate (simple playbook derived from inventory + returns + negative sentiment) and includes it in enriched candidates.
    - `copilot-v2/app/agents/orchestrator/advocate.py`, `critic.py`, `judge.py`: prompts now explicitly allow `action_type` in `{reprice, hold, promote, investigate, restock}` and include `suggested_action` in the candidate payload.
  - Observed (spot-check run):
    - Pipeline produced non-trivial `suggested_action` distribution (example run: 9× `reprice`, 1× `hold`).
    - Advocate proposed mixed actions (example included `hold` alongside `reprice`).
    - Judge produced at least one non-`reprice` action (`investigate`) with `judge_fallback=false`.
  - Notes:
    - This is intentionally “lightweight” (heuristics), but it breaks the previous degeneracy where the system could only ever recommend `reprice`.

- **E — UI: evidence-first rendering + honest labels**
  - Status: implemented (UI only).
  - What changed:
    - `copilot-v2/app/ui/src/App.jsx`: carries through `suggested_action`, `finalActionType`, `evidenceSnippet`, and `retrievalSimilarity` from `ranked_actions` into the UI “plan” model.
    - `copilot-v2/app/ui/src/components/ResultsView.jsx`: displays (a) suggested vs final action, (b) retrieval snippet, and relabels “Confidence” → “Retrieval similarity”.
    - `copilot-v2/app/api/schemas.py`: adds `suggested_action` to `RankedAction` response model so the UI reliably receives it.
  - Observed:
    - `ranked_actions[*].suggested_action` now appears in `/debate/judge` responses and renders in the Results UI alongside the Judge’s final `action_type`.

- **E2 — Inventory-safety guardrail (Judge alignment)**
  - Status: implemented (backend).
  - Why: after Step D/E, we could see cases where `suggested_action=hold` for `low_stock` items but the Judge still returned `reprice` (often `0%`), which conflicts with the business goal “avoid stockouts”.
  - What changed:
    - `copilot-v2/app/agents/orchestrator/judge.py`: adds a deterministic post-processing guard:
      - if `inventory.stock_status` is `low_stock` or `stockout_risk` (or `risk_flag=true`), force `action_type` to `hold` (or `restock` for `stockout_risk`) and set `recommended_price_change_pct=0.0`.
  - Verified:
    - For goal “Avoid stockouts…” the `/debate/judge` output now returns `final_action=hold` for low-stock candidates (no fallback).

- **UI redesign (Step 1: query page)**
  - Status: implemented.
  - Summary: integrated category/subcategory scoping and a live retrieval match preview into the query UX, so users can form retrieval-friendly goals before running debate.

**Recommended implementation order (to maximize demo usefulness)**
- A) Judge payload JSON bug → quick reliability win
- F) Retrieval score minimum threshold → honest “no match” behavior
- G) Query reformulation → improves candidate quality before debate
- B) Inventory thresholds → rebuild cache → unlock meaningful risk/action playbooks
- C) Pricing soft-cap (tanh) → rebuild cache → reduce extreme “±10% everywhere” behavior
- D) Action space/playbook wiring in enrichment + prompts (now meaningful)
- E) UI: evidence-first rendering, explicit retrieval snippet, remove/guard silent mock-plan fallback (implemented)

**UI inputs (API already supports; UI just needs wiring)**
- `horizon_days` (e.g., 7/14/30 selector)
- `top_n_actions`
- `constraints.max_abs_price_change_pct`
- Keep `owner_id` defaulted/hidden unless multi-owner is real.

**UI inputs that need small backend additions**
- Objective selector (profit / revenue / clear inventory / reduce returns / avoid stockouts): implemented end-to-end (UI → schema → prompts).
- Constraint filters: implemented end-to-end (UI → schema → pipeline enforcement)
  - `do_not_raise_if_p_neg_above` (schema + enforced pre-LLM by zeroing positive price changes when \(p\_neg\) exceeds threshold)
  - `exclude_low_stock` / `exclude_stockout_risk` (now wired since inventory is meaningful; forces `hold/restock` and zeroes repricing deltas before debate)

---

## Operational checklist / QA gates (to prevent low-quality outputs)

These are simple, automatable checks you can run after precompute and before demoing the app.
If a gate fails, the system should either (a) refuse to start, or (b) run in “degraded mode” with clear UI messaging.

### Retrieval QA

- **Corpus hygiene**
  - Flag or remove documents whose `product_document` contains placeholder strings like “PRODUCT NOT AVAILABLE”.
  - Flag unusually short docs (e.g., <200 chars) and inspect their source row.
- **Separation**
  - For a small set of canonical goals (profit, reduce returns, clear inventory), check that top-k retrieval scores show meaningful spread (not all ~equal).
  - If spread is weak, add/adjust structured tags in the retrieval document (price/rating/recency buckets).
  - Suggested gate: for each canonical goal, require \(\mathrm{p90}(score)-\mathrm{p50}(score) \ge 0.01\) across the top-k candidates (tune to your index/model).

### Pricing cache QA

- **Coverage**
  - Require coverage threshold (e.g., ≥90% of products in snapshot) or explicitly mark missing products as “no pricing model” and avoid repricing recommendations for them.
- **Hard-bound saturation (only applies if using `np.clip`)**
  - Gate: if \(|pct| \ge (b-0.5)\) for >30% of rows (where \(b\) is policy_bound), the cache is effectively saturated.
- **Soft-cap sanity (if using tanh soft-cap)**
  - Gate: after applying \(b\cdot\tanh(x/b)\), check that extreme pileups are reduced:
    - \(|pct| \ge 0.95b\) should be “rare” (set a target like <10–15%).
  - If still high, the upstream predictor distribution is too extreme and needs retraining or additional calibration.
- **Downstream flag**
  - Provide a boolean `near_bound` in the debate payload (\(|pct| \ge 0.9b\)) so LLMs can treat it as lower-trust.

### Sentiment cache QA

- **Low-signal detection**
  - If `n_reviews` is uniformly low or narrowly distributed, treat sentiment as low-confidence for ranking decisions.
  - Gate: if \(\mathrm{p75}(n\_reviews)-\mathrm{p25}(n\_reviews)\) is tiny (e.g., <=2) across most products, avoid using sentiment as a primary rank driver.
- **Explainability**
  - Add at least a lightweight explanation artifact (e.g., top negative keywords per product or per category) so the UI can justify “why risk is high”.

### Inventory cache QA

- **Distribution sanity**
  - Fail if `stock_status` distribution is degenerate (example: >95% in one class).
  - Fail if `risk_flag` is always false (or always true).
- **Stockout-risk fireability check (current rules)**
  - Gate: compute share where `on_hand_units <= safety_stock_units`. If ~0%, the `stockout_risk` rule cannot trigger and inventory will be non-informative.
- **Threshold alignment**
  - Print min/median/max of `available_to_sell`, `mean_daily_revenue`, `total_returns` and compare to rule thresholds.
  - Prefer dataset-aware thresholds (percentiles) when synthetic distributions shift.

### Debate / Judge QA

- **Prompt payload correctness**
  - Require that any `INPUT_JSON` sent to LLMs is valid JSON (not a Python dict string).
- **Fallback rate**
  - Track and surface whether Advocate/Critic/Judge used fallback behavior (JSON parse/validation failures). If fallbacks exceed a threshold, treat output as untrusted.
  - Gate: if judge fallback is used at all (or exceeds, say, 1–2% over a replay set), treat the judge prompt/payload as failing QA.
- **Critic output robustness**
  - Even when parsing succeeds, watch for Critic responses that include stray trailing text after the JSON object (the JSON extractor may “succeed silently” by pulling the first balanced object).
  - Gate: record raw Critic output and fail QA if raw output contains non-whitespace outside the extracted JSON object.

### UI QA (trust + transparency)

- **Metric honesty**
  - If “confidence” is derived from retrieval similarity, label it accordingly.
- **Evidence-first rendering**
  - For each ranked plan, show the top 1–2 signals that drove it (price position, sentiment risk, inventory risk) and indicate missing data explicitly.
- **No silent mock data**
  - Gate: the UI must not render hardcoded mock plans unless it clearly labels them as mock/demo data.
  - Prefer: show an error state if `ranked_actions` is empty, or if API health is degraded/offline.
- **Show retrieval snippet**
  - Gate: each displayed plan should show a short “why this SKU” excerpt from `evidence.points[0].text` when available.

---

## Remaining issues / things to pay attention to (next priorities)

These are the main reasons outputs can still feel “not reasonable” even after the fixes above, plus the most practical next improvements.

### 1) Business grounding is still thin (data limits)

- **Missing hard blockers**: we still don’t have true decision drivers like:
  - **COGS / unit cost / margin** (profit objective is a proxy)
  - **competitor prices / market price index**
  - **conversion rate / demand elasticity / ad ROAS**
  - **supplier lead time / replenishment constraints**
- Without these, many recommendations will remain **lightly justified** and may default to generic actions (especially `reprice`).

### 2) Pricing calibration is still “peaky” (post soft-cap)

- Step C removed the ±policy-bound pileups, but **predictions still cluster** (e.g., ~7–8%).
- This is now a **model/calibration** issue (not a clipping artifact). Recommended follow-ups:
  - add and pass a `near_bound`/`large_delta` flag so LLMs treat big deltas as lower-trust
  - add a simple calibration/shrinkage layer (e.g., shrink toward 0 when evidence is weak or pricing coverage is missing)

### 3) Retrieval remains sensitive to phrasing + min-score threshold

- With `COPILOT_RETRIEVAL_MIN_SCORE` set high (e.g., 0.80), many plausible goals will correctly return **0 strong matches**.
- The UI now previews this, but there are two product decisions to consider:
  - **Strict**: keep “0 matches” behavior (honest), and push users to scope to a category/subcategory.
  - **Soft fallback**: show “weak matches” below threshold (clearly labeled), to help users iterate faster.

### 4) Catalog document quality can still be low-signal

- Some `product_document` entries are extremely short/placeholder-like (e.g., “filter”).
- This reduces retrieval quality and makes downstream actions feel arbitrary.
- Recommended QA gate: flag very short docs and placeholder strings; consider excluding them from the retrieval corpus.

### 5) Judge rationales are not yet enforceably evidence-based

- Even when actions are guarded (e.g., low-stock → `hold`), the **rationale bullets** can still be generic.
- Next most impactful improvement for “small business usefulness”:
  - enforce a stricter rationale schema (must cite specific provided fields), and/or
  - compute deterministic “top driver signals” in code and render them in the UI (so explanations are auditable even if the LLM is vague).

### 6) Objective semantics (especially “profit”) are limited

- Because margin/COGS are missing, “profit” can only be used as a **policy framing** (e.g., be conservative on discounting), not a true optimizer.
- If profit is a core demo goal, you’ll need a cost proxy or add synthetic COGS.

### 7) Action space is broader but still not fully operational

- We added `hold/promote/investigate/restock`, but the system does not yet specify operational details like:
  - promotion mechanics (channel, discount depth, duration)
  - restock quantities / lead time constraints
- If you want “do this tomorrow” outputs, you’ll need either downstream integrations or a more concrete playbook with parameters.

### 8) Operational robustness (LLM + parsing + latency)

- Keep tracking: parse retries, fallback usage, and latency (index/model load can be heavy).
- Recommended UX: when the system is degraded (LLM down / caches missing), show explicit UI messaging and avoid producing high-confidence-looking plans.

### 9) Action classification thresholds are hardcoded (not configurable)

- The playbook rules in `pipeline.py` that determine `suggested_action` use fixed signal thresholds (e.g. `total_returns >= 2`, `p_neg >= 0.20`, `p_neg >= 0.30`, `n_reviews >= 10`).
- These are reasonable defaults and produce correct outputs for the current dataset, but they are not exposed as env vars or request parameters.
- For a production system or different product categories, these thresholds may need tuning — currently that requires a code change.
- **Note**: the shrinkage clamping bounds (0.70, 0.60, 0.80) are also hardcoded but intentionally so — they act as safety guardrails and should not be made configurable.


---

## Post-implementation fixes (after A–G + UI redesign)

These issues were identified by inspecting the two live runs after all A–G fixes were applied (snapshot `38710839ca6e1009`, queries: "Incxrease revenue for bed risers" and "Increase revenue for bed risers").

### Completed

- **mockPlans removed** — `mockPlans` constant in `copilot-v2/app/ui/src/App.jsx` was dead code (never used to populate the UI after E was implemented). Removed.
- **Impact Score removed** — `impactScore` was `retrievalSimilarity * 0.92`, a meaningless transform of retrieval score. Removed from `buildPlansFromRanked` in `App.jsx` and from the results card in `ResultsView.jsx`. The results card now shows only Risk Level and Retrieval similarity.

### Still outstanding

**Bugs (actively wrong)**

1. **Rationale hallucination** *(see Remaining issues #5)* — **Completed.** Implemented deterministic, signal-grounded rationale bullets in `Pipeline._merge_judge_output()` and auto-replaced empty/generic judge rationales with these bullets (pricing availability/flags, inventory, sentiment, returns, retrieval similarity).

2. **Typo handling in query rewriter** *(new finding from live runs)* — **Completed.** Added lightweight typo correction for common intent words (including adjacent transpositions like `Gorw → Grow`) and surfaced a `notes` field in `trace.query_rewrite` when corrections are applied.

3. **Advocate pricing direction** *(related to Remaining issues #2)* — **Completed.** Updated the Advocate prompt to start from the cache delta (`recommended_price_change_pct`) and treat `pricing_source="fallback"` as `investigate` with `0.0%` (no invented reprices).

**Quality gaps**

4. **Debate is not genuinely adversarial** *(related to Remaining issues #8)* — **Completed.** Made debate rounds causally linked: Critic disagreements are forced to cite `product_id` + concrete signals; Advocate revision is enforced (auto-rerun if unchanged); Judge output is deterministically linked to the latest Advocate plan (default per-SKU + default SKU selection).

5. **`suggested_action` is always "reprice"** *(related to Remaining issues #1)* — **Completed.** Tuned the playbook in `pipeline.py` so `suggested_action` fires on moderate signals (returns/sentiment), objective, pricing availability (fallback → investigate), and inventory risk (hold/restock), producing more diverse actions.

6. **Pricing still clusters around 7–8%** *(see Remaining issues #2)* — **Completed.** Added `pricing.large_delta` / `pricing.near_bound` flags to enriched candidates and passed them through debate payloads + UI Top Drivers. Added optional, tunable shrinkage (`COPILOT_ENABLE_PRICING_SHRINKAGE=1`) with env-var controls for shrink factors.

7. **No price-missing indicator in UI** *(related to Remaining issues #1)* — **Completed.** Added `pricing.price_missing` in the backend (true when pricing cache is missing) and rendered a clear UI warning + Top Driver entry (“Price: unknown / pricing unavailable”).

8. **Plan title is not user-friendly** *(new UI finding)* — **Completed.** Updated Results card titles to use product title parsed from the retrieval snippet (`evidence.points[0].text`) instead of raw `product_id`.

**Known limitations (require new data)**

9. **No COGS/margin** *(see Remaining issues #1 and #6)* — "Profit" objective is a proxy (conservative repricing framing). Already labelled as such in the UI (`"Grow profit (proxy)"`). Cannot be truly fixed without cost data.

10. **Operational action details missing** *(see Remaining issues #7)* — `restock`/`promote` actions have no quantities, channels, or durations attached. Would need a more concrete playbook with parameters or downstream integrations to produce "do this tomorrow" outputs.

### Suggested priority order

All items #1–#8 above have been completed; remaining work is primarily **known limitations** (#9–#10) and the broader “Remaining issues / next priorities” section (data and operational robustness).

| Priority | Item | Why |
|---|---|---|
| **Skip for now** | #9, #10 | Blocked by missing data or downstream integrations. Document as known limitations and move on. |

---

## Live run findings (post-fix validation)

Three rounds of runs were performed after all A–G fixes were applied (snapshot `38710839ca6e1009`). Each round validated a new set of fixes.

---

### Round 1 — Initial validation runs

**Setup**: query “Reduce returns for bed risers”, objective: reduce returns, `do_not_raise_if_p_neg_above=0.3`.

**What was working**
- `price_missing=true` correctly flagged for products with fallback pricing source
- `large_delta=true` correctly firing when delta ≥ ~7%
- `suggested_action` showing genuine variety (`hold`, `investigate`, `reprice`) across candidates
- Inventory classification working: `low_stock` + `risk_flag=true` correctly surfacing

**Issues found**
- Rationale bullets hallucinating specific field values (e.g. `p_neg=0.23` when actual was `0.125`, `total_returns=15` when actual was `1`)
- Query rewriter returning both a clarifying question and a retrieval query simultaneously; pipeline was proceeding with retrieval anyway
- Judge overriding `suggested_action=hold` with reprice at `large_delta=true` (debate linkage loop was undoing the guardrail)
- Retrieval min-score threshold (`COPILOT_RETRIEVAL_MIN_SCORE=0.80`) not enforced when env var was absent — retrieval agent defaults to `0.0`, so ~0.76-scoring non-bed-riser products passed through

---

### Round 2 — Post-fix validation (A, B, C, E)

**Setup**: same query. Fixes A (LLM grounding rule + few-shots), B (either/or), C (debate linkage guardrail), and E (min-score default) applied.

**What was working**
- Fix B confirmed: match preview correctly showed “Needs clarification” and did not proceed with retrieval when rewriter returned both fields
- Fix E confirmed: with min-score defaulting to 0.80, the previous ~0.76-scoring non-bed-riser products were filtered out

**Issues found**
- Rationale bullets still hallucinating (LLM grounding rule in `_SYSTEM` + few-shot update alone was not sufficient): e.g. rank 1 cited `p_neg=0.23` (actual: `0.125`) and `total_returns=15` (actual: `1`); rank 3 cited `total_returns=32` (actual: `0`)
- Retrieval scores were ~0.76 (below threshold) because `COPILOT_RETRIEVAL_MIN_SCORE` was not set in the environment for that run — confirmed the default was still `0.0` at the time

---

### Fixes applied (from live run findings)

- **A — Rationale hallucination**: Two-layer fix:
  - *Layer 1 (LLM)* — Added rationale grounding rule to `_SYSTEM` in `judge.py` and updated all three agents' few-shot examples to model correct numeric citations.
  - *Layer 2 (deterministic override)* — LLM grounding alone was still hallucinating specific numbers. Changed `_merge_judge_output()` in `pipeline.py` to always replace rationale bullets with deterministic ones built from the actual input data. This follows the same principle as RAG: the LLM's *reasoning* (what action to take) stays LLM-driven; the *factual citations* (field values from the payload) are grounded deterministically. The `_looks_generic()` check is no longer the gate — grounding is always enforced.
- **B — Query rewrite either/or**: Added `if rq and cq: rq = “”` in `pipeline.py` — if the rewriter returns both a clarifying question and a retrieval query, treat as clarification and do not proceed with retrieval.
- **C — Judge large_delta + hold guardrail**: Added `if suggested_ld == “hold” and large_delta: continue` to the debate linkage skip condition in `judge.py` — prevents the linkage loop from overwriting a hold that the guardrail already set.
- **D — Nondeterministic query rewriter**: No action needed (expected LLM behavior; both rewrite paths handled correctly after B).
- **E — Retrieval min-score not defaulting correctly**: Fixed by hardcoding `0.80` as the default in `pipeline.py` so the threshold is always active without requiring the env var.

---

### Round 3 — Final validation run

**Setup**: query “Reduce returns for bed risers due to negative customer reviews and quality complaints”, objective: reduce returns, `do_not_raise_if_p_neg_above=0.3`. All fixes (A layer 2, E) applied.

**Results** (snapshot `38710839ca6e1009`, run `run_20260429_042620_store_00_judge`)

| Rank | Product | Action | p_neg | total_returns | large_delta | Rationale grounded? |
|---|---|---|---|---|---|---|
| 1 | B09QM99LKG (Banqin Bed Risers) | reprice +5.33% | 0.08 | 2 | false | ✓ |
| 2 | B0C6FRCQDS (Headwind Furniture Riser) | hold | 0.00 | 4 | false | ✓ |
| 3 | B07D4358F2 (Tech Team Bed Risers) | investigate | 0.29 | 3 | true | ✓ |

**All fixes confirmed**
- Retrieval scores 0.83 / 0.83 / 0.82 — all above 0.80 threshold, all actual bed risers (E ✓)
- Rationale bullets exactly match real field values — no hallucinated numbers (A ✓)
- Rank 2 correctly `hold` due to `low_stock + risk_flag=true` guardrail (inventory guardrail ✓)
- Rank 3 correctly `investigate` due to `p_neg=0.29` + `large_delta=true` (C ✓)
- Actions are sensible for “reduce returns” objective: no unjustified upward repricing on high-return items

