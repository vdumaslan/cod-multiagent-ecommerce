# Pipeline & Agent Fix Plan

_Last updated: 2026-05-08 (revised after architectural review)_

This document consolidates the diagnosed issues, their root causes, and the agreed fix plan
for the multi-agent CoD pipeline. Priority order reflects "fix biggest ambiguity first, avoid
behavior changes before the data contract is clearer."

---

## Architecture overview

```
retrieval → enrichment (_enrich) → baseline → Advocate/Critic (N rounds) → Judge
         → _merge_judge_output (deterministic guardrails) → UI
```

The deterministic merge/guardrail step means the system **can produce a sensible output even
when the Judge fails** — but it also means the UI output is not always a pure ACJ decision.
This is an acceptable engineering trade-off for a 7–8B model system; the guardrails are the
primary reliability mechanism, not the LLM.

---

## Confirmed bugs (from artifact inspection)

### Bug 1 — `_fix_fact_claims` corrects field value but leaves broken sentence context

After `_fix_fact_claims` replaces `pricing_source='fallback'` with `'cache'`, the surrounding
sentence still says "repricing without a cache signal is ungrounded" — internally contradictory.

**Fix:** add phrase-level cleanup in both `advocate.py` and `critic.py`:

```python
if src == "cache":
    s = re.sub(
        r";?\s*repricing without a cache signal is ungrounded\.?",
        "; pricing signal is available, but verify other risk signals before repricing.",
        s, flags=re.IGNORECASE,
    )
    s = re.sub(
        r"cannot ground a reprice\s*[—\-]\s*investigate instead\.?",
        "pricing signal is available; investigate only if other risk signals are elevated.",
        s, flags=re.IGNORECASE,
    )
```

---

### Bug 2 — Critic confuses candidate baseline with Advocate's proposal

Critic says "Advocate proposed reprice=+6.08%" when Advocate actually proposed `hold`. The
+6.08% is the candidate's model signal, not the Advocate's choice — both fields are named
`recommended_price_change_pct`.

**Fix:** rename in `slim_candidates` in both `advocate.py` and `critic.py`:
```python
"model_price_change_signal_pct": c.get("recommended_price_change_pct", 0.0)
```
Update grounding rule: "Use `model_price_change_signal_pct` as the pricing model's baseline.
Do not attribute this value to the Advocate unless it appears in `round1_advocate.proposed_actions`."

---

### Bug 3 — Critic simultaneously agrees and disagrees with the same product

**Fix:** add explicit rule to Critic prompt:
```
"- Do not list the same product_id in both agreements and disagreements."
```

---

### Bug 4 — Few-shot examples cause phrase leakage onto unrelated candidates

The LLM copies template phrases (e.g. "holding price to avoid stockout", "cannot ground a reprice")
onto products where those conditions don't apply.

**Advocate `_FEW_SHOT` fixes:**
- `"holding price to avoid stockout."` → `"hold because inventory risk makes demand-increasing actions unsafe."`
- `"cannot ground a reprice — investigate instead."` → `"pricing data unavailable — investigate to gather evidence before acting."`

**Critic `_FEW_SHOT` fix:**
- `"repricing without a cache signal is ungrounded."` → `"no pricing model data available — cannot verify pricing direction; suggest investigate."`

---

### Bug 5 — UI formatting: `%%` double percent and raw floats *(FIXED)*

`cleanTechnicalText` now consumes trailing `%`, handles `Advocate proposed reprice=N%` pattern,
and rounds all floats to 2 decimal places. Product IDs shown in full.

---

## System-level design issues

### S1 — `action_type = "reprice"` for every enriched candidate *(HIGH PRIORITY)*

**Confirmed in `2_enriched.json`:**
```json
{ "action_type": "reprice", "suggested_action": "investigate", ... }
```
LLMs receive both fields with no label explaining which is authoritative.

**Fix (Option A — minimal, safe):** change `pipeline.py` enrichment:
```python
"action_type": suggested_action,   # was: "action_type": "reprice"
```
ACJ can still override it. Full `decision_packet` refactor is Group 3.

---

### S2 — Baseline ranking by retrieval score only *(MEDIUM PRIORITY — moved up)*

Baseline is used when Judge fails or before debate. Weak baseline = weak fallback outputs.

**Fix — simple scoring heuristic:**
```python
def _baseline_score(c):
    score = c.get("evidence", {}).get("retrieval_score", 0.0)
    if c.get("pricing", {}).get("source") == "cache":   score += 0.10
    if c.get("pricing", {}).get("price_missing"):        score -= 0.20
    if c.get("sentiment", {}).get("p_neg", 0) >= 0.30:  score -= 0.20
    if c.get("signals", {}).get("total_returns", 0) >= 5: score -= 0.15
    if c.get("inventory", {}).get("risk_flag"):          score -= 0.20
    return score
```

---

### S3 — Retrieval scope category mismatch *(LOW PRIORITY — UI label only)*

FAISS retrieves by document text similarity, not category strings. `Amazon Home / Bento Boxes`
vs `Tools & Home Improvement / Bento Boxes` are the same product space. Fix: change UI label
to "Comparable products in Bento Boxes" rather than implying strict scope.

---

### S4 — Query rewrite `used=false` sends business language to retrieval *(MEDIUM PRIORITY)*

When `used=false`, raw business goal hits FAISS directly. "Increase sales" and "safe repricing"
add noise. If scope category/subcategory is already selected in the UI, use that as the
retrieval query fallback instead of the raw goal.

**Fix:** in query rewrite fallback path, default to `selected_subcategory` as the retrieval
query when rewrite is not applied.

---

### S5 — Judge parse failures hidden behind deterministic fallback *(GROUP 2)*

If Judge JSON fails, retry produces a different product set. `_merge_judge_output` recovers,
but the output may not reflect a coherent Judge decision. The system is working because of
deterministic recovery, not because the Judge is reliable.

**Fix:**
- Validate: Judge product IDs must be subset of latest Advocate proposed_actions.
- If Judge introduces new product IDs, discard and fall back to Advocate order.
- Log `judge_used=false` explicitly when deterministic recovery fires.

---

### S6 — Too many overlapping fields for LLM *(GROUP 3 — after data contract stabilizes)*

A candidate currently carries `action_type`, `suggested_action`, `recommended_price_change_pct`,
pricing flags, and baseline actions simultaneously. Small models cannot reliably distinguish them.

**Long-term fix — build a `decision_packet` per candidate:**
```json
{
  "product_id": "B0971KKSLF",
  "playbook_action": "reprice",
  "pricing_signal_available": true,
  "model_price_change_signal_pct": -6.08,
  "pricing_confidence": "medium",
  "sentiment_p_neg": 0.08,
  "n_reviews": 24,
  "total_returns": 0,
  "inventory_status": "healthy"
}
```

---

### S7 — Inventory-related language for healthy products *(GROUP 1 — via prompt fix)*

"Holding to avoid stockout" appears for products with `stock_status=healthy`, `risk_flag=false`.
This is purely few-shot leakage (see Bug 4). Additionally add prompt guard:
```
"Only reference stockout risk if inventory_status='stockout_risk'.
 Only reference low-stock if inventory_status='low_stock' or risk_flag=true.
 For healthy inventory, state: inventory is healthy and does not block this action."
```

---

### S8 — `large_delta` wrong computation base and wrong behavioral effect *(GROUP 1)*

**Current:** `large_delta = abs(price_change_raw) >= 0.7 * policy_bound` (from raw, before shrinkage)

`B0971KKSLF` raw=7.6%, shrunk=6.08%, but `large_delta=true`. The flag overstates risk.
LLMs treat `large_delta=true` as a stop sign rather than a confidence qualifier.

**Fix (two parts):**

1. Compute flags from **post-shrink** `price_change`, not `price_change_raw`. Keep the shrinkage
   trigger at `0.7 * policy_bound` (internal only) but recompute the flags after:
   ```python
   moderate_delta = abs(price_change) >= 0.60 * policy_bound
   large_delta    = abs(price_change) >= 0.85 * policy_bound
   near_bound     = abs(price_change) >= 0.95 * policy_bound
   ```

2. Change prompt framing so delta flags mean **"apply gradually"**, not **"reject"**:
   ```
   "moderate_delta=true: apply carefully, monitor response.
    large_delta=true: human review recommended before full rollout.
    near_bound=true: high caution — consider staged rollout.
    large_delta alone does NOT require hold or investigate unless combined with
    elevated returns (>=5) OR high negative sentiment (p_neg>=0.30 with n_reviews>=10)."
   ```

---

### Returns threshold refinement *(GROUP 1 — IMPLEMENTED)*

**Problem:** raw `total_returns` count alone is not enough context.
- 5 returns / 50 sold = 10% return rate → serious
- 5 returns / 10,000 sold = 0.05% return rate → normal for volume

**Data available:** `sales_daily.parquet` has `units_sold` per day per product. `total_units_sold`
was **not** in the inventory cache — added and cache rebuilt.

**Actual distribution in this snapshot (50K products):**
- Median return rate: 2%, p75: 3%, max: 10.7%
- Products with rate ≥ 5%: ~1,700 (3.4%) — reasonable investigate tier
- Products with rate ≥ 3%: ~12,000 (24%) — too aggressive for hard block

**Implemented rule (dual condition):**

```python
return_rate = total_returns / max(total_units_sold, 1)

# Hard investigate: raw count AND rate both elevated
if total_returns >= 5 and return_rate >= 0.05:
    force_investigate = True
# Moderate risk: surfaced as caution in UI / risk bullets, not a hard block
elif return_rate >= 0.03:
    reduce_confidence = True  # via risk bullet and supportingSignals warn flag

# When no volume denominator:
elif total_units_sold == 0 and total_returns >= 5:
    force_investigate = True  # fallback using count alone
```

Changes made:
- `build_inventory_cache.py`: added `total_units_sold` aggregation from `sales_daily`
- `InventoryAgent` / `bigquery_cache.py`: expose `total_units_sold`
- `pipeline.py`: `_compute_return_rate()` helper; `_suggest_action()` and guardrail 2 use dual condition
- `_deterministic_risk()`: shows `return_rate` percentage in risk bullets (e.g. "2.1% of 238 sold")
- `advocate.py`, `critic.py`: prompt updated to cite `return_rate`, not raw count alone
- `App.jsx`: `buildSupportingSignals`, `buildWhyBullets` show rate and use `highReturnRisk` / `modReturnRisk` flags

---

### Post-LLM contradiction checker *(GROUP 2)*

Prompt instructions alone will not prevent contradictions in 7B models. Add a deterministic
validator after parsing Critic/Judge output:

```python
def validate_critic_output(critic, advocate, candidates):
    advocate_pids = {a["product_id"] for a in advocate.get("proposed_actions", [])}
    # Remove products from agreements that also appear in disagreements
    disagreement_pids = {extract_pid(d) for d in critic.get("disagreements", [])}
    critic["agreements"] = [a for a in critic.get("agreements", [])
                            if extract_pid(a) not in disagreement_pids]
    # Remove claims that reference "Advocate proposed X" where X doesn't match
    # (implement as regex check against advocate's actual proposed_actions)
    return critic
```

---

## Revised implementation order

### Group 1 — Implement first (high impact, low risk, all Group 1 together)

| # | Fix | File(s) |
|---|---|---|
| S8 | `large_delta` after shrinkage; threshold 0.85; add `moderate_delta` | `pipeline.py` |
| Bug 2 | Rename `model_price_change_signal_pct` in slim candidates | `advocate.py`, `critic.py` |
| Bug 3 | Critic rule: no same-product agreement+disagreement | `critic.py` |
| S1 | `action_type = suggested_action` at enrichment | `pipeline.py` |
| Bug 4 | Few-shot leakage cleanup | `advocate.py`, `critic.py` |
| S7 | Inventory-language prompt guard | `advocate.py`, `critic.py` |
| Returns | Change hard-block threshold 3→5; 3–4 = caution only | `pipeline.py` |

### Group 2 — Next batch *(IMPLEMENTED)*

| # | Fix | File(s) | Status |
|---|---|---|---|
| Bug 1 | Sentence-level cleanup in `_fix_fact_claims` | `advocate.py`, `critic.py` | Done |
| S2 | Baseline ranking heuristic (`_baseline_score`) | `pipeline.py` | Done |
| S5 | Judge product ID validation + `judge_used` flag | `judge.py`, `pipeline.py` | Done |
| — | `validate_critic_output` post-LLM checker | `critic.py` | Done |

**Sentence cleanup** (`_fix_fact_claims`): after field-value correction, also strips contradictory phrases
(`"repricing without a cache signal is ungrounded"`, `"cannot ground a reprice"`, `"no pricing model data available"`)
when `pricing_source='cache'`.

**Baseline heuristic** (`_make_baseline`): replaces pure retrieval-score sort with a composite score:
```
score = retrieval_score
  + 0.10 if pricing source == 'cache'
  − 0.20 if price_missing
  − 0.20 if p_neg >= 0.30
  − 0.15 if return_rate >= 0.05 (or −0.10 raw count fallback)
  − 0.20 if inventory risk_flag
```

**Judge validation** (`judge.py`): detects product_ids the Judge introduces that were not in the
Advocate's `proposed_actions`; logs them as `judge_introduced_pids` and `judge_pid_warning`.
`judge_used=true/false` now surfaced in both `judge_raw` and the final pipeline response.

**`validate_critic_output`** (`critic.py`): three deterministic rules applied after `_fix_fact_claims`:
1. Remove a product_id from `agreements` if it also appears in `disagreements`.
2. Remove disagreements that falsely attribute an action to the Advocate for a pid not in its `proposed_actions`.
3. Replace `stockout_risk` language for products whose actual `stock_status` is not `stockout_risk`.

### Group 3 — Later / larger refactor

| # | Fix | Notes |
|---|---|---|
| S6 | Full `decision_packet` per candidate | Large refactor, replaces all overlapping fields |
| S4 | Query rewrite fallback to `selected_subcategory` | Medium effort |
| S3 | UI scope label change | Cosmetic, easy |
| — | Advanced confidence/risk scoring | Future |

---

## What this does NOT fix

- Critic being generally too aggressive (adversarial by design, 7B model limitation)
- Advocate flipping position entirely in Round 2 under Critic pressure (small model behavior)
- Judge multi-product coherence for large product sets

These are LLM size constraints. The deterministic guardrails in `_merge_judge_output` remain
the primary reliability mechanism, and that is the correct engineering approach for a 7–8B
research/demo system. The fixes above make the guardrails more principled and the LLM context
cleaner — they do not try to make the LLM perfect.
