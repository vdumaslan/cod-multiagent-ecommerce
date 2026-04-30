# RL Implementation + Evaluation Report (Version B)

This document describes how the **contextual bandit RL** component is implemented in this repo (Version B / AI Copilot only), and records a small evaluation run: **10 dataset-grounded business queries** and their end-to-end results.

---

## Scope and goals

### What RL is optimizing

We use a lightweight contextual bandit to tune a few internal “knobs” in **Version B** to optimize for:

- **More decisions / fewer abandonments**
- **Higher confidence ratings**

This is **not** a full RL environment simulation. It is an online tuning loop driven by the app’s existing A/B event stream.

### What RL is allowed to change (knobs)

RL selects one arm per run and applies its configuration for that run only:

- **Retrieval strictness**: `retrieval_min_score`
- **Query rewrite enable**: `COPILOT_ENABLE_QUERY_REWRITE`
- **Pricing shrinkage enable**: `COPILOT_ENABLE_PRICING_SHRINKAGE`

Arms are defined in `copilot-v2/app/rl.py` (`default_arms()`).

---

## Implementation overview

### Key files

- **Bandit logic**: `copilot-v2/app/rl.py`
  - arm definitions (`RLArm`, `default_arms()`)
  - selection policy (UCB-style)
  - state load/save (JSON, file-based)
  - reward computation and event update logic
- **Apply arm at run start**: `copilot-v2/app/pipeline.py`
  - selects an arm
  - applies overrides temporarily for that run
  - restores previous values after the run
  - writes arm selection into `trace.rl`
- **Update bandit from events**: `copilot-v2/app/api/app.py`
  - `/ab/event` updates the bandit **only when** `metadata.mode_used == "B"` and a `run_id` is present
  - `/rl/stats` returns current bandit stats
- **Trace schema**: `copilot-v2/app/api/schemas.py`
  - `TraceInfo.rl` holds arm + applied knobs

### Run attribution (run_id → arm_id)

Bandit updates happen after the run (decision/confidence events arrive later), so we attach:

- `run_id → arm_id` in the bandit state (so later events are credited to the correct arm)

This is handled via helper functions in `copilot-v2/app/rl.py` and called from `copilot-v2/app/pipeline.py`.

### Reward function

Reward is a scalar computed from the event stream (implemented in `copilot-v2/app/rl.py`):

- `+2.0` if `decision_made`
- `-2.0` if `abandoned`
- confidence rating \(c \in [1,5]\) adds \(0.6 \times (c - 3)\)

Example: `decision_made + confidence=5` → \(2.0 + 0.6 \times 2 = 3.2\)

### Where RL shows up in responses

Each Version B pipeline/orchestrate run returns `trace.rl`, containing:

- arm id/name
- the arm’s config
- the applied runtime values for that run
- `run_id` and `objective` (for debugging and attribution)

### State persistence

State is stored as a local JSON file under the artifacts/runs folder (see `copilot-v2/app/rl.py` for the exact path). It tracks:

- per-arm `n` and `mean_reward`
- a small `run_id → arm_id` map for pending attribution

### Important limitation (concurrency)

The current implementation applies some settings by mutating process-global state (environment variables and retrieval config) and then restoring them after the run. This is safe for typical **single-user local dev** workflows, but can race under **concurrent requests** in a shared server process.

---

## API endpoints (how to inspect RL)

- **Bandit stats**: `GET /rl/stats`
- **Event stream** (also updates RL for Version B): `POST /ab/event`
  - RL update occurs only when `metadata.mode_used == "B"` and `run_id` is provided

---

## Evaluation: 10 dataset-grounded queries + results

### How queries were chosen

Queries were chosen from catalog facets exposed by the API:

- `GET /catalog/summary`
- `GET /catalog/facets`

We selected a mix of high-coverage categories/subcategories (e.g., **Amazon Home**, **All Electronics**, **Appliances**, **Tools & Home Improvement**) and rotated objectives across:

- `revenue`, `reduce_returns`, `clear_inventory`, `avoid_stockouts`, `profit`

### End-to-end pipeline used

For each query we ran:

1. `POST /orchestrate` (Version B)
2. `POST /debate/judge` using:
   - `enriched_candidates`
   - `baseline_ranked_actions`
   - latest advocate/critic outputs from `debate_trace`

### Results (one run)

| Query | Objective | RL arm | run_id | Top 3 actions |
| --- | --- | --- | --- | --- |
| Revenue \| Amazon Home → Bed Frames | revenue | no_rewrite | 20260430_192720_visitor_doc_run | reprice B07DGB1DWX (+6.09%) \| hold B077F5FQ6L (+0.00%) \| hold B07FTL7RLT (+0.00%) |
| Reduce returns \| Amazon Home → Bath Rugs | reduce_returns | no_rewrite | 20260430_192732_visitor_doc_run | reprice B08RF111FZ (+4.38%) \| hold B07XL8MV9W (+0.00%) \| reprice B08X4G54T9 (+1.73%) |
| Clear inventory \| Amazon Home → Area Rugs | clear_inventory | no_rewrite | 20260430_192745_visitor_doc_run | hold B08FLZCTG3 (+0.00%) \| reprice B07ZN5WCYW (-6.86%) \| investigate B078SL4JM4 (+0.00%) |
| Avoid stockouts \| Amazon Home → Bed Pillows | avoid_stockouts | no_rewrite | 20260430_192757_visitor_doc_run | hold B07W5FLMGR (+0.00%) \| investigate B01NA0ELP2 (+0.00%) \| hold B073FFBLFZ (+0.00%) |
| Profit \| Amazon Home → Barstools | profit | no_rewrite | 20260430_192807_visitor_doc_run | hold B07L6NBVSV (+0.00%) \| reprice B08WXLFPY3 (-4.18%) \| hold B078Y4NCYQ (+0.00%) |
| Revenue \| Tools & Home Improvement → Bottles | revenue | no_rewrite | 20260430_192821_visitor_doc_run | reprice B008IRXCWA (+6.08%) \| reprice B07Q39BNSP (+6.09%) \| hold B00HJYUY4W (+0.00%) |
| Reduce returns \| All Electronics → Alarm Clocks | reduce_returns | no_rewrite | 20260430_192834_visitor_doc_run | reprice B0BK4LKSGZ (+6.09%) \| hold B094PX97H2 (+0.00%) \| investigate B00NKB6GAY (+0.00%) |
| Clear inventory \| Appliances → Air Purifier Parts & Accessories | clear_inventory | no_rewrite | 20260430_192846_visitor_doc_run | reprice B081FSZJ88 (-6.10%) \| reprice B07S2RXPNP (-6.09%) \| hold B06XVRRC6F (+0.00%) |
| Avoid stockouts \| All Electronics → Batteries | avoid_stockouts | no_rewrite | 20260430_192858_visitor_doc_run | *(no candidates returned in this run)* |
| Profit \| Amazon Home → Chairs | profit | no_rewrite | 20260430_192906_visitor_doc_run | hold B07Q72Y3PB (+0.00%) \| investigate B06X15BH5G (+0.00%) \| investigate B08XWVR2C9 (+0.00%) |

### Notable findings

- **Action diversity**: The top actions across queries include `hold`, `investigate`, and `reprice` (not all “reprice”).
- **One empty case**: “Avoid stockouts → Batteries” returned **0 enriched candidates** and therefore **0 ranked actions**.
  - This is a useful regression/coverage test: it shows a query that looks valid by facets can still miss retrieval (depending on the retrieval index content and current `retrieval_min_score`).
  - Next improvements could include: query rewrite enablement for that arm, lowering `retrieval_min_score` for stockout-type objectives, or improving the retrieval index coverage for that subcategory.

---

## Bugs found during evaluation (and fixes)

### Objective alias 422s (fixed)

During smoke testing, some clients/test harnesses used objective aliases (e.g., `profit_proxy`, `liquidation`, `clear-inventory`) which caused request validation errors (HTTP 422).

Fix: `copilot-v2/app/api/schemas.py` now normalizes a few common aliases to the canonical objective strings.
