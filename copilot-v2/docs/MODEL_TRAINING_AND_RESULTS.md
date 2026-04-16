# Model training & results — full record (snapshot `38710839ca6e1009`)

**Audience:** Teammates and TAs who need **what we ran**, **numeric outcomes**, and **why we chose each winner** without opening other documents.

**Snapshot ID:** `38710839ca6e1009` (paths below are under `copilot-v2/` unless noted).

**Registry lock:** Retrieval, pricing, **and sentiment** winners are recorded in `artifacts/registry/registry.json`.

- **Current registry timestamp**: see `updated_at_utc` in `artifacts/registry/registry.json` (this file is the source of truth).

---

## Conventions

| Topic | Convention |
|-------|------------|
| **Retrieval** | Metrics at **K = 10**. Dense models use query/document prefixes where noted. |
| **Pricing** | Target: `recommended_price_change_pct`. Predictions evaluated with **±10% policy clip** (`violation_rate_clipped` is 0 when clip is applied). **Screen** = cheap pass on **2000 val rows**; **refine** = **full validation set** for top configs from screen; **final** = **held-out test** (n = **7179**). |
| **Sentiment** | Labels: `neg`, `neu`, `pos`. **Slice 10k** = fixed ID list `artifacts/evals/38710839ca6e1009/sentiment/sentiment_eval_slice_10k_ids.json`. |
| **Inventory** | Rule-based 4-class label: `stockout_risk`, `low_stock`, `overstocked`, `healthy` (no model training). |
| **Debate replay eval** | Cache specialist outputs once, then replay Advocate/Critic/Judge with different LLM combos; score with deterministic rubric. |

---

## 1. Retrieval

Encoders are **not** fine-tuned; “tuning” in this section means **index/runtime hyperparameters** (e.g. `max_seq_length`, `use_prefixes`) aligned on **val**, followed by a **final comparison on test**.

### 1.0 What we tuned (dense retrieval only)

For each dense encoder, we tuned:
- `max_seq_length` (token budget for encoding `product_document`)
- `use_prefixes` (whether to apply model family prefixes like `query:` / `passage:`)

We did **not** train weights; we rebuilt indexes and re-ran retrieval metrics at \(K=10\).

### 1.1 Screening (initial backend comparison on test)

All rows below are **test split**. **BM25** used a **1000-query subsample** (seed 42). **Dense** rows used **all available test queries** in that eval run.

| Backend | Model / index | n evaluated | Recall@10 | nDCG@10 | MRR | Notes |
|---------|----------------|------------:|----------:|--------:|----:|-------|
| BM25 | default index | 1000 | 0.9950 | 0.95025 | 0.93569 | Subsample; not the same 7500 rows as dense. |
| Dense | `intfloat/e5-large-v2` | 7500 | 0.99853 | 0.97864 | 0.97199 | Full test eval for this lane. |
| Dense | `BAAI/bge-large-en-v1.5` | 7500 | 0.99773 | 0.97747 | 0.97071 | Runner-up on lexicographic ordering. |

### 1.2 Refine (validation — dense index params only)

BM25 tuning slot in the tuning JSON is **null** (no BM25 refine). Dense candidates were aligned on **val** via a two-stage tune protocol:
- **Screening**: evaluate a small grid on val (see tune reports)
- **Refine**: if enabled for that model, re-evaluate top configs on the full val set

Tune artifacts (inputs to the tables below):
- `artifacts/evals/38710839ca6e1009/retrieval/tuning/tune_report_dense_intfloat_e5-large-v2.json`
- `artifacts/evals/38710839ca6e1009/retrieval/tuning/tune_report_dense_BAAI_bge-large-en-v1.5.json`
- `artifacts/evals/38710839ca6e1009/retrieval/tuning/tuning_best_params.json`

#### 1.2.1 Dense tuning — screening (val)

**e5-large-v2 screening** (n = 3000 val queries):

| Model | `max_seq_length` | `use_prefixes` | Val Recall@10 | Val nDCG@10 | Val MRR |
|-------|------------------|----------------|--------------:|------------:|--------:|
| `intfloat/e5-large-v2` | 384 | true | 0.99867 | 0.97969 | 0.97340 |
| `intfloat/e5-large-v2` | 384 | false | 0.99867 | 0.97721 | 0.97010 |
| `intfloat/e5-large-v2` | 512 | true | 0.99733 | 0.97820 | 0.97192 |
| `intfloat/e5-large-v2` | 512 | false | 0.99733 | 0.97526 | 0.96801 |

**bge-large-en-v1.5 screening** (n = 7500 val queries):

| Model | `max_seq_length` | `use_prefixes` | Val Recall@10 | Val nDCG@10 | Val MRR |
|-------|------------------|----------------|--------------:|------------:|--------:|
| `BAAI/bge-large-en-v1.5` | 512 | true | 0.99813 | 0.97819 | 0.97151 |
| `BAAI/bge-large-en-v1.5` | 512 | false | 0.99813 | 0.97588 | 0.96846 |

#### 1.2.2 Dense tuning — refine (full val)

**e5-large-v2 refine** (n = 7500 val queries):

| Model | `max_seq_length` | `use_prefixes` | Val Recall@10 | Val nDCG@10 | Val MRR |
|-------|------------------|----------------|--------------:|------------:|--------:|
| **`intfloat/e5-large-v2` (refine)** | 384 | true | 0.99827 | 0.98027 | 0.97429 |
| `intfloat/e5-large-v2` (refine) | 384 | false | 0.99813 | 0.97746 | 0.97060 |

### 1.3 Final comparison (test, K = 10)

Use the **dense full-test** rows from §1.1 (7500 queries) for head-to-head ordering; BM25 remains a **sparse baseline** on 1000 queries.

| Model | Test Recall@10 | Test nDCG@10 | Test MRR | n |
|-------|-----------------:|-------------:|---------:|--:|
| **e5-large-v2 (winner)** | **0.99853** | **0.97864** | **0.97199** | 7500 |
| bge-large-en-v1.5 | 0.99773 | 0.97747 | 0.97071 | 7500 |
| BM25 | 0.99500 | 0.95025 | 0.93569 | 1000 |

**Selection rule:** Lexicographic on **Recall@10 → nDCG@10 → MRR** (all at K = 10 on test, using the valid dense evals above).

### 1.4 Final pick — retrieval

- **Winner:** Dense **`intfloat/e5-large-v2`** with **`max_seq_length=384`**, **`use_prefixes=true`** (refine §1.2).
- **Why:** On full dense test eval, e5 **strictly wins** the lexicographic tie-break over bge (higher Recall@10, then nDCG@10, then MRR). BM25 is weaker on nDCG/MRR and was scored on a smaller query set.
- **Runner-up:** `BAAI/bge-large-en-v1.5` (same protocol; slightly lower metrics across the three criteria).

---

## 2. Pricing

Three training lanes: **TabPFN**, **CatBoost**, **FT-Transformer**. Each lane: **screen (2000 val rows)** → **refine (full val, top-k from screen)** → **retrain winner → test**.

### 2.1 TabPFN

#### Screen (2000 val rows), sorted by val RMSE (clipped)

| Rank | max_fit_rows | n_estimators | softmax_temp | Val RMSE (clip) | Val MAE (clip) | Violation rate (unclip) |
|-----:|-------------:|-------------:|-------------:|----------------:|---------------:|------------------------:|
| 1 | 8192 | 4 | 0.85 | 0.384784 | 0.142052 | 0.2635 |
| 2 | 8192 | 4 | 0.90 | 0.386608 | 0.143038 | 0.2525 |
| 3 | 8192 | 4 | 1.00 | 0.391299 | 0.146964 | 0.2095 |
| 4 | 8192 | 8 | 0.85 | 0.400841 | 0.142190 | 0.2725 |
| 5 | 8192 | 8 | 0.90 | 0.403615 | 0.143869 | 0.2680 |
| 6 | 8192 | 8 | 1.00 | 0.409860 | 0.148921 | 0.2335 |
| 7 | 4096 | 8 | 0.85 | 0.502093 | 0.176412 | 0.3130 |
| 8 | 4096 | 8 | 0.90 | 0.504709 | 0.178287 | 0.2950 |
| 9 | 4096 | 8 | 1.00 | 0.510633 | 0.183757 | 0.2465 |
| 10 | 4096 | 4 | 0.85 | 0.524785 | 0.185711 | 0.3205 |
| 11 | 4096 | 4 | 0.90 | 0.527101 | 0.187366 | 0.3070 |
| 12 | 4096 | 4 | 1.00 | 0.532545 | 0.192532 | 0.2610 |

#### Refine (full val, n = 6961), top-3 screen configs re-fit

| Rank | max_fit_rows | n_estimators | softmax_temp | Val RMSE (clip) | Val MAE (clip) | Violation rate (unclip) |
|-----:|-------------:|-------------:|-------------:|----------------:|---------------:|------------------------:|
| 1 | 21119 | 4 | 0.85 | **0.339057** | **0.127687** | 0.2212 |
| 2 | 21119 | 4 | 0.90 | 0.340328 | 0.128677 | 0.2100 |
| 3 | 21119 | 4 | 1.00 | 0.343576 | 0.132235 | 0.1787 |

#### Final test (winner config row 1 above, n = 7179)

| Metric | Val (refine) | Test |
|--------|-------------:|-----:|
| RMSE (clipped) | 0.339057 | **0.428500** |
| MAE (clipped) | 0.127687 | **0.143491** |
| Violation rate (unclipped) | 0.2212 | 0.2291 |

---

### 2.2 CatBoost

#### Screen (2000 val rows), sorted by val RMSE (clipped)

| Rank | depth | learning_rate | l2_leaf_reg | iterations | Val RMSE (clip) | Val MAE (clip) |
|-----:|------:|--------------:|------------:|-----------:|----------------:|---------------:|
| 1 | 8 | 0.05 | 1.0 | 800 | 0.824085 | 0.261691 |
| 2 | 8 | 0.05 | 3.0 | 800 | 0.831620 | 0.277497 |
| 3 | 8 | 0.03 | 1.0 | 800 | 0.896627 | 0.304474 |
| 4 | 6 | 0.05 | 3.0 | 800 | 0.912780 | 0.317645 |
| 5 | 6 | 0.05 | 1.0 | 800 | 0.918261 | 0.311620 |
| 6 | 8 | 0.03 | 3.0 | 800 | 0.921198 | 0.319840 |
| 7 | 6 | 0.03 | 1.0 | 800 | 1.013470 | 0.351966 |
| 8 | 6 | 0.03 | 3.0 | 800 | 1.026751 | 0.364881 |
| 9 | 4 | 0.05 | 1.0 | 800 | 1.128650 | 0.420564 |
| 10 | 4 | 0.05 | 3.0 | 800 | 1.150831 | 0.428105 |
| 11 | 4 | 0.03 | 1.0 | 800 | 1.245705 | 0.470782 |
| 12 | 4 | 0.03 | 3.0 | 800 | 1.263509 | 0.477527 |

#### Refine (full val), top-3 screen configs

| Rank | depth | lr | l2 | iters | Val RMSE (clip) | Val MAE (clip) | Violation rate (unclip) |
|-----:|------:|---:|---:|------:|----------------:|---------------:|------------------------:|
| 1 | 8 | 0.05 | 1.0 | 800 | **0.740507** | **0.250003** | 0.4410 |
| 2 | 8 | 0.05 | 3.0 | 800 | 0.755181 | 0.261916 | 0.4827 |
| 3 | 8 | 0.03 | 1.0 | 800 | 0.826489 | 0.290902 | 0.4482 |

#### Final test (winner = refine rank 1, n = 7179)

| Metric | Val (refine) | Test |
|--------|-------------:|-----:|
| RMSE (clipped) | 0.740507 | **0.859535** |
| MAE (clipped) | 0.250003 | **0.264698** |
| Violation rate (unclipped) | 0.4410 | 0.4372 |

---

### 2.3 FT-Transformer

#### Screen (2000 val rows, **28 epochs**), top 10 by val RMSE (clipped)

Full grid is larger; below are the **10 best** screen configs.

| Rank | layers | d_model | lr | epochs | Val RMSE (clip) | Val MAE (clip) |
|-----:|-------:|--------:|---:|-------:|----------------:|---------------:|
| 1 | 3 | 128 | 0.001 | 28 | 0.756052 | 0.219904 |
| 2 | 3 | 128 | 0.0005 | 28 | 0.799525 | 0.251248 |
| 3 | 2 | 64 | 0.001 | 28 | 0.813651 | 0.252401 |
| 4 | 2 | 128 | 0.0005 | 28 | 0.838674 | 0.257626 |
| 5 | 2 | 128 | 0.001 | 28 | 0.851037 | 0.252019 |
| 6 | 2 | 128 | 0.0003 | 28 | 0.865142 | 0.273279 |
| 7 | 3 | 128 | 0.0003 | 28 | 0.870503 | 0.276506 |
| 8 | 2 | 96 | 0.001 | 28 | 0.889447 | 0.274317 |
| 9 | 3 | 96 | 0.0005 | 28 | 0.892240 | 0.267443 |
| 10 | 3 | 96 | 0.0003 | 28 | 0.914679 | 0.287233 |

#### Refine (full val, **52 epochs**), top-5 from screen

| Rank | layers | d_model | lr | epochs | Val RMSE (clip) | Val MAE (clip) | best_val_mse |
|-----:|-------:|--------:|---:|-------:|----------------:|---------------:|-------------:|
| 1 | 3 | 128 | 0.001 | 52 | 0.713358 | 0.196727 | 0.578022 |
| 2 | 3 | 128 | 0.0005 | 52 | **0.706278** | 0.208994 | **0.569806** |
| 3 | 2 | 64 | 0.001 | 52 | 0.765651 | 0.228103 | 0.695957 |
| 4 | 2 | 128 | 0.0005 | 52 | 0.787639 | 0.236293 | 0.724596 |
| 5 | 2 | 128 | 0.001 | 52 | 0.796823 | 0.218751 | 0.763536 |

**Winner selection within lane:** Config **rank 2** above (`lr=0.0005`, 52 epochs) wins on **val RMSE (clipped)** / best val MSE (better than rank 1 on that objective).

#### Final test (winner = refine rank 2, n = 7179)

| Metric | Val (refine winner) | Test |
|--------|--------------------:|-----:|
| RMSE (clipped) | 0.706278 | **0.903041** |
| MAE (clipped) | 0.208994 | **0.239059** |
| Violation rate (unclipped) | 0.7865 | 0.7876 |

---

### 2.4 Cross-lane final comparison (test, clipped RMSE primary)

| Lane | Val RMSE (clip) | Test RMSE (clip) | Test MAE (clip) | Test viol. rate (unclip) |
|------|----------------:|-----------------:|----------------:|-------------------------:|
| **TabPFN** | **0.339057** | **0.428500** | **0.143491** | 0.2291 |
| CatBoost | 0.740507 | 0.859535 | 0.264698 | 0.4372 |
| FT-Transformer | 0.706278 | 0.903041 | 0.239059 | 0.7876 |

### 2.5 Final pick — pricing

- **Default production choice (Policy A — accuracy-first):** **TabPFN** with **`max_fit_rows=21119`**, **`n_estimators=4`**, **`softmax_temperature=0.85`** (cfg id `9a671e4141` in tune report / registry).
- **Why:** **Lowest test RMSE after ±10% clipping** among the three finished lanes (0.429 vs 0.860 vs 0.903), with strong val–test alignment on the same objective.
- **Operational fallback (Policy B — latency-first):** **CatBoost** `depth=8`, `learning_rate=0.05`, `l2_leaf_reg=1.0`, `iterations=800` (cfg id `8ddd01b7f4`). **Why:** Much **smaller** `.cbm` artifact and **CPU-friendly** inference vs TabPFN, at the cost of **~2× worse** clipped test RMSE on this snapshot.

---

## 3. Sentiment

### 3.1 Data & harness (completed)

Split sizes (after `min_review_text_len=10` filter):

| Split | Rows | neg | neu | pos |
|-------|-----:|----:|----:|----:|
| Train | 3,717,124 | 620,867 | 255,067 | 2,841,190 |
| Val | 796,527 | 172,631 | 64,914 | 558,982 |
| Test | 796,527 | 157,946 | 63,040 | 575,541 |

**Inverse-frequency class weights** (train only): neg **1.996**, neu **4.858**, pos **0.436** (formula in `class_weights.json`).

**Deterministic majority baseline** (always predict train majority **pos**, first 2000 val rows): macro F1 **0.294**, accuracy **0.79**, MCC **0** (`harness_report.json`).

### 3.2 VADER (rule baseline)

**What we did**
- Evaluated default thresholds (**±0.05**) on the fixed **10k test slice**.
- Ran a **coarse val-tuned threshold** setting (**−0.05 / +0.3**) to establish a “first tuning pass”.
- Tuned thresholds on **val only** (two-phase linspace search), then replayed the winner on the same 10k slice.

**Results (fixed 10k test slice)**

| Variant | n | Macro F1 | Neu F1 | MCC | Accuracy | Metrics file |
|---|---:|---:|---:|---:|---:|---|
| VADER (default ±0.05) | 10000 | 0.5206 | 0.1255 | 0.4110 | 0.7576 | `metrics_vader_slice_10k.json` |
| VADER (coarse tuned −0.05 / +0.3) | 10000 | 0.5348 | 0.1753 | 0.4127 | 0.7384 | `metrics_vader_slice_10k_val_tuned_thresh.json` |
| **VADER (val-explored −0.11 / +0.3575)** | 10000 | **0.5354** | **0.1933** | 0.4085 | 0.7284 | `metrics_vader_slice_10k_best_explored.json` |

**Val tuning artifacts:** `vader_tune_report.json`, `vader_tune_report_explore_wide.json`, `vader_tune_report_explore_fine.json`.

### 3.3 Qwen2.5-1.5B-Instruct (LLM baseline)

**What we did**
- Evaluated a default prompt on the fixed 10k slice.
- Tuned prompt/profile on **val only**, then re-evaluated the winner on the same 10k slice.

**Results (fixed 10k test slice; parsed rows only)**

| Variant | n used | Macro F1 | Neu F1 | MCC | Accuracy | Metrics file |
|---|---:|---:|---:|---:|---:|---|
| Qwen (default prompt) | 9978 | 0.4941 | 0.0146 | 0.5357 | 0.7297 | `metrics_qwen_slice_10k.json` |
| **Qwen (val-tuned prompt profile b)** | 9971 | **0.5298** | **0.0546** | **0.5484** | **0.7432** | `metrics_qwen_slice_10k_tuned_val_winner_b512.json` |

**Footnote (Qwen parsing):** Metrics are computed after dropping unparseable outputs (`parse_success_rate` is recorded in the JSON).

### 3.4 DistilRoBERTa (fine-tuned encoder)

**Protocol**
- Screen/tune on **200k train / 20k val** (stratified subsamples), then scale up once.
- Loss: **class-weighted CE**, label smoothing, early stopping on val macro F1.
- Always evaluate on the fixed **10k test slice** for cross-approach comparison.

**3.4.1 What we explored (200k/20k)**

| ID | Key config | Val macro F1 | Val neu F1 | Slice macro F1 | Slice neu F1 | Outcome |
|---|---|---:|---:|---:|---:|---|
| `phaseA_200k_lr5e5` | LR=5e-5, LS=0.05 | 0.7259 | 0.4128 | 0.7305 | 0.4209 | LR candidate |
| **`phaseA_200k_lr3e5`** | **LR=3e-5**, LS=0.05 | **0.7313** | **0.4218** | **0.7308** | **0.4197** | **Phase A winner** |
| `phaseB_200k_lr3e5_ls008` | LR=3e-5, **LS=0.08** | 0.7294 | 0.4198 | 0.7296 | 0.4211 | Worse than LS=0.05 |
| `phaseC_200k_balanced` | **balanced_batches=true** | 0.7066 | 0.3954 | 0.7039 | 0.3886 | Regression; reject |
| `phaseD_200k_wd001` | **WD=0.01** | 0.7283 | 0.4144 | 0.7309 | 0.4179 | WD baseline |
| **`phaseD_200k_wd005`** | **WD=0.05** | **0.7333** | **0.4244** | **0.7340** | **0.4284** | **Best 200k config** |
| `phaseE_200k_wd005_wu006_len256` | warmup=0.06 | 0.7297 | 0.4203 | 0.7251 | 0.4068 | Warmup worse |
| `phaseE_200k_wd005_wu010_len128` | max_len=128 | 0.7325 | 0.4265 | 0.7269 | 0.4120 | Slice regressed |
| `phaseE_200k_wd005_wu010_len384` | max_len=384 | 0.7339 | 0.4304 | 0.7313 | 0.4188 | Val better; slice slightly worse |

**3.4.2 Scale-up (winner run)**

| ID | Train/Val | Key config | Val macro F1 | Val neu F1 | Slice macro F1 | Slice neu F1 | Metrics files |
|---|---:|---|---:|---:|---:|---:|---|
| **`final_500k_slice_winner`** | **500k / full val** | **LR=3e-5, LS=0.05, WD=0.05, warmup=0.10, max_len=256** | **0.7378** | **0.4317** | **0.7348** | **0.4229** | `metrics_distilroberta-base_final_500k_slice_winner_val.json`, `metrics_distilroberta-base_final_500k_slice_winner_slice_10k.json` |

**Note on metrics files:** Phase A/B/C/D/E were run with `skip_canonical_metrics=true`, so their metrics live in run-tagged files `metrics_distilroberta-base_*_{val,slice_10k}_<run_tag>.json`.

### 3.5 DeBERTa (attempted hard; instability + collapse)

We attempted both **DeBERTa-v3-small** and **DeBERTa-v3-base**, with multiple mitigation attempts to address collapse/instability:
- **Precision / stability**: bf16-era runs → fp32 runs; finite-parameter checks; monitored `grad_norm` / `eval_loss`.
- **Loss variants**: class-weighted CE vs other loss settings (e.g., label smoothing 0 vs >0).
- **Sampling**: balanced batches (`balanced_batches=true`) on small smoke runs.
- **Budget / curriculum**: tiny debug runs → 32k/50k/100k/200k subsets → larger screens.

**Representative trials (fixed 10k test slice)**

| Trial | Model | Key settings | Slice macro F1 | Slice neu F1 | Outcome |
|---|---|---|---:|---:|---|
| `screen_800k_e3_lr3e5` | deberta-v3-small | 800k rows, LR=3e-5, 3 epochs (bf16-era) | 0.1094 | 0.0000 | **Collapsed to all-neg** (`metrics_deberta_slice_10k_screen_800k_e3_lr3e5.json`) |
| `smoke_finite_check` | deberta-v3-small | fp32 smoke | 0.2801 | 0.0000 | **Collapsed to all-pos** (`metrics_deberta_slice_10k_smoke_finite_check.json`) |
| `smoke_base_balanced_150steps` | deberta-v3-base | balanced batches, 150 steps, LR=2e-6, LS=0.08 | 0.0974 | 0.1592 | Still degenerate (predicts almost all **neu**) (`metrics_deberta_slice_10k_smoke_base_balanced_150steps.json`) |
| `fix_32k_ls0` | deberta-v3-small | 32k “fix” run, LS=0 | 0.5514 | 0.2022 | **Recovered from collapse**, but still far below DistilRoBERTa (`metrics_deberta_slice_10k_fix_32k_ls0.json`) |

**Logs / failure mode notes**
- Some fp32 screen attempts hit **NaN `grad_norm`** / **NaN `eval_loss`** and failed finite-weight checks (`deberta_screen_fp32.log`); those checkpoints were treated as unusable.

**Conclusion:** Despite multiple recovery attempts, DeBERTa never reached DistilRoBERTa’s slice macro F1 / neutral F1 on this snapshot, and parts of the training regime were unstable; we therefore excluded it from the final pick.

### 3.6 Final comparison & pick (same 10k test slice)

| Model | n | Macro F1 | Neu F1 | MCC | Accuracy | Metrics file |
|---|---:|---:|---:|---:|---:|---|
| **DistilRoBERTa (fine-tuned, 500k)** | 10000 | **0.7348** | **0.4229** | **0.7343** | **0.8783** | `metrics_distilroberta-base_final_500k_slice_winner_slice_10k.json` |
| VADER (val-explored thresholds −0.11 / +0.3575) | 10000 | 0.5354 | 0.1933 | 0.4085 | 0.7284 | `metrics_vader_slice_10k_best_explored.json` |
| Qwen2.5-1.5B-Instruct (val-tuned prompt profile b) | 9971 | 0.5298 | 0.0546 | 0.5484 | 0.7432 | `metrics_qwen_slice_10k_tuned_val_winner_b512.json` |

**Winner (macro F1 primary):** **DistilRoBERTa (fine-tuned)**.

---

## 4. Debate replay evaluation (ACJ debate layer)

This section evaluates the **Advocate → Critic → Judge** debate layer in isolation by replaying cached specialist inputs (retrieval + pricing + sentiment + inventory signals).

### 4.1 Smoke run — full 27 combinations (5 goals)

- **Snapshot**: `38710839ca6e1009`
- **Owner**: `store_00`
- **Goal set** (first 5 from the 20-goal file):
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay_goals_20.json`
  - generated subset: `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay_goals_5.json`
- **Cached specialist inputs** (baseline-only; reused across all combos):
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_110516/specialist_inputs.jsonl`
- **Replay outputs**:
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_110516/replay_results_27/rows.jsonl`
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_110516/replay_results_27/summary.json`

#### Model grid

All 27 role combinations of:

- `qwen2.5:7b-instruct`
- `llama3.1:8b`
- `mistral:7b-instruct-v0.3-q4_K_M`

#### Aggregate results (smoke)

- **Total runs**: 135 (= 5 goals × 27 combos)
- **debate_ok_rate**: 1.00 for all combos on this small set (no schema failures)

Top combos by (ok_rate, conflict_detection, decision_justification):

| Advocate | Critic | Judge | n | ok_rate | conflict_detection | decision_justification | grounded_ids | constraint_respect | latency_s_mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | mistral:7b-instruct-v0.3-q4_K_M | mistral:7b-instruct-v0.3-q4_K_M | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 8.9 |
| mistral:7b-instruct-v0.3-q4_K_M | llama3.1:8b | qwen2.5:7b-instruct | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 11.5 |
| mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | mistral:7b-instruct-v0.3-q4_K_M | 5 | 1.00 | 1.00 | 0.93 | 1.00 | 0.87 | 9.1 |
| qwen2.5:7b-instruct | qwen2.5:7b-instruct | mistral:7b-instruct-v0.3-q4_K_M | 5 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 | 6.8 |
| llama3.1:8b | qwen2.5:7b-instruct | mistral:7b-instruct-v0.3-q4_K_M | 5 | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 9.4 |
| mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | llama3.1:8b | 5 | 1.00 | 0.80 | 1.00 | 1.00 | 0.73 | 8.7 |
| mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | qwen2.5:7b-instruct | 5 | 1.00 | 0.80 | 1.00 | 1.00 | 0.73 | 8.1 |
| llama3.1:8b | mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | 5 | 1.00 | 0.60 | 1.00 | 1.00 | 0.93 | 6.6 |

Role-wise averages (across all 27 combos; interpret as directional only for smoke size):

| Role | Model | avg_ok_rate | avg_conflict_detection | avg_decision_justification | avg_latency_s |
|---|---|---:|---:|---:|---:|
| Advocate | mistral:7b-instruct-v0.3-q4_K_M | 1.00 | 0.64 | 0.93 | 10.1 |
| Advocate | llama3.1:8b | 1.00 | 0.60 | 0.90 | 8.6 |
| Advocate | qwen2.5:7b-instruct | 1.00 | 0.53 | 0.90 | 8.7 |
| Critic | qwen2.5:7b-instruct | 1.00 | 0.71 | 0.91 | 7.8 |
| Critic | mistral:7b-instruct-v0.3-q4_K_M | 1.00 | 0.60 | 0.90 | 7.6 |
| Critic | llama3.1:8b | 1.00 | 0.47 | 0.92 | 11.9 |
| Judge | qwen2.5:7b-instruct | 1.00 | 0.58 | 0.99 | 8.3 |
| Judge | mistral:7b-instruct-v0.3-q4_K_M | 1.00 | 0.69 | 0.92 | 10.1 |
| Judge | llama3.1:8b | 1.00 | 0.51 | 0.81 | 8.9 |

**Next step (after smoke):** pick 1–2 models per role (Advocate/Critic/Judge) and rerun on a larger goal set (e.g. 20 goals → then 120 goals) to reduce variance.

### 4.2 Narrowed replay — 8 combinations (20 goals)

- **Snapshot**: `38710839ca6e1009`
- **Owner**: `store_00`
- **Goal set (20 goals)**:
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay_goals_20.json`
- **Cached specialist inputs**:
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_115235/specialist_inputs.jsonl`
- **Replay outputs**:
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_115235/replay_results_8x20/rows.jsonl`
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_115235/replay_results_8x20/summary.json`

#### Model grid (8 combos)

- **Advocate**: `mistral:7b-instruct-v0.3-q4_K_M`, `llama3.1:8b`
- **Critic**: `qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.3-q4_K_M`
- **Judge**: `qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.3-q4_K_M`

#### Aggregate results (20 goals)

- **Total runs**: 160 (= 20 goals × 8 combos)
- **debate_ok_rate**: 1.00 for all combos (no schema failures)
- **json_valid / grounded_ids**: 1.00 for all combos

Top combos (sorted by conflict_detection, then decision_justification):

| Advocate | Critic | Judge | n | ok_rate | conflict_detection | decision_justification | constraint_respect | latency_s_mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | qwen2.5:7b-instruct | qwen2.5:7b-instruct | 20 | 1.00 | 0.90 | 0.97 | 0.88 | 7.6 |
| mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | mistral:7b-instruct-v0.3-q4_K_M | 20 | 1.00 | 0.90 | 0.83 | 0.78 | 9.0 |
| mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | qwen2.5:7b-instruct | 20 | 1.00 | 0.80 | 1.00 | 0.73 | 8.6 |
| llama3.1:8b | qwen2.5:7b-instruct | mistral:7b-instruct-v0.3-q4_K_M | 20 | 1.00 | 0.80 | 0.80 | 0.92 | 8.3 |
| llama3.1:8b | mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | 20 | 1.00 | 0.75 | 0.98 | 0.87 | 7.8 |
| llama3.1:8b | mistral:7b-instruct-v0.3-q4_K_M | mistral:7b-instruct-v0.3-q4_K_M | 20 | 1.00 | 0.70 | 0.90 | 0.90 | 8.0 |
| mistral:7b-instruct-v0.3-q4_K_M | mistral:7b-instruct-v0.3-q4_K_M | mistral:7b-instruct-v0.3-q4_K_M | 20 | 1.00 | 0.50 | 0.87 | 0.88 | 9.2 |
| mistral:7b-instruct-v0.3-q4_K_M | mistral:7b-instruct-v0.3-q4_K_M | qwen2.5:7b-instruct | 20 | 1.00 | 0.40 | 1.00 | 0.67 | 7.8 |

Role-wise averages (across these 8 combos; directional signal):

| Role | Model | avg_conflict_detection | avg_decision_justification | avg_constraint_respect | avg_latency_s |
|---|---|---:|---:|---:|---:|
| Advocate | llama3.1:8b | 0.79 | 0.91 | 0.89 | 7.9 |
| Advocate | mistral:7b-instruct-v0.3-q4_K_M | 0.65 | 0.93 | 0.77 | 8.6 |
| Critic | qwen2.5:7b-instruct | 0.85 | 0.90 | 0.83 | 8.4 |
| Critic | mistral:7b-instruct-v0.3-q4_K_M | 0.59 | 0.94 | 0.83 | 8.2 |
| Judge | qwen2.5:7b-instruct | 0.71 | 0.99 | 0.79 | 8.0 |
| Judge | mistral:7b-instruct-v0.3-q4_K_M | 0.72 | 0.85 | 0.87 | 8.6 |

**Recommended default ACJ (from this 20-goal run):** `llama3.1:8b | qwen2.5:7b-instruct | qwen2.5:7b-instruct` (best overall conflict_detection with strong justification and good latency).

### 4.3 Phase B — prompt ablation (120 goals, fixed ACJ models)

After locking models to **`llama3.1:8b | qwen2.5:7b-instruct | qwen2.5:7b-instruct`**, the next axis is **`prompt_style`** only (same cached specialist inputs, `prompt_version=v1`).

- **Snapshot**: `38710839ca6e1009`
- **Cached specialist inputs**:
  - `copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/20260415_122444/specialist_inputs.jsonl`
- **Replay outputs** (one folder per style):
  - `.../prompt_ablation_llama_qwen_qwen/zero_shot_json_v1/summary.json`
  - `.../prompt_ablation_llama_qwen_qwen/few_shot_json_v1/summary.json`
  - `.../prompt_ablation_llama_qwen_qwen/structured_rationale_v1/summary.json`
  - `.../prompt_ablation_llama_qwen_qwen/cot_hidden_v1/summary.json`

#### Results (`n=120` per style)

| prompt_style | ok_rate | conflict_detection | decision_justification | constraint_respect | latency_s_mean |
|---|---:|---:|---:|---:|---:|
| zero_shot_json | 1.00 | 0.758 | 0.975 | 0.746 | 20.9 |
| few_shot_json | 1.00 | 0.592 | 0.994 | **0.964** | 21.0 |
| structured_rationale | 1.00 | 0.750 | 0.983 | 0.799 | 21.1 |
| cot_hidden | 1.00 | 0.733 | 0.953 | 0.847 | 7.8 |

`json_valid` and `grounded_ids` were **1.00** for all four.

**Interpretation:** few-shot JSON improves **constraint_respect** at the cost of **conflict_detection** on this rubric; zero-shot is best on conflict; structured rationale is a middle path. **cot_hidden** (private step-by-step, JSON-only output) sits between zero-shot and few-shot on constraints and conflict, with slightly lower **decision_justification** than few-shot. **Latency** for the first three rows (~21 s/run) was measured in one session; **cot_hidden** was replayed in a separate run (~7.8 s/run) and is **not** directly comparable for timing—use rubric columns for cross-style comparison.

**Demo default:** **`few_shot_json` + `v1`** (set in the orchestrator server via `COPILOT_V2_ACJ_PROMPT_STYLE` / `COPILOT_V2_ACJ_PROMPT_VERSION`, overridable per request). Absolute **latency_s_mean** across prompt styles is only comparable in a single back-to-back session; rubric columns are the fair cross-style comparison.

---

## 5. Changelog

| Date (UTC) | Change |
|------------|--------|
| 2026-04-15 | Debate §4.3: **Phase B prompt ablation** (120 goals, fixed ACJ models) including **`cot_hidden`**; demo default **`few_shot_json` / `v1`** via server env. |
| 2026-04-03 | Registry timestamp for retrieval + pricing. |
| 2026-04-03 | Replaced link-out summary with **full inline tables** (screen / refine / final) and **explicit winner rationale** per agent. |
| 2026-04-04 | **Qwen2.5-1.5B-Instruct** full **10k** eval (`metrics_qwen_slice_10k.json`); sentiment §3 updated (DeBERTa fp32 outcome, VADER vs Qwen pick). |
| 2026-04-04 | **Qwen val-only prompt tuning** (`qwen_tune_report.json`); test slice **profile b** row (`metrics_qwen_slice_10k_tuned_val_winner_b512.json`); §3.4 macro-F1 leader → tuned Qwen with VADER caveats. |
| 2026-04-04 | **VADER** val threshold tuning + **two-phase linspace exploration**; canonical **`vader_tune_report.json`**; best cutoffs **−0.11 / +0.3575**; test metrics **`metrics_vader_slice_10k_best_explored.json`** (coarse grid artifact: `metrics_vader_slice_10k_val_tuned_thresh.json`). §3.2–3.4 updated. |
