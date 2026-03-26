# Final model registry

This file records compared models and the v1 lock for each subsystem.

## 1) Sentiment agent

Compared:
- VADER
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

Result (artifacts/evals/sentiment/summary.json):
- VADER macro F1: 0.3463
- RoBERTa macro F1: 0.3315

**Primary (v1): VADER**
- Reason: higher macro F1 on proxy labels with zero additional model serving overhead.
- Fallback: RoBERTa sentiment pipeline.

## 2) Pricing / value (tabular)

Compared:
- CatBoostRegressor
- HistGradientBoostingRegressor (sklearn baseline)

Result (artifacts/evals/tabular/metrics.json, target=`log_price`):
- CatBoost RMSE: 0.0359
- HGB RMSE: 0.0688

**Primary (v1): CatBoost**
- Reason: best RMSE on locked test split.
- Fallback: sklearn HGB.

## 3) Retrieval

Compared:
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

Result (artifacts/evals/retrieval/metrics.json):
- MiniLM Recall@10: 0.8489, nDCG@10: 0.7274, MRR: 0.6941
- bge-small Recall@10: 0.7911, nDCG@10: 0.6768, MRR: 0.6443

**Primary (v1): all-MiniLM-L6-v2 + FAISS IndexFlatIP**
- Reason: best retrieval metrics and low-latency CPU inference.
- Fallback: bge-small-en-v1.5.

## 4) Orchestrator / debate

Compared (v1 practical):
- Deterministic two-round debate + policy synthesis
- Direct single-pass rule synthesis (baseline)

**Primary (v1): deterministic two-round debate**
- Reason: satisfies debate architecture requirement while keeping zero paid API constraint.
- Fallback: single-pass rules-only summary if runtime fails.

---

## v1 model IDs used in app logs

- retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- debate: `deterministic_v1`
- policy: `rules_v1`
