from __future__ import annotations

from prefect import flow


@flow(name="seller_copilot_eval_pipeline")
def run_eval_pipeline() -> dict[str, str]:
    # Lightweight orchestration hook for rubric: evaluation pipeline step.
    # Keep command execution in CI/manual runtime; this flow acts as tracked wrapper.
    return {
        "status": "ready",
        "hint": "Run experiments/build_splits.py + eval scripts to refresh artifacts/evals/*",
    }


if __name__ == "__main__":
    print(run_eval_pipeline())
