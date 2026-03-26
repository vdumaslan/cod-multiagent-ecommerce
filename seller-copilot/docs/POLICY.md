# Policy Guardrails

Deterministic checks run before plans are shown in the app:

1. **Minimum margin floor** (default `0.08`) on impacted SKUs.
2. **Maximum absolute price change cap** (default `10%`).
3. **Must-cite evidence**: each plan must include non-empty `evidence_refs`.
4. **Safe fallback**: if all plans fail checks, return a single conservative fallback plan.

These guardrails are implemented in `src/copilot_runtime/policy.py` and parameterized in the Streamlit sidebar.
