const riskStyles = {
  Low: "text-emerald-300",
  Medium: "text-amber-300",
  High: "text-rose-300",
};

export default function ResultsView({
  plans,
  resultsMessage = "",
  runContext = null,
  selectedPlanId,
  onChoosePlan,
  onRejectAll,
  isSavingPlan,
  saveStatusMessage,
}) {
  if (!plans || plans.length === 0) {
    return (
      <div className="mx-auto max-w-3xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
        <p className="text-sm uppercase tracking-wider text-cyan-400">Step 4</p>
        <h2 className="mt-2 text-3xl font-semibold text-slate-100">No ranked plans</h2>
        <p className="mt-3 text-slate-300">
          {resultsMessage || "No ranked actions were produced for this query."}
        </p>
        <p className="mt-3 text-sm text-slate-400">
          Tip: check `copilot-v2/artifacts/runs/...` for `1_retrieval.json` and confirm whether retrieval returned any strong matches.
        </p>
        <button
          type="button"
          onClick={onRejectAll}
          className="mt-6 rounded-lg border border-rose-500/60 px-4 py-2 text-sm font-semibold text-rose-300 hover:bg-rose-500/10"
        >
          Back to query
        </button>
      </div>
    );
  }

  return (
    <div className="relative mx-auto max-w-5xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
      <p className="text-sm uppercase tracking-wider text-cyan-400">Step 4</p>
      <h2 className="mt-2 text-3xl font-semibold text-slate-100">Ranked strategic plans</h2>
      <p className="mt-3 text-slate-400">Select one proposal to continue or reject all generated plans.</p>

      {!!runContext && (
        <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Run context</p>
          <div className="mt-2 space-y-2 text-sm text-slate-300">
            <div>
              <span className="text-slate-500">Goal:</span>{" "}
              <span className="font-semibold text-slate-100">{runContext.goal || ""}</span>
            </div>
            {(runContext.selected_category || runContext.selected_subcategory) && (
              <div>
                <span className="text-slate-500">Scope:</span>{" "}
                <span className="font-semibold text-slate-100">
                  {[runContext.selected_category, runContext.selected_subcategory].filter(Boolean).join(" / ")}
                </span>
              </div>
            )}
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
              <span>objective={runContext.objective}</span>
              <span>horizon_days={runContext.horizon_days}</span>
              <span>top_n_actions={runContext.top_n_actions}</span>
              <span>max_abs_price_change_pct={runContext.max_abs_price_change_pct}</span>
              {runContext.retrieval_min_score != null && (
                <span>retrieval_min_score={Number(runContext.retrieval_min_score).toFixed(2)}</span>
              )}
              {runContext.exclude_low_stock ? <span>exclude_low_stock=true</span> : null}
              {runContext.exclude_stockout_risk ? <span>exclude_stockout_risk=true</span> : null}
              {runContext.do_not_raise_if_p_neg_above != null ? (
                <span>do_not_raise_if_p_neg_above={runContext.do_not_raise_if_p_neg_above}</span>
              ) : null}
            </div>
            {!!runContext.retrieval_query_preview && (
              <div className="text-xs text-slate-400">
                <span className="font-mono">{String(runContext.retrieval_query_preview).slice(0, 160)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        {plans.map((plan, idx) => (
          <div
            key={plan.id}
            className={`rounded-xl border p-5 ${
              selectedPlanId === plan.id ? "border-cyan-400 bg-cyan-500/10" : "border-slate-800 bg-slate-950/70"
            }`}
          >
            <p className="text-xs uppercase tracking-wide text-slate-500">Rank #{idx + 1}</p>
            <h3 className="mt-2 text-lg font-semibold text-slate-100">{plan.title}</h3>

            {(plan.suggestedAction || plan.finalActionType) && (
              <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
                  <p className="uppercase tracking-wide text-slate-500">Suggested action (playbook)</p>
                  <p className="mt-1 font-semibold text-slate-100">{plan.suggestedAction || "n/a"}</p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
                  <p className="uppercase tracking-wide text-slate-500">Final action (judge)</p>
                  <p className="mt-1 font-semibold text-cyan-200">{plan.finalActionType || "n/a"}</p>
                </div>
              </div>
            )}

            {plan.evidenceSnippet && (
              <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-sm">
                <p className="text-xs uppercase tracking-wide text-slate-500">Retrieval snippet</p>
                <p className="mt-1 text-slate-300">{plan.evidenceSnippet}</p>
              </div>
            )}

            {!!(plan.topDrivers || []).length && (
              <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-sm">
                <p className="text-xs uppercase tracking-wide text-slate-500">Top drivers (deterministic)</p>
                <ul className="mt-1 space-y-1 text-slate-300">
                  {(plan.topDrivers || []).map((d) => (
                    <li key={d}>- {d}</li>
                  ))}
                </ul>
              </div>
            )}

            <ul className="mt-3 space-y-2 text-sm text-slate-300">
              {plan.actions.map((action) => (
                <li key={action}>- {action}</li>
              ))}
            </ul>

            <div className="mt-4 space-y-1 text-sm">
              <p className="text-slate-300">
                Risk Level: <span className={`font-semibold ${riskStyles[plan.riskLevel]}`}>{plan.riskLevel}</span>
              </p>
              <p className="text-slate-300">
                Retrieval similarity: <span className="font-semibold text-slate-100">{plan.confidence}%</span>
              </p>
            </div>

            <button
              type="button"
              onClick={() => onChoosePlan(plan.id)}
              disabled={isSavingPlan}
              className="mt-5 w-full rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            >
              Choose this plan
            </button>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={onRejectAll}
        disabled={isSavingPlan}
        className="mt-6 rounded-lg border border-rose-500/60 px-4 py-2 text-sm font-semibold text-rose-300 hover:bg-rose-500/10"
      >
        Reject all
      </button>

      {isSavingPlan && (
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-slate-950/80 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-xl border border-cyan-500/40 bg-slate-900 p-6 text-center shadow-2xl shadow-cyan-900/20">
            <p className="text-xs uppercase tracking-wider text-cyan-400">BigQuery Persistence</p>
            <h3 className="mt-2 text-xl font-semibold text-slate-100">Saving selected plan</h3>
            <p className="mt-3 text-sm text-slate-300">{saveStatusMessage}</p>
            <div className="mx-auto mt-5 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-slate-700">
              <div className="h-full w-1/2 animate-pulse rounded-full bg-cyan-400" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
