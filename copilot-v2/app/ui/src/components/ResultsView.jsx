const riskStyles = {
  Low: "text-emerald-300",
  Medium: "text-amber-300",
  High: "text-rose-300",
};

function PortfolioSummary({ summary }) {
  if (!summary) return null;
  const { n, actionCounts, highNegSentiment, inventoryRisk, pricingMissing, highReturns, strategy } = summary;
  const pct = (k) => `${Math.round((actionCounts[k] || 0) / n * 100)}%`;
  const riskPct = (v) => `${Math.round(v / n * 100)}%`;
  return (
    <div className="mt-5 rounded-xl border border-slate-700 bg-slate-900/80 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-400">Portfolio Overview ({n} candidates)</p>
      <p className="mt-2 text-sm text-slate-200 leading-relaxed">{strategy}</p>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 text-xs">
        {[
          { label: "Reprice", value: pct("reprice"), color: "text-cyan-300" },
          { label: "Hold", value: pct("hold"), color: "text-slate-300" },
          { label: "Investigate", value: pct("investigate"), color: "text-amber-300" },
          { label: "Promote", value: pct("promote"), color: "text-emerald-300" },
          { label: "Restock", value: pct("restock"), color: "text-violet-300" },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-center">
            <p className={`text-base font-bold ${color}`}>{value}</p>
            <p className="mt-0.5 text-slate-500">{label}</p>
          </div>
        ))}
      </div>
      {(highNegSentiment > 0 || inventoryRisk > 0 || pricingMissing > 0 || highReturns > 0) && (
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {inventoryRisk > 0 && (
            <span className="rounded-full border border-rose-500/40 bg-rose-500/10 px-2 py-1 text-rose-300">
              {riskPct(inventoryRisk)} inventory risk
            </span>
          )}
          {highNegSentiment > 0 && (
            <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-amber-300">
              {riskPct(highNegSentiment)} high negative sentiment
            </span>
          )}
          {highReturns > 0 && (
            <span className="rounded-full border border-orange-500/40 bg-orange-500/10 px-2 py-1 text-orange-300">
              {riskPct(highReturns)} elevated returns
            </span>
          )}
          {pricingMissing > 0 && (
            <span className="rounded-full border border-slate-500/40 bg-slate-500/10 px-2 py-1 text-slate-400">
              {riskPct(pricingMissing)} pricing unavailable
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default function ResultsView({
  plans,
  resultsMessage = "",
  runContext = null,
  portfolioSummary = null,
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
            {/* Action badge — primary decision signal */}
            <div className={`mt-2 inline-block rounded-full border px-3 py-1 text-xs font-semibold ${plan.actionBadgeColor || "bg-slate-500/20 text-slate-300 border-slate-500/40"}`}>
              {plan.actionLabel || plan.finalActionType}
            </div>
            {/* Product name — secondary, smaller; full title in tooltip */}
            <h3 className="mt-2 text-sm font-medium text-slate-300 leading-snug" title={plan.fullTitle || plan.title}>
              {plan.title}
            </h3>

            {/* Pricing line — always visible, concrete dollar amounts */}
            {plan.pricingLine && (
              <p className={`mt-2 text-sm font-mono ${plan.pricingLine.dim ? "text-slate-500" : "text-slate-200"}`}>
                {plan.pricingLine.icon} {plan.pricingLine.text}
              </p>
            )}

            {/* WHY section */}
            {!!(plan.whyBullets || []).length && (
              <div className="mt-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Why</p>
                <ul className="mt-1.5 space-y-1.5">
                  {plan.whyBullets.map((b) => (
                    <li key={b} className="flex gap-1.5 text-sm text-slate-300">
                      <span className="mt-0.5 shrink-0 text-slate-600">—</span>
                      <span>{b}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Override note — seller language, shown only when recommendation was adjusted */}
            {plan.overrideNote && (
              <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                {plan.overrideNote}
              </div>
            )}

            {/* Caution notes — visible when action still contradicts a key signal */}
            {!!(plan.cautionNotes || []).length && (
              <div className="mt-2 space-y-1">
                {plan.cautionNotes.map((n) => (
                  <div key={n} className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
                    ⚠ {n}
                  </div>
                ))}
              </div>
            )}

            {/* Evidence — collapsed by default */}
            {plan.evidenceSnippet && (
              <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-sm">
                <summary className="cursor-pointer select-none text-xs uppercase tracking-wide text-slate-500 hover:text-slate-300">
                  Source evidence <span className="normal-case text-slate-600">(click to expand)</span>
                </summary>
                <p className="mt-2 text-slate-400 leading-relaxed">{plan.evidenceSnippet}</p>
              </details>
            )}

            {/* Risk | Confidence row */}
            <div className="mt-4 flex items-center justify-between gap-2">
              <div className="text-sm text-slate-400">
                Risk:{" "}
                <span className={`font-semibold ${riskStyles[plan.riskLevel]}`}>{plan.riskLevel}</span>
              </div>
              {plan.confidence && (
                <div className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${plan.confidence.bg}`}>
                  <span className={plan.confidence.color}>{plan.confidence.level} confidence</span>
                </div>
              )}
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
