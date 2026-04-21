const riskStyles = {
  Low: "text-emerald-300",
  Medium: "text-amber-300",
  High: "text-rose-300",
};

export default function ResultsView({
  plans,
  selectedPlanId,
  onChoosePlan,
  onRejectAll,
  isSavingPlan,
  saveStatusMessage,
}) {
  return (
    <div className="relative mx-auto max-w-5xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
      <p className="text-sm uppercase tracking-wider text-cyan-400">Step 4</p>
      <h2 className="mt-2 text-3xl font-semibold text-slate-100">Ranked strategic plans</h2>
      <p className="mt-3 text-slate-400">Select one proposal to continue or reject all generated plans.</p>

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

            <ul className="mt-3 space-y-2 text-sm text-slate-300">
              {plan.actions.map((action) => (
                <li key={action}>- {action}</li>
              ))}
            </ul>

            <div className="mt-4 space-y-1 text-sm">
              <p className="text-slate-300">
                Impact Score: <span className="font-semibold text-cyan-300">{plan.impactScore}</span>
              </p>
              <p className="text-slate-300">
                Risk Level: <span className={`font-semibold ${riskStyles[plan.riskLevel]}`}>{plan.riskLevel}</span>
              </p>
              <p className="text-slate-300">
                Confidence: <span className="font-semibold text-slate-100">{plan.confidence}%</span>
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
