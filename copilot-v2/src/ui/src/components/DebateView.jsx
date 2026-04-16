const actorStyles = {
  "Advocate LLM": {
    container: "justify-start",
    bubble: "border-cyan-500/30 bg-cyan-500/10",
    label: "text-cyan-300",
  },
  "Critic LLM": {
    container: "justify-end",
    bubble: "border-amber-500/30 bg-amber-500/10",
    label: "text-amber-300",
  },
  "Human Review": {
    container: "justify-end",
    bubble: "border-violet-500/30 bg-violet-500/10",
    label: "text-violet-300",
  },
  "Judge LLM": {
    container: "justify-start",
    bubble: "border-emerald-500/30 bg-emerald-500/10",
    label: "text-emerald-300",
  },
};

export default function DebateView({
  log,
  isDebatePlaying,
  isAwaitingContext,
  contextDraft,
  setContextDraft,
  canViewResults,
  onAddContext,
  onAnotherRound,
  onSubmitContext,
  onCancelContext,
  onMoveOn,
  onViewResults,
}) {
  const isDecisionTurn = !isDebatePlaying && !canViewResults && !isAwaitingContext;
  const isContextTurn = !isDebatePlaying && isAwaitingContext;
  const isResultsTurn = !isDebatePlaying && canViewResults;

  return (
    <div className="mx-auto max-w-4xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
      <p className="text-sm uppercase tracking-wider text-cyan-400">Step 3</p>
      <h2 className="mt-2 text-3xl font-semibold text-slate-100">Chain of Debate</h2>
      <p className="mt-3 text-slate-400">
        Advocate and critic models challenge assumptions before a judge synthesizes a plan.
      </p>
      {(isDecisionTurn || isContextTurn || isResultsTurn) && (
        <div className="mt-5 rounded-xl border border-amber-400/50 bg-amber-500/15 px-4 py-3">
          <p className="text-sm font-semibold text-amber-200">Your turn</p>
          {isDecisionTurn && (
            <p className="mt-1 text-sm text-amber-100">
              Choose one: add context, run another round, or move on to judge synthesis.
            </p>
          )}
          {isContextTurn && (
            <p className="mt-1 text-sm text-amber-100">
              Enter context and submit to continue the next advocate/critic round.
            </p>
          )}
          {isResultsTurn && (
            <p className="mt-1 text-sm text-amber-100">
              Judge synthesis is complete. Continue to the ranked plans.
            </p>
          )}
        </div>
      )}

      <div className="mt-6 h-[420px] space-y-3 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/70 p-4">
        {log.map((entry) => (
          <div key={entry.id} className={`flex ${actorStyles[entry.actor]?.container ?? "justify-start"}`}>
            <div className={`max-w-[85%] rounded-2xl border p-4 ${actorStyles[entry.actor]?.bubble ?? "border-slate-700 bg-slate-900"}`}>
              <p className={`text-sm font-semibold ${actorStyles[entry.actor]?.label ?? "text-cyan-300"}`}>{entry.actor}</p>
              <p className="mt-1 text-slate-200">{entry.message}</p>
            </div>
          </div>
        ))}
        {isDebatePlaying && (
          <p className="text-sm italic text-slate-400">New message incoming...</p>
        )}
      </div>

      {!canViewResults && (
        <div className="mt-6">
          {isAwaitingContext ? (
            <div className="rounded-xl border border-violet-500/30 bg-violet-500/10 p-4">
              <p className="text-sm font-semibold text-violet-300">Human Review Input</p>
              <textarea
                value={contextDraft}
                onChange={(e) => setContextDraft(e.target.value)}
                placeholder="Add more business context for the next round..."
                className="mt-3 h-24 w-full resize-none rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-violet-400 placeholder:text-slate-500 focus:ring-2"
              />
              <div className="mt-3 flex gap-3">
                <button
                  type="button"
                  onClick={onSubmitContext}
                  className="rounded-lg bg-violet-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-violet-300"
                >
                  Submit Context + Run Round
                </button>
                <button
                  type="button"
                  onClick={onCancelContext}
                  className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap gap-3">
              <button
                onClick={onAddContext}
                disabled={isDebatePlaying}
                className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Add Context
              </button>
              <button
                onClick={onAnotherRound}
                disabled={isDebatePlaying}
                className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Another Round
              </button>
              <button
                onClick={onMoveOn}
                disabled={isDebatePlaying}
                className="rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                Move On
              </button>
            </div>
          )}
        </div>
      )}

      {canViewResults && (
        <div className="mt-6">
          <button
            type="button"
            onClick={onViewResults}
            className="rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400"
          >
            View Ranked Plans
          </button>
        </div>
      )}
    </div>
  );
}
