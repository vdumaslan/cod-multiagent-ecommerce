const statusStyles = {
  idle: "bg-slate-700 text-slate-200",
  running: "bg-amber-500/20 text-amber-300 border border-amber-400/50",
  done: "bg-emerald-500/20 text-emerald-300 border border-emerald-400/40",
};

export default function PipelineStatusView({ agents }) {
  return (
    <div className="mx-auto max-w-4xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
      <p className="text-sm uppercase tracking-wider text-cyan-400">Step 2</p>
      <h2 className="mt-2 text-3xl font-semibold text-slate-100">Agent pipeline status</h2>
      <p className="mt-3 text-slate-400">Retrieval and analysis modules are running with mock staggered timing.</p>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {agents.map((agent) => (
          <div
            key={agent.name}
            className="rounded-xl border border-slate-800 bg-slate-950/70 p-4"
          >
            <div className="flex items-center justify-between">
              <span className="text-lg font-medium text-slate-100">{agent.name}</span>
              <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${statusStyles[agent.status]}`}>
                {agent.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
