export default function QueryInputView({ query, setQuery, onSubmit, isLoading }) {
  return (
    <div className="mx-auto max-w-3xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
      <p className="text-sm uppercase tracking-wider text-cyan-400">Step 1</p>
      <h1 className="mt-2 text-3xl font-semibold text-slate-100">Ask a business question</h1>
      <p className="mt-3 text-slate-400">
        Example: How do I increase revenue while reducing inventory risk?
      </p>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="How do I increase revenue?"
        className="mt-6 h-32 w-full resize-none rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none ring-cyan-400 placeholder:text-slate-500 focus:ring-2"
      />

      <button
        type="button"
        onClick={onSubmit}
        disabled={isLoading || !query.trim()}
        className="mt-5 rounded-xl bg-cyan-500 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
      >
        {isLoading ? "Launching agents..." : "Submit question"}
      </button>
    </div>
  );
}
