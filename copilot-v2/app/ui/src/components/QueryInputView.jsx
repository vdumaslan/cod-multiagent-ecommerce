export default function QueryInputView({
  query,
  setQuery,
  onSubmit,
  isLoading,
  clarifyingQuestion = "",
  rewriteNotes = "",
  catalogSummary = null,
  catalogFacets = null,
  selectedCategory = "",
  setSelectedCategory = () => {},
  selectedSubcategory = "",
  setSelectedSubcategory = () => {},
  matchPreview = null,
  horizonDays = 7,
  setHorizonDays = () => {},
  topNActions = 3,
  setTopNActions = () => {},
  maxAbsPriceChangePct = 10,
  setMaxAbsPriceChangePct = () => {},
  objective = "revenue",
  setObjective = () => {},
  excludeLowStock = false,
  setExcludeLowStock = () => {},
  excludeStockoutRisk = false,
  setExcludeStockoutRisk = () => {},
  doNotRaiseIfPNegAbove = "",
  setDoNotRaiseIfPNegAbove = () => {},
}) {
  const categories = (catalogFacets?.categories || []).slice(0, 50);
  const subcatsForCategory = (catalogFacets?.subcategories_by_category || {})[selectedCategory] || [];

  const objectiveLabel = {
    revenue: "grow revenue",
    profit: "grow profit",
    clear_inventory: "clear inventory",
    reduce_returns: "reduce returns",
    avoid_stockouts: "avoid stockouts",
  }[objective] || "improve performance";

  const suggestedGoals = (() => {
    const scope = selectedSubcategory || selectedCategory || "a product category";
    if (objective === "avoid_stockouts") {
      return [
        `Avoid stockouts for ${scope} ahead of upcoming events.`,
        `Prevent stockouts for ${scope}: prioritize restock and hold pricing on low-stock items.`,
      ];
    }
    if (objective === "clear_inventory") {
      return [
        `Clear overstock in ${scope} without increasing returns.`,
        `Reduce overstock for ${scope}: promote slow movers and investigate high-return items.`,
      ];
    }
    if (objective === "reduce_returns") {
      return [
        `Reduce returns for ${scope}: prioritize items with high negative sentiment and high returns.`,
        `Reduce complaints for ${scope}: investigate top-return SKUs and adjust promotions.`,
      ];
    }
    if (objective === "profit") {
      return [
        `Improve profit for ${scope}: avoid aggressive discounting and investigate high-return items.`,
        `Grow profit for ${scope}: focus on stable sentiment and manageable inventory risk.`,
      ];
    }
    return [
      `Grow revenue for ${scope} while controlling inventory risk and returns.`,
      `Increase sales for ${scope} with safe repricing and targeted promotions.`,
    ];
  })();

  return (
    <div className="mx-auto max-w-3xl rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-950/50">
      <p className="text-sm uppercase tracking-wider text-cyan-400">Step 1</p>
      <h1 className="mt-2 text-3xl font-semibold text-slate-100">Ask a business question</h1>
      <p className="mt-3 text-slate-400">
        Best results: include a category/subcategory or a concrete product keyword.
      </p>

      {!!clarifyingQuestion && (
        <div className="mt-5 rounded-xl border border-amber-400/50 bg-amber-500/15 px-4 py-3">
          <p className="text-sm font-semibold text-amber-200">Clarifying question</p>
          <p className="mt-1 text-sm text-amber-100">{clarifyingQuestion}</p>
          {!!rewriteNotes && (
            <p className="mt-2 text-xs text-amber-100/80">{rewriteNotes}</p>
          )}
        </div>
      )}

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Example: Avoid stockouts for Balloons ahead of upcoming events"
        className="mt-6 h-32 w-full resize-none rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none ring-cyan-400 placeholder:text-slate-500 focus:ring-2"
      />

      <div className="mt-5 grid gap-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4 sm:grid-cols-2">
        <label className="text-sm text-slate-300">
          <span className="text-xs uppercase tracking-wide text-slate-500">Category</span>
          <select
            value={selectedCategory}
            onChange={(e) => { setSelectedCategory(e.target.value); setSelectedSubcategory(""); }}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400"
          >
            <option value="">(Any)</option>
            {categories.map((c) => (
              <option key={c.name} value={c.name}>{c.name} ({c.count})</option>
            ))}
          </select>
        </label>

        <label className="text-sm text-slate-300">
          <span className="text-xs uppercase tracking-wide text-slate-500">Subcategory</span>
          <select
            value={selectedSubcategory}
            onChange={(e) => setSelectedSubcategory(e.target.value)}
            disabled={!selectedCategory}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400 disabled:opacity-50"
          >
            <option value="">{selectedCategory ? "(Any)" : "Select category first"}</option>
            {subcatsForCategory.map((s) => (
              <option key={s.name} value={s.name}>{s.name} ({s.count})</option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-5 grid gap-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4 sm:grid-cols-3">
        <label className="text-sm text-slate-300 sm:col-span-3">
          <span className="text-xs uppercase tracking-wide text-slate-500">Objective</span>
          <select
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400"
          >
            <option value="revenue">Grow revenue</option>
            <option value="profit">Grow profit (proxy)</option>
            <option value="clear_inventory">Clear inventory / reduce overstock</option>
            <option value="reduce_returns">Reduce returns / complaints</option>
            <option value="avoid_stockouts">Avoid stockouts</option>
          </select>
          <p className="mt-2 text-xs text-slate-500">
            Note: profit is a proxy here (margin/COGS are missing). This mainly affects ranking priorities and guardrails.
          </p>
        </label>

        <label className="text-sm text-slate-300">
          <span className="text-xs uppercase tracking-wide text-slate-500">Horizon</span>
          <select
            value={horizonDays}
            onChange={(e) => setHorizonDays(Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400"
          >
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
          </select>
        </label>

        <label className="text-sm text-slate-300">
          <span className="text-xs uppercase tracking-wide text-slate-500">Top actions</span>
          <select
            value={topNActions}
            onChange={(e) => setTopNActions(Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400"
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={5}>5</option>
          </select>
        </label>

        <label className="text-sm text-slate-300">
          <span className="text-xs uppercase tracking-wide text-slate-500">Max price change</span>
          <select
            value={maxAbsPriceChangePct}
            onChange={(e) => setMaxAbsPriceChangePct(Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400"
          >
            <option value={5}>±5%</option>
            <option value={10}>±10%</option>
            <option value={15}>±15%</option>
            <option value={20}>±20%</option>
          </select>
        </label>
      </div>

      <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
        <p className="text-xs uppercase tracking-wide text-slate-500">Suggested goals (based on objective + scope)</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {suggestedGoals.map((g) => (
            <button
              key={g}
              type="button"
              onClick={() => setQuery(g)}
              className="rounded-full border border-slate-700 bg-slate-950 px-3 py-1 text-xs text-slate-200 hover:bg-slate-900"
              title={`Insert goal to ${objectiveLabel}`}
            >
              {g}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Tip: selecting a category/subcategory improves retrieval. You can still edit the goal text freely.
        </p>
      </div>

      <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
        <p className="text-xs uppercase tracking-wide text-slate-500">Match preview (retrieval)</p>
        {!matchPreview ? (
          <p className="mt-2 text-sm text-slate-400">Type a goal to see how many strong matches exist.</p>
        ) : matchPreview.clarifying_question && (!matchPreview.retrieval_query || matchPreview.retrieval_query.trim() === "") ? (
          <div className="mt-2 rounded-lg border border-amber-400/50 bg-amber-500/10 px-3 py-2">
            <p className="text-sm font-semibold text-amber-200">Needs clarification</p>
            <p className="mt-1 text-sm text-amber-100">{matchPreview.clarifying_question}</p>
          </div>
        ) : (
          <div className="mt-2">
            <p className="text-sm text-slate-300">
              Strong matches (score ≥ {Number(matchPreview.min_score || 0).toFixed(2)}):{" "}
              <span className="font-semibold text-slate-100">{matchPreview.n_candidates_above_min_score}</span>
            </p>
            {!!(matchPreview.top_candidates || []).length && (
              <ul className="mt-2 space-y-1 text-sm text-slate-300">
                {(matchPreview.top_candidates || []).slice(0, 5).map((c) => (
                  <li key={c.product_id} className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
                    <p className="font-semibold text-slate-100">{c.title || c.product_id}</p>
                    <p className="mt-0.5 text-xs text-slate-400">
                      {c.category}{c.subcategory ? ` • ${c.subcategory}` : ""} • score {Number(c.score || 0).toFixed(3)}
                    </p>
                  </li>
                ))}
              </ul>
            )}
            {!matchPreview.n_candidates_above_min_score && (
              <p className="mt-2 text-sm text-slate-400">
                No strong matches. Add a category/subcategory or example product keyword.
              </p>
            )}
          </div>
        )}
      </div>

      <details className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-slate-200">Advanced filters</summary>
        <div className="mt-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={excludeLowStock}
                onChange={(e) => setExcludeLowStock(e.target.checked)}
                className="h-4 w-4 accent-cyan-400"
              />
              Exclude low-stock items from pricing actions
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={excludeStockoutRisk}
                onChange={(e) => setExcludeStockoutRisk(e.target.checked)}
                className="h-4 w-4 accent-cyan-400"
              />
              Exclude stockout-risk items from pricing actions
            </label>
          </div>

          <label className="mt-3 block text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wide text-slate-500">Do not raise price if negative sentiment above</span>
            <input
              value={doNotRaiseIfPNegAbove}
              onChange={(e) => setDoNotRaiseIfPNegAbove(e.target.value)}
              placeholder="e.g. 0.40 (leave blank to disable)"
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-cyan-400"
            />
            <p className="mt-2 text-xs text-slate-500">
              If enabled, candidates with \(p\_neg\) above the threshold won’t get positive price changes.
            </p>
          </label>
        </div>
      </details>

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
