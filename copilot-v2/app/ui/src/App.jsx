import { useEffect, useReducer, useRef, useState } from "react";
import DebateView from "./components/DebateView";
import PipelineStatusView from "./components/PipelineStatusView";
import QueryInputView from "./components/QueryInputView";
import ResultsView from "./components/ResultsView";

const API_BASE = "";
const DEFAULT_OWNER_ID = "store_00";
const pipelineAgents = ["Retrieval", "Sentiment", "Pricing", "Inventory"];
const initialPipelineState = pipelineAgents.map((name) => ({ name, status: "idle" }));

function pipelineReducer(state, action) {
  if (action.type === "RESET") return initialPipelineState;
  if (action.type === "SET_STATUS") {
    return state.map((a) => (a.name === action.agentName ? { ...a, status: action.status } : a));
  }
  return state;
}

async function postJson(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || `${path} returned not ok`);
  return data;
}

// ── Debate trace → human-readable turns ──────────────────────────────────────

function formatAdvocate(adv, round) {
  const prefix = round > 1 ? `[Round ${round}] ` : "";
  const actions = (adv.proposed_actions || [])
    .map((a) => {
      const pct = Number(a.recommended_price_change_pct || 0);
      return `• ${a.product_id}: ${a.action_type} ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
    })
    .join("\n");
  const claims = (adv.key_claims || []).map((c) => `• ${c}`).join("\n");
  const concerns = (adv.concerns || []).map((c) => `• ${c}`).join("\n");
  const parts = [];
  if (actions) parts.push(`${prefix}Proposed Actions:\n${actions}`);
  if (claims) parts.push(`Key Claims:\n${claims}`);
  if (concerns) parts.push(`Concerns:\n${concerns}`);
  return parts.join("\n\n") || "No advocate output.";
}

function formatCritic(crit, round) {
  const prefix = round > 1 ? `[Round ${round}] ` : "";
  const parts = [];
  const agreements = crit.agreements || [];
  const disagreements = crit.disagreements || [];
  const changes = crit.suggested_changes || [];
  if (agreements.length) parts.push(`${prefix}Agreements:\n${agreements.map((x) => `• ${x}`).join("\n")}`);
  if (disagreements.length) parts.push(`Disagreements:\n${disagreements.map((x) => `• ${x}`).join("\n")}`);
  if (changes.length) parts.push(`Suggested Changes:\n${changes.map((x) => `• ${x}`).join("\n")}`);
  return parts.join("\n\n") || "No critic output.";
}

function advCritTurns(adv, crit, round) {
  const now = Date.now();
  return [
    { id: `adv-r${round}-${now}`, actor: "Advocate LLM", message: formatAdvocate(adv, round) },
    { id: `crit-r${round}-${now + 1}`, actor: "Critic LLM", message: formatCritic(crit, round) },
  ];
}

// ── Plan building ─────────────────────────────────────────────────────────────

const mockPlans = [
  { id: "plan-a", title: "Margin-First Bundle Expansion", actions: ["Launch 3 category-specific bundles.", "Increase top ROAS campaigns by 15%.", "Weekly inventory checks before promo pushes."], impactScore: 88, riskLevel: "Medium", confidence: 82 },
  { id: "plan-b", title: "Inventory-Protected Growth", actions: ["Gate promotions by 21-day stock coverage.", "Prioritize substitutes for low-coverage SKUs.", "Reduce discount depth for volatile categories."], impactScore: 79, riskLevel: "Low", confidence: 76 },
  { id: "plan-c", title: "Aggressive Revenue Sprint", actions: ["Double ad spend in best channels for 2 weeks.", "Flash discount windows at peak traffic.", "Backfill risky SKUs via expedited suppliers."], impactScore: 91, riskLevel: "High", confidence: 67 },
];

function buildPlansFromRanked(ranked) {
  if (!ranked || !ranked.length) return mockPlans;
  return ranked.map((a) => {
    const pct = Number(a.recommended_price_change_pct || 0);
    const sent = a.sentiment || {};
    const inv = a.inventory || {};
    const pNeg = Number(sent.p_neg || 0);
    const highReturns = (a.signals?.total_returns || 0) > 3;
    const riskLevel = inv.risk_flag || pNeg > 0.4
      ? "High"
      : highReturns || pNeg > 0.2
      ? "Medium"
      : "Low";
    const retrieval = a.evidence?.retrieval_score ?? 0.7;
    const confidence = Math.max(55, Math.min(95, Math.round(Number(retrieval) * 100)));
    return {
      id: String(a.product_id),
      title: `${a.action_type || "reprice"} — ${a.product_id}`,
      actions: [
        `Pricing: ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}% (source=${a?.pricing?.source || "unknown"})`,
        sent.n_reviews ? `Sentiment (n=${sent.n_reviews}): +${Number(sent.p_pos || 0).toFixed(2)} ~${Number(sent.p_neu || 0).toFixed(2)} -${Number(sent.p_neg || 0).toFixed(2)}` : "Sentiment: (n/a)",
        `Inventory: ${inv.stock_status || "unknown"}`,
        ...(a.llm_rationale_bullets || []).map((x) => `Rationale: ${x}`),
        ...(a.llm_risk_bullets || []).map((x) => `Risk: ${x}`),
      ].slice(0, 8),
      impactScore: Math.max(50, Math.min(99, Math.round(confidence * 0.92))),
      riskLevel,
      confidence,
    };
  });
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [view, setView] = useState("query");
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [pipeline, dispatchPipeline] = useReducer(pipelineReducer, initialPipelineState);
  const [debateLog, setDebateLog] = useState([]);
  const [isDebatePlaying, setIsDebatePlaying] = useState(false);
  const [isAwaitingContext, setIsAwaitingContext] = useState(false);
  const [contextDraft, setContextDraft] = useState("");
  const [canViewResults, setCanViewResults] = useState(false);
  const [selectedPlanId, setSelectedPlanId] = useState(null);
  const [isSavingPlan, setIsSavingPlan] = useState(false);
  const [saveStatusMessage, setSaveStatusMessage] = useState("");
  const [apiHealth, setApiHealth] = useState("checking");
  const [plans, setPlans] = useState([]);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [llmRunningLabel, setLlmRunningLabel] = useState("");

  // Debate state across N rounds
  const [roundNumber, setRoundNumber] = useState(1);
  const [latestAdvocate, setLatestAdvocate] = useState(null);
  const [latestCritic, setLatestCritic] = useState(null);

  const debateTimersRef = useRef([]);
  const saveTimerRef = useRef(null);

  // ── Health check ────────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error();
        if (!cancelled) setApiHealth("online");
      } catch {
        if (!cancelled) setApiHealth("offline");
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); }, []);

  // ── Debate helpers ──────────────────────────────────────────────────────────

  const clearDebateTimers = () => {
    debateTimersRef.current.forEach(clearTimeout);
    debateTimersRef.current = [];
  };

  const playDebateTurns = (turns, onDone) => {
    clearDebateTimers();
    setIsDebatePlaying(true);
    turns.forEach((turn, idx) => {
      const t = setTimeout(() => setDebateLog((prev) => [...prev, turn]), idx * 1200);
      debateTimersRef.current.push(t);
    });
    const doneT = setTimeout(() => {
      setIsDebatePlaying(false);
      if (onDone) onDone();
    }, turns.length * 1200);
    debateTimersRef.current.push(doneT);
  };

  // ── Step 1 → Step 2: submit query ──────────────────────────────────────────

  const handleSubmitQuery = () => {
    if (!query.trim()) return;
    setIsLoading(true);
    setPipelineResult(null);
    dispatchPipeline({ type: "RESET" });
    setView("pipeline");

    postJson("/pipeline", {
      goal: query.trim(),
      owner_id: DEFAULT_OWNER_ID,
      enable_pricing: true,
      enable_sentiment: true,
    })
      .then((data) => setPipelineResult(data))
      .catch(() => { setView("query"); setIsLoading(false); });
  };

  // Pipeline animation — Inventory stays "running" until API returns
  useEffect(() => {
    if (view !== "pipeline") return undefined;
    const timers = [];
    const s = (fn, ms) => { const t = setTimeout(fn, ms); timers.push(t); };
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Retrieval", status: "running" }), 0);
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Retrieval", status: "done" }), 700);
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Sentiment", status: "running" }), 700);
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Sentiment", status: "done" }), 1400);
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Pricing", status: "running" }), 1400);
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Pricing", status: "done" }), 2100);
    s(() => dispatchPipeline({ type: "SET_STATUS", agentName: "Inventory", status: "running" }), 2100);
    return () => timers.forEach(clearTimeout);
  }, [view]);

  // When /pipeline resolves → finish animation, go to debate
  useEffect(() => {
    if (view !== "pipeline" || !pipelineResult) return undefined;
    dispatchPipeline({ type: "SET_STATUS", agentName: "Inventory", status: "done" });
    const t = setTimeout(() => {
      setIsLoading(false);
      setView("debate");
    }, 400);
    return () => clearTimeout(t);
  }, [view, pipelineResult]);

  // ── Step 3: debate — N rounds, judge runs only on Move On ──────────────────

  useEffect(() => {
    if (view !== "debate") { clearDebateTimers(); return undefined; }

    setDebateLog([]);
    setCanViewResults(false);
    setContextDraft("");
    setIsAwaitingContext(false);
    setRoundNumber(1);
    setLatestAdvocate(null);
    setLatestCritic(null);
    setLlmRunningLabel("Advocate and Critic LLMs are running...");

    let cancelled = false;
    postJson("/debate/start", {
      goal: query,
      owner_id: DEFAULT_OWNER_ID,
      enriched_candidates: pipelineResult?.enriched_candidates || [],
      baseline_actions: pipelineResult?.baseline_ranked_actions || [],
      advocate_model: "llama3.1:8b",
      critic_model: "qwen2.5:7b-instruct",
      judge_model: "qwen2.5:7b-instruct",
      prompt_style: "few_shot_json",
      prompt_version: "v1",
    }).then((data) => {
      if (cancelled) return;
      setLlmRunningLabel("");
      setLatestAdvocate(data.advocate);
      setLatestCritic(data.critic);
      if (data.advocate && data.critic) {
        playDebateTurns(advCritTurns(data.advocate, data.critic, 1));
      }
    }).catch(() => {
      if (!cancelled) setLlmRunningLabel("");
    });

    return () => { cancelled = true; clearDebateTimers(); };
  }, [view]); // eslint-disable-line react-hooks/exhaustive-deps

  const addErrorTurn = (msg) => setDebateLog((prev) => [...prev, { id: `err-${Date.now()}`, actor: "System", message: msg }]);

  // "Another Round" — advocate + critic again, decision panel returns after
  const handleAnotherRound = async () => {
    if (llmRunningLabel || isDebatePlaying || isAwaitingContext || canViewResults) return;
    const nextRound = roundNumber + 1;
    setLlmRunningLabel("Advocate and Critic LLMs are running...");
    try {
      const data = await postJson("/debate/continue", {
        goal: query,
        owner_id: DEFAULT_OWNER_ID,
        enriched_candidates: pipelineResult?.enriched_candidates || [],
        baseline_actions: pipelineResult?.baseline_ranked_actions || [],
        prev_advocate: latestAdvocate || {},
        prev_critic: latestCritic || {},
        human_feedback: null,
      });
      setRoundNumber(nextRound);
      setLatestAdvocate(data.advocate);
      setLatestCritic(data.critic);
      setLlmRunningLabel("");
      playDebateTurns(advCritTurns(data.advocate, data.critic, nextRound));
    } catch {
      setLlmRunningLabel("");
      addErrorTurn("Round failed — LLM may be unavailable. Try again.");
    }
  };

  // "Add Context" — show text input
  const handleAddContext = () => {
    if (llmRunningLabel || isDebatePlaying || canViewResults) return;
    setIsAwaitingContext(true);
  };

  // "Submit Context + Run Round" — advocate + critic with human feedback
  const handleSubmitContext = async () => {
    if (!contextDraft.trim() || llmRunningLabel || isDebatePlaying || canViewResults) return;
    const feedback = contextDraft.trim();
    const nextRound = roundNumber + 1;

    setDebateLog((prev) => [...prev, { id: `human-${Date.now()}`, actor: "Human Review", message: feedback }]);
    setIsAwaitingContext(false);
    setContextDraft("");
    setLlmRunningLabel("Advocate and Critic LLMs are running...");

    try {
      const data = await postJson("/debate/continue", {
        goal: query,
        owner_id: DEFAULT_OWNER_ID,
        enriched_candidates: pipelineResult?.enriched_candidates || [],
        baseline_actions: pipelineResult?.baseline_ranked_actions || [],
        prev_advocate: latestAdvocate || {},
        prev_critic: latestCritic || {},
        human_feedback: feedback,
      });
      setRoundNumber(nextRound);
      setLatestAdvocate(data.advocate);
      setLatestCritic(data.critic);
      setLlmRunningLabel("");
      playDebateTurns(advCritTurns(data.advocate, data.critic, nextRound));
    } catch {
      setLlmRunningLabel("");
      addErrorTurn("Round failed — LLM may be unavailable. Try again.");
    }
  };

  // "Move On" — judge runs exactly once here
  const handleMoveOn = async () => {
    if (llmRunningLabel || isDebatePlaying || isAwaitingContext || canViewResults) return;
    setLlmRunningLabel("Judge LLM is running...");
    try {
      const data = await postJson("/debate/judge", {
        goal: query,
        owner_id: DEFAULT_OWNER_ID,
        enriched_candidates: pipelineResult?.enriched_candidates || [],
        baseline_actions: pipelineResult?.baseline_ranked_actions || [],
        latest_advocate: latestAdvocate || {},
        latest_critic: latestCritic || {},
      });
      setLlmRunningLabel("");
      const judgeContent = data.judge_raw?.judge?.content;
      const judgeTurn = {
        id: `judge-${Date.now()}`,
        actor: "Judge LLM",
        message: judgeContent ? judgeContent.slice(0, 600) : "Judge synthesis complete. Final ranked actions selected.",
      };
      playDebateTurns([judgeTurn], () => {
        setPlans(buildPlansFromRanked(data.ranked_actions));
        setCanViewResults(true);
      });
    } catch {
      setLlmRunningLabel("");
      addErrorTurn("Judge failed — LLM may be unavailable. Try again.");
    }
  };

  // ── Step 4: results ─────────────────────────────────────────────────────────

  const handleRejectAll = () => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    clearDebateTimers();
    setSelectedPlanId(null);
    setQuery("");
    setDebateLog([]);
    setIsDebatePlaying(false);
    setIsAwaitingContext(false);
    setCanViewResults(false);
    setContextDraft("");
    setIsSavingPlan(false);
    setSaveStatusMessage("");
    setPlans([]);
    setPipelineResult(null);
    setLatestAdvocate(null);
    setLatestCritic(null);
    setRoundNumber(1);
    setIsLoading(false);
    setView("query");
  };

  const handleChoosePlan = (planId) => {
    const chosen = plans.find((p) => p.id === planId);
    if (!chosen) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    setSelectedPlanId(planId);
    setIsSavingPlan(true);
    setSaveStatusMessage(`Saving "${chosen.title}"...`);
    saveTimerRef.current = setTimeout(() => {
      setSaveStatusMessage(`Saved "${chosen.title}".`);
      saveTimerRef.current = setTimeout(handleRejectAll, 1200);
    }, 1000);
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <div className="mx-auto mb-8 max-w-6xl">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm uppercase tracking-[0.2em] text-cyan-400">Multi-Agent BI Dashboard</p>
          <span
            className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${
              apiHealth === "online" ? "bg-emerald-500/20 text-emerald-300"
              : apiHealth === "offline" ? "bg-rose-500/20 text-rose-300"
              : "bg-amber-500/20 text-amber-300"
            }`}
          >
            API {apiHealth}
          </span>
        </div>
        <p className="mt-2 text-slate-400">
          Flow: query → pipeline → debate (round {roundNumber}) → judge → results
        </p>
      </div>

      {view === "query" && (
        <QueryInputView query={query} setQuery={setQuery} onSubmit={handleSubmitQuery} isLoading={isLoading} />
      )}

      {view === "pipeline" && <PipelineStatusView agents={pipeline} />}

      {view === "debate" && (
        <DebateView
          log={debateLog}
          llmRunningLabel={llmRunningLabel}
          isDebatePlaying={isDebatePlaying}
          isAwaitingContext={isAwaitingContext}
          contextDraft={contextDraft}
          setContextDraft={setContextDraft}
          canViewResults={canViewResults}
          onAddContext={handleAddContext}
          onAnotherRound={handleAnotherRound}
          onSubmitContext={handleSubmitContext}
          onCancelContext={() => { setIsAwaitingContext(false); setContextDraft(""); }}
          onMoveOn={handleMoveOn}
          onViewResults={() => setView("results")}
        />
      )}

      {view === "results" && (
        <ResultsView
          plans={plans.length ? plans : mockPlans}
          selectedPlanId={selectedPlanId}
          onChoosePlan={handleChoosePlan}
          onRejectAll={handleRejectAll}
          isSavingPlan={isSavingPlan}
          saveStatusMessage={saveStatusMessage}
        />
      )}
    </main>
  );
}
