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

async function getJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return await res.json();
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


function extractTitleFromEvidenceSnippet(snippet) {
  const s = String(snippet || "");
  if (!s) return "";
  const m = s.match(/(?:^|\n)title:\s*([^\n]+?)(?:\s+brand:|\n|$)/i);
  return (m?.[1] || "").trim();
}

function buildTopDrivers(a) {
  const drivers = [];
  const sent = a.sentiment || {};
  const inv = a.inventory || {};
  const sig = a.signals || {};

  const stock = String(inv.stock_status || "unknown");
  const risk = !!inv.risk_flag;
  const returns = Number(sig.total_returns || 0);
  const nReviews = Number(sent.n_reviews || 0);
  const pNeg = Number(sent.p_neg || 0);
  const pricingSource = String(a?.pricing?.source || "unknown");
  const pct = Number(a.recommended_price_change_pct || 0);
  const priceMissing = !!a?.pricing?.price_missing;

  if (stock && stock !== "healthy" && stock !== "unknown") {
    drivers.push(`Inventory: ${stock}${risk ? " (risk)" : ""}`);
  } else if (risk) {
    drivers.push("Inventory: risk_flag=true");
  }

  if (nReviews > 0 && pNeg >= 0.25) {
    drivers.push(`Sentiment: p_neg=${pNeg.toFixed(2)} (n=${nReviews})`);
  }
  if (returns >= 3) {
    drivers.push(`Returns: total_returns=${returns}`);
  }

  if (priceMissing || pricingSource === "fallback") {
    drivers.push("Price: unknown / pricing unavailable");
  } else if (a?.pricing?.near_bound) {
    drivers.push("Pricing: near bound (treat delta skeptically)");
  } else if (a?.pricing?.large_delta) {
    drivers.push("Pricing: large delta (treat delta skeptically)");
  } else if (Math.abs(pct) >= 5) {
    drivers.push(`Pricing delta: ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`);
  }

  const retrieval = Number(a.evidence?.retrieval_score ?? 0);
  drivers.push(`Retrieval similarity: ${Math.round(retrieval * 100)}%`);

  return Array.from(new Set(drivers)).slice(0, 3);
}

function buildPlansFromRanked(ranked) {
  if (!ranked || !ranked.length) return [];
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
    const retrieval = Number(a.evidence?.retrieval_score ?? 0);
    const retrievalSimilarity = Math.max(0, Math.min(100, Math.round(retrieval * 100)));
    const evidenceSnippet = a.evidence?.points?.[0]?.text || "";
    const productTitle = extractTitleFromEvidenceSnippet(evidenceSnippet);
    const suggestedAction = String(a.suggested_action || a.signals?.suggested_action || "").trim();
    const finalActionType = String(a.action_type || "reprice").trim();
    const topDrivers = buildTopDrivers(a);
    return {
      id: String(a.product_id),
      title: `${finalActionType} — ${productTitle || a.product_id}`,
      finalActionType,
      suggestedAction,
      retrievalSimilarity,
      retrievalScore: retrieval,
      evidenceSnippet: evidenceSnippet ? String(evidenceSnippet).slice(0, 240) : "",
      topDrivers,
      actions: [
        `SKU: ${a.product_id}`,
        `Pricing: ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}% (source=${a?.pricing?.source || "unknown"})`,
        (a?.pricing?.price_missing || a?.pricing?.source === "fallback") ? "Price missing/unknown: treat pricing rationale as low-confidence" : null,
        sent.n_reviews ? `Sentiment (n=${sent.n_reviews}): +${Number(sent.p_pos || 0).toFixed(2)} ~${Number(sent.p_neu || 0).toFixed(2)} -${Number(sent.p_neg || 0).toFixed(2)}` : "Sentiment: (n/a)",
        `Inventory: ${inv.stock_status || "unknown"}`,
        ...(a.llm_rationale_bullets || []).map((x) => `Rationale: ${x}`),
        ...(a.llm_risk_bullets || []).map((x) => `Risk: ${x}`),
      ].filter(Boolean).slice(0, 8),
      riskLevel,
      confidence: retrievalSimilarity,
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
  const [resultsMessage, setResultsMessage] = useState("");
  const [pipelineResult, setPipelineResult] = useState(null);
  const [llmRunningLabel, setLlmRunningLabel] = useState("");
  const [clarifyingQuestion, setClarifyingQuestion] = useState("");
  const [rewriteNotes, setRewriteNotes] = useState("");
  const [catalogSummary, setCatalogSummary] = useState(null);
  const [catalogFacets, setCatalogFacets] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedSubcategory, setSelectedSubcategory] = useState("");
  const [matchPreview, setMatchPreview] = useState(null);
  const [runContext, setRunContext] = useState(null);
  const [horizonDays, setHorizonDays] = useState(7);
  const [topNActions, setTopNActions] = useState(3);
  const [maxAbsPriceChangePct, setMaxAbsPriceChangePct] = useState(10);
  const [objective, setObjective] = useState("revenue");
  const [excludeLowStock, setExcludeLowStock] = useState(false);
  const [excludeStockoutRisk, setExcludeStockoutRisk] = useState(false);
  const [doNotRaiseIfPNegAbove, setDoNotRaiseIfPNegAbove] = useState("");

  // A/B testing
  const [abMode, setAbMode] = useState("B"); // mode used right now: "A" manual, "B" AI
  const [abVariant, setAbVariant] = useState(""); // assigned variant for analytics
  const [abId, setAbId] = useState(""); // stable browser identifier for assignment/logging
  const [showConfidence, setShowConfidence] = useState(false);
  const [versionAResults, setVersionAResults] = useState([]);
  const [versionADecisions, setVersionADecisions] = useState({});
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const formatElapsed = (s) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${String(sec).padStart(2, "0")}`;
  };

  const activeResultViews = ["version-a-results", "debate", "results"];
  useEffect(() => {
    if (!activeResultViews.includes(view) || showConfidence) {
      setElapsedSeconds(0);
      return undefined;
    }
    setElapsedSeconds(sessionStartRef.current ? Math.round((Date.now() - sessionStartRef.current) / 1000) : 0);
    const id = setInterval(() => {
      setElapsedSeconds(sessionStartRef.current ? Math.round((Date.now() - sessionStartRef.current) / 1000) : 0);
    }, 1000);
    return () => clearInterval(id);
  }, [view, showConfidence]); // eslint-disable-line react-hooks/exhaustive-deps

  const postAbEvent = (event, metadata = {}) => {
    fetch(`${API_BASE}/ab/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        owner_id: abId || DEFAULT_OWNER_ID,
        variant: abVariant || "",
        run_id: pipelineResult?.run_id || "",
        event,
        metadata: { ...metadata, mode_used: abMode, assigned_variant: abVariant || "" },
      }),
    }).catch(() => {});
  };

  const parseOptionalFloat = (v) => {
    const s = String(v ?? "").trim();
    if (!s) return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  };

  // Debate state across N rounds
  const [roundNumber, setRoundNumber] = useState(1);
  const [latestAdvocate, setLatestAdvocate] = useState(null);
  const [latestCritic, setLatestCritic] = useState(null);

  const debateTimersRef = useRef([]);
  const saveTimerRef = useRef(null);
  const sessionStartRef = useRef(null);
  const pendingChosenPlanRef = useRef(null);

  // ── Health check ────────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        const allReady = data.has_pricing_cache && data.has_sentiment_cache && data.has_inventory_cache && data.has_retrieval_index;
        if (!cancelled) setApiHealth(allReady ? "online" : "degraded");
      } catch {
        if (!cancelled) setApiHealth("offline");
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    // True A/B needs a stable identifier; use a browser-scoped visitor id.
    try {
      const key = "copilot_ab_visitor_id";
      let vid = window.localStorage.getItem(key);
      if (!vid) {
        vid = `visitor_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
        window.localStorage.setItem(key, vid);
      }
      setAbId(vid);
      getJson(`/ab/variant/${encodeURIComponent(vid)}`)
        .then((d) => {
          const v = d.variant || "";
          setAbVariant(v);
          // Default to assigned mode for convenience (user can still switch freely).
          if (v) setAbMode(v);
          fetch(`${API_BASE}/ab/event`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              owner_id: vid,
              variant: v,
              run_id: "",
              event: "session_start",
              metadata: { mode_used: v || abMode, assigned_variant: v },
            }),
          }).catch(() => {});
        })
        .catch(() => {});
    } catch {
      // Fallback: keep DEFAULT_OWNER_ID (demo-only)
      getJson(`/ab/variant/${DEFAULT_OWNER_ID}`).then((d) => setAbVariant(d.variant || "")).catch(() => {});
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let cancelled = false;
    getJson("/catalog/summary")
      .then((d) => {
        if (!cancelled && d && d.ok) setCatalogSummary(d);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getJson("/catalog/facets")
      .then((d) => {
        if (!cancelled && d && d.ok) setCatalogFacets(d);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Debounced match preview (retrieval-only)
  useEffect(() => {
    const scopeHint = [selectedCategory, selectedSubcategory].filter(Boolean).join(" / ");
    const goal = query.trim();
    const scopedGoalForPreview = scopeHint ? `${goal} (scope: ${scopeHint})` : goal;
    if (!goal) {
      setMatchPreview(null);
      return undefined;
    }
    const t = setTimeout(() => {
      postJson("/retrieval/preview", {
        goal: scopedGoalForPreview,
        top_k_preview: 5,
        constraints: {
          max_abs_price_change_pct: maxAbsPriceChangePct,
          objective,
          exclude_low_stock: excludeLowStock,
          exclude_stockout_risk: excludeStockoutRisk,
          do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
        },
      })
        .then((d) => setMatchPreview(d))
        .catch(() => {});
    }, 400);
    return () => clearTimeout(t);
  }, [
    query,
    selectedCategory,
    selectedSubcategory,
    maxAbsPriceChangePct,
    objective,
    excludeLowStock,
    excludeStockoutRisk,
    doNotRaiseIfPNegAbove,
  ]);

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
    setResultsMessage("");
    setClarifyingQuestion("");
    setRewriteNotes("");
    dispatchPipeline({ type: "RESET" });
    setView("pipeline");

    const scopeHint = [selectedCategory, selectedSubcategory].filter(Boolean).join(" / ");
    const scopedGoal = scopeHint ? `${query.trim()} (scope: ${scopeHint})` : query.trim();
    setRunContext({
      goal: query.trim(),
      scoped_goal: scopedGoal,
      selected_category: selectedCategory || "",
      selected_subcategory: selectedSubcategory || "",
      objective,
      horizon_days: horizonDays,
      top_n_actions: topNActions,
      max_abs_price_change_pct: maxAbsPriceChangePct,
      exclude_low_stock: excludeLowStock,
      exclude_stockout_risk: excludeStockoutRisk,
      do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
      retrieval_min_score: matchPreview?.min_score ?? null,
      retrieval_query_preview: matchPreview?.retrieval_query ?? "",
    });

    postJson("/pipeline", {
      goal: scopedGoal,
      owner_id: DEFAULT_OWNER_ID,
      horizon_days: horizonDays,
      top_n_actions: topNActions,
      constraints: {
        max_abs_price_change_pct: maxAbsPriceChangePct,
        objective,
        exclude_low_stock: excludeLowStock,
        exclude_stockout_risk: excludeStockoutRisk,
        do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
      },
      enable_pricing: true,
      enable_sentiment: true,
    })
      .then((data) => setPipelineResult(data))
      .catch(() => { setView("query"); setIsLoading(false); });
  };

  // ── Version A: manual decision flow ───────────────────────────────────────

  const handleSubmitQueryVersionA = () => {
    if (!query.trim()) return;
    setIsLoading(true);
    // Build a concrete product-oriented query from category/subcategory so the
    // query rewrite doesn't treat it as a vague business goal and ask for clarification.
    const categoryTerm = selectedSubcategory || selectedCategory || "";
    const retrievalGoal = categoryTerm
      ? `${categoryTerm} products`
      : query.trim();
    postJson("/retrieval/preview", {
      goal: retrievalGoal,
      top_k_preview: 5,
      constraints: { max_abs_price_change_pct: 10, objective: "revenue" },
    })
      .then((data) => {
        if (data.clarifying_question && !(data.top_candidates || []).length) {
          setClarifyingQuestion(data.clarifying_question);
          setIsLoading(false);
          return;
        }
        setClarifyingQuestion("");
        const candidates = data.top_candidates || [];
        setVersionAResults(candidates);
        const initial = {};
        candidates.forEach((c) => { initial[c.product_id] = { action: "reprice", priceChangePct: "0" }; });
        setVersionADecisions(initial);
        setIsLoading(false);
        sessionStartRef.current = Date.now();
        setView("version-a-results");
      })
      .catch(() => setIsLoading(false));
  };

  const handleVersionASubmitDecision = () => {
    const elapsed = sessionStartRef.current ? Math.round((Date.now() - sessionStartRef.current) / 1000) : null;
    postAbEvent("decision_made", { mode: "A", time_to_decision_s: elapsed, decisions: versionADecisions });
    pendingChosenPlanRef.current = "version-a";
    setShowConfidence(true);
  };

  const handleConfidenceRate = (rating) => {
    postAbEvent("confidence_rated", { rating, mode: abMode });
    setShowConfidence(false);
    const pending = pendingChosenPlanRef.current;
    pendingChosenPlanRef.current = null;
    if (pending === "version-a") {
      const elapsed = sessionStartRef.current ? Math.round((Date.now() - sessionStartRef.current) / 1000) : null;
      fetch(`${API_BASE}/ab/save_run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          owner_id: DEFAULT_OWNER_ID,
          variant: abVariant || abMode,
          goal: query,
          retrieval_candidates: versionAResults,
          decisions: versionADecisions,
          time_to_decision_s: elapsed,
          confidence_rating: rating,
        }),
      }).catch(() => {});
      setVersionAResults([]);
      setVersionADecisions({});
      setQuery("");
      setIsLoading(false);
      setView("query");
    } else if (pending) {
      // continue with Version B save animation
      const chosen = plans.find((p) => p.id === pending);
      if (!chosen) return;
      setSelectedPlanId(pending);
      setIsSavingPlan(true);
      setSaveStatusMessage(`Saving "${chosen.title}"...`);
      saveTimerRef.current = setTimeout(() => {
        setSaveStatusMessage(`Saved "${chosen.title}".`);
        saveTimerRef.current = setTimeout(handleRejectAll, 1200);
      }, 1000);
    }
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

    const cq = pipelineResult?.trace?.query_rewrite?.clarifying_question;
    const rq = pipelineResult?.trace?.query_rewrite?.retrieval_query;
    const used = !!pipelineResult?.trace?.query_rewrite?.used;
    const notes = pipelineResult?.trace?.query_rewrite?.notes;
    const shouldAsk =
      used && typeof cq === "string" && cq.trim().length > 0 && (!rq || String(rq).trim().length === 0);

    const t = setTimeout(() => {
      if (shouldAsk) {
        setClarifyingQuestion(String(cq).trim());
        setRewriteNotes(typeof notes === "string" ? notes.trim() : "");
        setPipelineResult(null);
        setIsLoading(false);
        setView("query");
        return;
      }
      setIsLoading(false);
      sessionStartRef.current = Date.now();
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
      run_id: pipelineResult?.run_id || undefined,
      top_n_actions: topNActions,
      constraints: {
        max_abs_price_change_pct: maxAbsPriceChangePct,
        objective,
        exclude_low_stock: excludeLowStock,
        exclude_stockout_risk: excludeStockoutRisk,
        do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
      },
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
        top_n_actions: topNActions,
        constraints: {
          max_abs_price_change_pct: maxAbsPriceChangePct,
          objective,
          exclude_low_stock: excludeLowStock,
          exclude_stockout_risk: excludeStockoutRisk,
          do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
        },
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
        top_n_actions: topNActions,
        constraints: {
          max_abs_price_change_pct: maxAbsPriceChangePct,
          objective,
          exclude_low_stock: excludeLowStock,
          exclude_stockout_risk: excludeStockoutRisk,
          do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
        },
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
        run_id: pipelineResult?.run_id || undefined,
        top_n_actions: topNActions,
        constraints: {
          max_abs_price_change_pct: maxAbsPriceChangePct,
          objective,
          exclude_low_stock: excludeLowStock,
          exclude_stockout_risk: excludeStockoutRisk,
          do_not_raise_if_p_neg_above: parseOptionalFloat(doNotRaiseIfPNegAbove),
        },
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
        const built = buildPlansFromRanked(data.ranked_actions);
        setPlans(built);
        if (!built.length) {
          setResultsMessage(
            "No ranked plans were returned for this query. This can happen when retrieval finds no strong matches (or filters remove all candidates). Try adding more product/category detail to your goal, or relax constraints."
          );
        }
        setCanViewResults(true);
      });
    } catch {
      setLlmRunningLabel("");
      addErrorTurn("Judge failed — LLM may be unavailable. Try again.");
    }
  };

  // ── Step 4: results ─────────────────────────────────────────────────────────

  const handleRejectAll = () => {
    if (view === "results" && !selectedPlanId) {
      const elapsed = sessionStartRef.current ? Math.round((Date.now() - sessionStartRef.current) / 1000) : null;
      postAbEvent("abandoned", { mode: "B", time_to_decision_s: elapsed });
    }
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    clearDebateTimers();
    setSelectedPlanId(null);
    setVersionAResults([]);
    setVersionADecisions({});
    setQuery("");
    setDebateLog([]);
    setIsDebatePlaying(false);
    setIsAwaitingContext(false);
    setCanViewResults(false);
    setContextDraft("");
    setIsSavingPlan(false);
    setSaveStatusMessage("");
    setPlans([]);
    setResultsMessage("");
    setPipelineResult(null);
    setRunContext(null);
    setLatestAdvocate(null);
    setLatestCritic(null);
    setRoundNumber(1);
    setIsLoading(false);
    setView("query");
  };

  const handleChoosePlan = (planId) => {
    const chosen = plans.find((p) => p.id === planId);
    if (!chosen) return;
    const elapsed = sessionStartRef.current ? Math.round((Date.now() - sessionStartRef.current) / 1000) : null;
    postAbEvent("decision_made", { mode: "B", time_to_decision_s: elapsed, plan_id: planId });
    pendingChosenPlanRef.current = planId;
    setShowConfidence(true);
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <div className="sticky top-0 z-20 -mx-6 mb-8 border-b border-slate-900 bg-slate-950/90 px-6 py-4 backdrop-blur">
        <div className="mx-auto max-w-6xl">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-[220px]">
              <p className="text-sm uppercase tracking-[0.2em] text-cyan-400">Multi-Agent BI Dashboard</p>
              <p className="mt-1 text-xs text-slate-500">
                {abMode === "A" ? "Manual (A): retrieval only, you decide actions." : "AI Copilot (B): rewrite + enrichment + debate + judge."}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center rounded-xl border border-slate-800 bg-slate-950/40 p-1">
                <button
                  type="button"
                  onClick={() => {
                    setAbMode("A");
                    setView("query");
                    setVersionAResults([]);
                    setVersionADecisions({});
                  }}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    abMode === "A" ? "bg-slate-100 text-slate-900" : "text-slate-300 hover:bg-slate-900"
                  }`}
                >
                  Manual (A)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAbMode("B");
                    setView("query");
                    setVersionAResults([]);
                    setVersionADecisions({});
                  }}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    abMode === "B" ? "bg-cyan-400 text-slate-950" : "text-slate-300 hover:bg-slate-900"
                  }`}
                >
                  AI Copilot (B)
                </button>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-2 text-xs text-slate-300">
                <span className="text-slate-500">Assigned:</span>{" "}
                <span className="font-semibold text-slate-100">{abVariant || "?"}</span>
                <span className="text-slate-500"> · id:</span>{" "}
                <span className="font-mono text-slate-200">{(abId || DEFAULT_OWNER_ID).slice(0, 18)}</span>
              </div>

              {activeResultViews.includes(view) && (
                <span className="rounded-full border border-slate-700 bg-slate-900/40 px-3 py-1 font-mono text-sm text-cyan-300">
                  ⏱ {formatElapsed(elapsedSeconds)}
                </span>
              )}

              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${
                  apiHealth === "online" ? "bg-emerald-500/20 text-emerald-300"
                  : apiHealth === "offline" ? "bg-rose-500/20 text-rose-300"
                  : apiHealth === "degraded" ? "bg-orange-500/20 text-orange-300"
                  : "bg-amber-500/20 text-amber-300"
                }`}
              >
                API {apiHealth}
              </span>
            </div>
          </div>
        </div>
      </div>

      {view === "query" && abMode === "B" && (
        <QueryInputView
          query={query}
          setQuery={setQuery}
          onSubmit={handleSubmitQuery}
          isLoading={isLoading}
          clarifyingQuestion={clarifyingQuestion}
          rewriteNotes={rewriteNotes}
          catalogSummary={catalogSummary}
          catalogFacets={catalogFacets}
          selectedCategory={selectedCategory}
          setSelectedCategory={setSelectedCategory}
          selectedSubcategory={selectedSubcategory}
          setSelectedSubcategory={setSelectedSubcategory}
          matchPreview={matchPreview}
          horizonDays={horizonDays}
          setHorizonDays={setHorizonDays}
          topNActions={topNActions}
          setTopNActions={setTopNActions}
          maxAbsPriceChangePct={maxAbsPriceChangePct}
          setMaxAbsPriceChangePct={setMaxAbsPriceChangePct}
          objective={objective}
          setObjective={setObjective}
          excludeLowStock={excludeLowStock}
          setExcludeLowStock={setExcludeLowStock}
          excludeStockoutRisk={excludeStockoutRisk}
          setExcludeStockoutRisk={setExcludeStockoutRisk}
          doNotRaiseIfPNegAbove={doNotRaiseIfPNegAbove}
          setDoNotRaiseIfPNegAbove={setDoNotRaiseIfPNegAbove}
        />
      )}

      {view === "query" && abMode === "A" && (
        <div className="mx-auto max-w-2xl space-y-6">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">What are you trying to improve?</h2>
            <p className="mt-1 text-sm text-slate-400">Describe your goal in plain language.</p>
          </div>
          <textarea
            className="w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-slate-100 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
            rows={3}
            placeholder="e.g. increase revenue for kitchen products"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmitQueryVersionA(); } }}
          />
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-widest text-slate-400">Category</label>
              <select
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                value={selectedCategory}
                onChange={(e) => { setSelectedCategory(e.target.value); setSelectedSubcategory(""); }}
              >
                <option value="">(Any)</option>
                {(catalogFacets?.categories || []).map((c) => (
                  <option key={c.name} value={c.name}>{c.name} ({c.count})</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-widest text-slate-400">Subcategory</label>
              <select
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                value={selectedSubcategory}
                onChange={(e) => setSelectedSubcategory(e.target.value)}
                disabled={!selectedCategory}
              >
                <option value="">(Any)</option>
                {(catalogFacets?.subcategories_by_category?.[selectedCategory] || []).map((s) => (
                  <option key={s.name} value={s.name}>{s.name} ({s.count})</option>
                ))}
              </select>
            </div>
          </div>
          {clarifyingQuestion && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              <span className="font-semibold">Needs more detail: </span>{clarifyingQuestion}
            </div>
          )}
          {!selectedCategory && query.trim() && (
            <p className="text-xs text-slate-400">Select a category to get better results.</p>
          )}
          <button
            onClick={handleSubmitQueryVersionA}
            disabled={!query.trim() || isLoading}
            className="w-full rounded-xl bg-cyan-600 py-3 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-40"
          >
            {isLoading ? "Loading products..." : "Find Products"}
          </button>
        </div>
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
          plans={plans}
          resultsMessage={resultsMessage}
          runContext={runContext}
          selectedPlanId={selectedPlanId}
          onChoosePlan={handleChoosePlan}
          onRejectAll={handleRejectAll}
          isSavingPlan={isSavingPlan}
          saveStatusMessage={saveStatusMessage}
        />
      )}
      {view === "version-a-results" && (
        <div className="mx-auto max-w-4xl space-y-4">
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-slate-200">Products matching your goal</h2>
            <p className="text-sm text-slate-400">Review the list and assign actions manually.</p>
          </div>
          {versionAResults.length === 0 && (
            <p className="text-slate-400">No products found. Try a different goal.</p>
          )}
          {versionAResults.map((c) => {
            const dec = versionADecisions[c.product_id] || { action: "reprice", priceChangePct: "0" };
            return (
              <div key={c.product_id} className="rounded-xl border border-slate-700 bg-slate-900 p-4">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="font-medium text-slate-100">{c.title || c.product_id}</p>
                    <p className="text-xs text-slate-400">{c.category}{c.subcategory ? ` › ${c.subcategory}` : ""}</p>
                    <p className="mt-1 text-xs text-slate-500">Match score: {Math.round(c.score * 100)}%</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <select
                      value={dec.action}
                      onChange={(e) => setVersionADecisions((prev) => ({ ...prev, [c.product_id]: { ...prev[c.product_id], action: e.target.value } }))}
                      className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-sm text-slate-200"
                    >
                      <option value="reprice">Reprice</option>
                      <option value="restock">Restock</option>
                      <option value="promote">Promote</option>
                      <option value="hold">Hold</option>
                      <option value="investigate">Investigate</option>
                    </select>
                    {dec.action === "reprice" && (
                      <input
                        type="number"
                        value={dec.priceChangePct}
                        onChange={(e) => setVersionADecisions((prev) => ({ ...prev, [c.product_id]: { ...prev[c.product_id], priceChangePct: e.target.value } }))}
                        className="w-24 rounded border border-slate-600 bg-slate-800 px-2 py-1 text-sm text-slate-200"
                        placeholder="% change"
                      />
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          <div className="flex gap-3 pt-2">
            <button
              onClick={handleVersionASubmitDecision}
              disabled={versionAResults.length === 0}
              className="rounded-lg bg-cyan-600 px-6 py-2 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-40"
            >
              Submit Decision
            </button>
            <button
              onClick={handleRejectAll}
              className="rounded-lg border border-slate-600 px-6 py-2 text-sm text-slate-300 hover:bg-slate-800"
            >
              Start Over
            </button>
          </div>
        </div>
      )}

      {showConfidence && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="w-80 rounded-2xl bg-slate-800 p-8 text-center shadow-2xl">
            <p className="mb-1 text-lg font-semibold text-slate-100">How confident are you?</p>
            <p className="mb-6 text-sm text-slate-400">Rate your confidence in this decision</p>
            <div className="flex justify-center gap-3">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  onClick={() => handleConfidenceRate(n)}
                  className="flex h-10 w-10 items-center justify-center rounded-full border border-slate-600 text-sm font-bold text-slate-300 hover:border-cyan-400 hover:bg-cyan-600/20 hover:text-cyan-300"
                >
                  {n}
                </button>
              ))}
            </div>
            <p className="mt-4 text-xs text-slate-500">1 = Not confident · 5 = Very confident</p>
          </div>
        </div>
      )}
    </main>
  );
}
