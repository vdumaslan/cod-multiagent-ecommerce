import { useEffect, useReducer, useRef, useState } from "react";
import DebateView from "./components/DebateView";
import PipelineStatusView from "./components/PipelineStatusView";
import QueryInputView from "./components/QueryInputView";
import ResultsView from "./components/ResultsView";

const API_BASE = "http://127.0.0.1:8010";
const pipelineAgents = ["Retrieval", "Sentiment", "Pricing", "Inventory"];
const initialPipelineState = pipelineAgents.map((name) => ({ name, status: "idle" }));

function pipelineReducer(state, action) {
  if (action.type === "RESET") {
    return initialPipelineState;
  }
  if (action.type === "SET_STATUS") {
    return state.map((agent) => (agent.name === action.agentName ? { ...agent, status: action.status } : agent));
  }
  return state;
}

async function postJson(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed with status ${res.status}`);
  }
  const data = await res.json();
  if (!data.ok) {
    throw new Error(data.error || `API ${path} returned not ok`);
  }
  return data.output;
}

const mockPlans = [
  {
    id: "plan-a",
    title: "Margin-First Bundle Expansion",
    actions: [
      "Launch 3 category-specific bundles with margin thresholds.",
      "Increase budget on top 20% ROAS campaigns by 15%.",
      "Run weekly inventory checks before promo pushes.",
    ],
    impactScore: 88,
    riskLevel: "Medium",
    confidence: 82,
  },
  {
    id: "plan-b",
    title: "Inventory-Protected Growth",
    actions: [
      "Gate promotions by 21-day stock coverage rule.",
      "Prioritize substitutes for low-coverage SKUs.",
      "Reduce discount depth for volatile demand categories.",
    ],
    impactScore: 79,
    riskLevel: "Low",
    confidence: 76,
  },
  {
    id: "plan-c",
    title: "Aggressive Revenue Sprint",
    actions: [
      "Double ad spend in best-performing channels for 2 weeks.",
      "Enable site-wide flash discount windows at peak traffic.",
      "Backfill risky SKUs through expedited suppliers.",
    ],
    impactScore: 91,
    riskLevel: "High",
    confidence: 67,
  },
];

function buildPlansFromSample(sample) {
  if (!sample || !Array.isArray(sample.baseline_actions)) {
    return mockPlans;
  }
  return sample.baseline_actions.slice(0, 3).map((action, idx) => {
    const productId = action.product_id || `unknown-${idx + 1}`;
    const retrievalScore = action?.evidence?.retrieval_score ?? 0.7;
    const returns = action?.signals?.total_returns ?? 0;
    const riskFlag = Boolean(action?.inventory?.risk_flag);
    const riskLevel = riskFlag ? "High" : returns > 3 ? "Medium" : "Low";
    const confidence = Math.max(55, Math.min(95, Math.round(Number(retrievalScore) * 100)));
    const impactScore = Math.max(50, Math.min(99, Math.round(confidence * 0.92)));
    return {
      id: productId,
      title: `Replay Plan for ${productId}`,
      actions: [
        `Action: ${action.action_type || "reprice"} (${Number(action.recommended_price_change_pct || 0).toFixed(2)}%)`,
        `Inventory status: ${action?.inventory?.stock_status || "unknown"}`,
        `Evidence score: ${Number(retrievalScore).toFixed(3)}`,
      ],
      impactScore,
      riskLevel,
      confidence,
    };
  });
}

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
  const [roundNumber, setRoundNumber] = useState(1);
  const [agentOutputs, setAgentOutputs] = useState(null);
  const [lastAdvocateOutput, setLastAdvocateOutput] = useState("");
  const [lastCriticOutput, setLastCriticOutput] = useState("");
  const [contextHistory, setContextHistory] = useState([]);
  const [selectedPlanId, setSelectedPlanId] = useState(null);
  const [isSavingPlan, setIsSavingPlan] = useState(false);
  const [saveStatusMessage, setSaveStatusMessage] = useState("");
  const [apiHealth, setApiHealth] = useState("checking");
  const [plans, setPlans] = useState(mockPlans);
  const debateTimersRef = useRef([]);
  const saveTimerRef = useRef(null);

  useEffect(() => {
    if (view !== "pipeline" || !query.trim()) {
      return undefined;
    }
    let cancelled = false;
    const runPipeline = async () => {
      try {
        dispatchPipeline({ type: "RESET" });
        dispatchPipeline({ type: "SET_STATUS", agentName: "Retrieval", status: "running" });
        const retrieval = await postJson("/ui/retrieval", { user_query: query.trim() });
        if (cancelled) return;
        dispatchPipeline({ type: "SET_STATUS", agentName: "Retrieval", status: "done" });

        ["Sentiment", "Pricing", "Inventory"].forEach((name) =>
          dispatchPipeline({ type: "SET_STATUS", agentName: name, status: "running" })
        );
        const [sentiment, pricing, inventory] = await Promise.all([
          postJson("/ui/sentiment", { user_query: query.trim(), retrieval_output: retrieval }),
          postJson("/ui/pricing", { user_query: query.trim(), retrieval_output: retrieval }),
          postJson("/ui/inventory", { user_query: query.trim(), retrieval_output: retrieval }),
        ]);
        if (cancelled) return;
        ["Sentiment", "Pricing", "Inventory"].forEach((name) =>
          dispatchPipeline({ type: "SET_STATUS", agentName: name, status: "done" })
        );
        setAgentOutputs({ userQuery: query.trim(), retrieval, sentiment, pricing, inventory });
        setPlans(buildPlansFromSample(retrieval.selected_sample));
        setView("debate");
      } catch {
        if (!cancelled) {
          setView("query");
          setIsLoading(false);
        }
      }
    };
    runPipeline();
    return () => {
      cancelled = true;
    };
  }, [view, query]);

  const clearDebateTimers = () => {
    debateTimersRef.current.forEach((timer) => clearTimeout(timer));
    debateTimersRef.current = [];
  };

  const clearSaveTimer = () => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
  };

  const buildRoundTurns = async (currentRound, contextText = "") => {
    if (!agentOutputs) {
      return [];
    }
    const advocate = await postJson("/ui/debate/advocate", {
      user_query: agentOutputs.userQuery,
      retrieval_output: agentOutputs.retrieval,
      sentiment_output: agentOutputs.sentiment,
      pricing_output: agentOutputs.pricing,
      inventory_output: agentOutputs.inventory,
      previous_advocate_output: lastAdvocateOutput,
      previous_critic_output: lastCriticOutput,
      added_context: contextText,
      round_number: currentRound,
    });
    const critic = await postJson("/ui/debate/critic", {
      user_query: agentOutputs.userQuery,
      retrieval_output: agentOutputs.retrieval,
      sentiment_output: agentOutputs.sentiment,
      pricing_output: agentOutputs.pricing,
      inventory_output: agentOutputs.inventory,
      advocate_output: advocate.message,
      previous_advocate_output: lastAdvocateOutput,
      previous_critic_output: lastCriticOutput,
      added_context: contextText,
      round_number: currentRound,
    });

    return [
      {
        id: `advocate-${currentRound}-${Date.now()}`,
        actor: "Advocate LLM",
        message: advocate.message,
      },
      {
        id: `critic-${currentRound}-${Date.now() + 1}`,
        actor: "Critic LLM",
        message: critic.message,
      },
    ];
  };

  const playDebateTurns = (turns, onDone) => {
    clearDebateTimers();
    setIsDebatePlaying(true);

    turns.forEach((turn, idx) => {
      const timer = setTimeout(() => {
        setDebateLog((prev) => [...prev, turn]);
        if (turn.actor === "Advocate LLM") {
          setLastAdvocateOutput(turn.message);
        }
        if (turn.actor === "Critic LLM") {
          setLastCriticOutput(turn.message);
        }
      }, idx * 1200);
      debateTimersRef.current.push(timer);
    });

    const doneTimer = setTimeout(() => {
      setIsDebatePlaying(false);
      if (onDone) {
        onDone();
      }
    }, turns.length * 1200);
    debateTimersRef.current.push(doneTimer);
  };

  useEffect(() => {
    if (view !== "debate") {
      clearDebateTimers();
      return undefined;
    }

    (async () => {
      const turns = await buildRoundTurns(1);
      if (turns.length) {
        playDebateTurns(turns);
      }
    })();
    setDebateLog([]);
    setCanViewResults(false);
    setContextDraft("");
    setIsAwaitingContext(false);
    setContextHistory([]);
    setRoundNumber(1);
    setLastAdvocateOutput("");
    setLastCriticOutput("");

    return () => {
      clearDebateTimers();
    };
  }, [view]);

  const handleSubmitQuery = () => {
    if (!query.trim()) {
      return;
    }

    setIsLoading(true);
    setView("pipeline");
    setTimeout(() => setIsLoading(false), 700);
  };

  const handleAnotherRound = async () => {
    if (isDebatePlaying || isAwaitingContext || canViewResults) {
      return;
    }
    const nextRound = roundNumber + 1;
    setRoundNumber(nextRound);
    const turns = await buildRoundTurns(nextRound);
    playDebateTurns(turns);
  };

  const handleAddContext = () => {
    if (isDebatePlaying || canViewResults) {
      return;
    }
    setIsAwaitingContext(true);
  };

  const handleSubmitContext = async () => {
    if (!contextDraft.trim() || isDebatePlaying || canViewResults) {
      return;
    }

    const userTurn = {
      id: `human-${Date.now()}`,
      actor: "Human Review",
      message: contextDraft.trim(),
    };
    setDebateLog((prev) => [...prev, userTurn]);
    setIsAwaitingContext(false);

    const nextRound = roundNumber + 1;
    setRoundNumber(nextRound);
    const submittedContext = contextDraft.trim();
    setContextHistory((prev) => [...prev, submittedContext]);
    setContextDraft("");
    const turns = await buildRoundTurns(nextRound, submittedContext);
    playDebateTurns(turns);
  };

  const handleMoveOn = async () => {
    if (isDebatePlaying || isAwaitingContext || canViewResults) {
      return;
    }

    const judge = await postJson("/ui/debate/judge", {
      user_query: agentOutputs.userQuery,
      retrieval_output: agentOutputs.retrieval,
      sentiment_output: agentOutputs.sentiment,
      pricing_output: agentOutputs.pricing,
      inventory_output: agentOutputs.inventory,
      latest_advocate_output: lastAdvocateOutput,
      latest_critic_output: lastCriticOutput,
      added_context_history: contextHistory,
    });
    const judgeTurn = {
      id: `judge-${Date.now()}`,
      actor: "Judge LLM",
      message: judge.message,
    };
    playDebateTurns([judgeTurn], () => setCanViewResults(true));
  };

  const handleRejectAll = () => {
    clearSaveTimer();
    clearDebateTimers();
    setSelectedPlanId(null);
    setQuery("");
    setDebateLog([]);
    setIsDebatePlaying(false);
    setIsAwaitingContext(false);
    setCanViewResults(false);
    setContextDraft("");
    setRoundNumber(1);
    setAgentOutputs(null);
    setLastAdvocateOutput("");
    setLastCriticOutput("");
    setContextHistory([]);
    setIsSavingPlan(false);
    setSaveStatusMessage("");
    setPlans(mockPlans);
    setView("query");
  };

  const handleChoosePlan = (planId) => {
    const chosenPlan = plans.find((plan) => plan.id === planId);
    if (!chosenPlan) {
      return;
    }

    clearSaveTimer();
    setSelectedPlanId(planId);
    setIsSavingPlan(true);
    setSaveStatusMessage(`Saving "${chosenPlan.title}" to BigQuery...`);

    postJson("/ui/save_plan", { plan_id: chosenPlan.id, title: chosenPlan.title }).then((saved) => {
      setSaveStatusMessage(`Saved "${chosenPlan.title}" to BigQuery (${saved.target}).`);
      saveTimerRef.current = setTimeout(() => {
        handleRejectAll();
      }, 1200);
    });
  };

  useEffect(() => {
    let cancelled = false;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) {
          throw new Error("health check failed");
        }
        if (!cancelled) {
          setApiHealth("online");
        }
      } catch {
        if (!cancelled) {
          setApiHealth("offline");
        }
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    return () => {
      clearSaveTimer();
    };
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <div className="mx-auto mb-8 max-w-6xl">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm uppercase tracking-[0.2em] text-cyan-400">Multi-Agent BI Dashboard</p>
          <span
            className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${
              apiHealth === "online"
                ? "bg-emerald-500/20 text-emerald-300"
                : apiHealth === "offline"
                ? "bg-rose-500/20 text-rose-300"
                : "bg-amber-500/20 text-amber-300"
            }`}
          >
            API {apiHealth}
          </span>
        </div>
        <p className="mt-2 text-slate-400">View flow: query → pipeline → debate → results</p>
      </div>

      {view === "query" && (
        <QueryInputView query={query} setQuery={setQuery} onSubmit={handleSubmitQuery} isLoading={isLoading} />
      )}
      {view === "pipeline" && <PipelineStatusView agents={pipeline} />}
      {view === "debate" && (
        <DebateView
          log={debateLog}
          isDebatePlaying={isDebatePlaying}
          isAwaitingContext={isAwaitingContext}
          contextDraft={contextDraft}
          setContextDraft={setContextDraft}
          canViewResults={canViewResults}
          onAddContext={handleAddContext}
          onAnotherRound={handleAnotherRound}
          onSubmitContext={handleSubmitContext}
          onCancelContext={() => {
            setIsAwaitingContext(false);
            setContextDraft("");
          }}
          onMoveOn={handleMoveOn}
          onViewResults={() => setView("results")}
        />
      )}
      {view === "results" && (
        <ResultsView
          plans={plans}
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
