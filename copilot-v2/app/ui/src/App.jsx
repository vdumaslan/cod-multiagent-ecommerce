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

function stripMarkdown(text) {
  return String(text || "")
    .replace(/#{1,6}\s*/g, "")          // remove heading markers
    .replace(/\*\*([^*]+)\*\*/g, "$1")  // bold
    .replace(/\*([^*]+)\*/g, "$1")      // italic
    .replace(/__([^_]+)__/g, "$1")      // bold alt
    .replace(/_([^_]+)_/g, "$1")        // italic alt
    .replace(/`([^`]+)`/g, "$1")        // inline code
    .replace(/^\s*[-*]\s+/gm, "• ")     // list markers → bullet
    .replace(/\n{3,}/g, "\n\n")         // collapse excess blank lines
    .trim();
}

function cleanTechnicalText(text, maxLen = 300) {
  let s = String(text || "");

  // Shorten raw product IDs: B0XXXXXXXX → …XXXXXX
  s = s.replace(/\b(B0[A-Z0-9]{8,})\b/g, (m) => `…${m.slice(-6)}`);
  // Remove redundant "product_id=…XXXXXX" fragments
  s = s.replace(/\(product_id=[^)]+\)/gi, "");

  // Translate common JSON field names → plain English
  s = s.replace(/p_neg\s*=\s*([\d.]+)/gi, (_, v) => `${Math.round(parseFloat(v) * 100)}% negative sentiment`);
  s = s.replace(/p_pos\s*=\s*([\d.]+)/gi, (_, v) => `${Math.round(parseFloat(v) * 100)}% positive`);
  s = s.replace(/n_?reviews\s*=\s*(\d+)/gi, (_, v) => `${v} reviews`);
  s = s.replace(/total_?returns\s*=\s*(\d+)/gi, (_, v) => `${v} returns`);
  s = s.replace(/recommended_?price_?change_?pct\s*=\s*([-\d.]+)/gi, (_, v) => {
    const n = parseFloat(v);
    if (Math.abs(n) < 0.01) return "price change: 0%";
    return `price change: ${n > 0 ? "+" : ""}${n.toFixed(2)}%`;
  });
  s = s.replace(/large_delta\s*=\s*true/gi, "large price delta");
  s = s.replace(/large_delta\s*=\s*false/gi, "normal price delta");
  s = s.replace(/near_bound\s*=\s*true/gi, "near policy cap");
  s = s.replace(/risk_flag\s*=\s*true/gi, "inventory risk flagged");
  s = s.replace(/action_type\s*(?:to|=)\s*(\w+)/gi, (_, t) => t);
  s = s.replace(/stock_status\s*=\s*(\w+)/gi, (_, v) => `stock: ${v}`);
  // Clean up double spaces left by removals
  s = s.replace(/\s{2,}/g, " ").trim();

  return s.length > maxLen ? s.slice(0, maxLen - 1) + "…" : s;
}

function formatAdvocate(adv, round) {
  const prefix = round > 1 ? `[Round ${round}] ` : "";
  const parts = [];

  // Lead with key claims — clean and truncate product IDs / raw fields
  const claims = (adv.key_claims || []).slice(0, 3)
    .map((c) => `• ${cleanTechnicalText(c, 160)}`)
    .join("\n");
  if (claims) parts.push(`${prefix}Advocate's position:\n${claims}`);

  // Proposed actions — human-readable direction
  const actions = (adv.proposed_actions || []).slice(0, 3)
    .map((a) => {
      const pct = Number(a.recommended_price_change_pct || 0);
      const type = String(a.action_type || "reprice");
      let dir = type;
      if (type === "reprice") dir = Math.abs(pct) < 0.01 ? "hold price" : (pct > 0 ? `increase price +${pct.toFixed(1)}%` : `decrease price ${pct.toFixed(1)}%`);
      return `• ${dir} (…${String(a.product_id).slice(-6)})`;
    })
    .join("\n");
  if (actions) parts.push(`Proposed:\n${actions}`);

  // Concerns — clean product IDs
  const concerns = (adv.concerns || []).slice(0, 2)
    .map((c) => `• ${cleanTechnicalText(c, 160)}`)
    .join("\n");
  if (concerns) parts.push(`Concerns:\n${concerns}`);

  return parts.join("\n\n") || "No advocate output.";
}

function formatCritic(crit, round) {
  const prefix = round > 1 ? `[Round ${round}] ` : "";
  const parts = [];

  // Lead with disagreements (most actionable for sellers)
  const disagreements = (crit.disagreements || []).slice(0, 3)
    .map((x) => `• ${cleanTechnicalText(x, 180)}`);
  if (disagreements.length) {
    parts.push(`${prefix}Critic challenges:\n${disagreements.join("\n")}`);
  }

  const changes = (crit.suggested_changes || []).slice(0, 2)
    .map((x) => `• ${cleanTechnicalText(x, 140)}`);
  if (changes.length) parts.push(`Suggested changes:\n${changes.join("\n")}`);

  const agreements = (crit.agreements || []).slice(0, 2)
    .map((x) => `• ${cleanTechnicalText(x, 140)}`);
  if (agreements.length) parts.push(`Agreed on:\n${agreements.join("\n")}`);

  return parts.join("\n\n") || "No critic output.";
}

function advCritTurns(adv, crit, round) {
  const now = Date.now();
  return [
    { id: `adv-r${round}-${now}`, actor: "Advocate LLM", message: formatAdvocate(adv, round) },
    { id: `crit-r${round}-${now + 1}`, actor: "Critic LLM", message: formatCritic(crit, round) },
  ];
}

// ── Portfolio summary ─────────────────────────────────────────────────────────

function computePortfolioSummary(enrichedCandidates, objective) {
  if (!enrichedCandidates || !enrichedCandidates.length) return null;
  const n = enrichedCandidates.length;
  const actionCounts = { reprice: 0, hold: 0, investigate: 0, promote: 0, restock: 0 };
  let highNegSentiment = 0, inventoryRisk = 0, pricingMissing = 0, highReturns = 0;
  let raiseCount = 0, lowerCount = 0;

  for (const c of enrichedCandidates) {
    const action = String(c.suggested_action || "reprice").toLowerCase();
    actionCounts[action] = (actionCounts[action] || 0) + 1;

    if ((c.sentiment?.p_neg || 0) >= 0.30) highNegSentiment++;
    if (c.inventory?.risk_flag || ["low_stock", "stockout_risk"].includes(c.inventory?.stock_status)) inventoryRisk++;
    if (c.pricing?.price_missing || c.pricing?.source === "fallback") pricingMissing++;
    if ((c.signals?.total_returns || 0) >= 3) highReturns++;

    const pct = Number(c.recommended_price_change_pct || 0);
    if (action === "reprice" && pct > 0.001) raiseCount++;
    if (action === "reprice" && pct < -0.001) lowerCount++;
  }

  const obj = String(objective || "revenue").toLowerCase();
  let strategy = "";
  if (inventoryRisk / n >= 0.4) {
    strategy = "Inventory risk is elevated across many candidates. Prioritize HOLD and restock actions; avoid aggressive price increases.";
  } else if (highNegSentiment / n >= 0.3) {
    strategy = "Negative sentiment is widespread. Focus on investigation and quality improvements before repricing.";
  } else if (obj === "clear_inventory") {
    strategy = "Most candidates are eligible for promotion or price reduction to move surplus inventory.";
  } else if (obj === "reduce_returns") {
    strategy = "Elevated returns present. Prioritize investigation on high-return SKUs before price changes.";
  } else if (raiseCount > lowerCount && raiseCount > 0) {
    strategy = "Most candidates show upward pricing opportunity with manageable risk. Focus on selective price increases on healthy SKUs.";
  } else {
    strategy = "Portfolio is mixed. Review individual product signals before applying blanket pricing changes.";
  }

  return { n, actionCounts, highNegSentiment, inventoryRisk, pricingMissing, highReturns, strategy };
}

// ── Plan building ─────────────────────────────────────────────────────────────


function extractTitleFromEvidenceSnippet(snippet) {
  const s = String(snippet || "");
  if (!s) return "";
  const m = s.match(/(?:^|\n)title:\s*([^\n]+?)(?:\s+brand:|\n|$)/i);
  return (m?.[1] || "").trim();
}

function extractPriceFromDoc(docText) {
  // Matches "price_usd: 24.99" or "price: 24.99"
  const m = String(docText || "").match(/(?:^|\n)price(?:_usd)?:\s*([\d.]+)/i);
  return m ? parseFloat(m[1]) : null;
}

function buildPricingLine(a, pct, finalActionType) {
  const pr = a?.pricing || {};
  const doc = a?.evidence?.points?.[0]?.text || "";
  const currentPrice = extractPriceFromDoc(doc);

  if (pr.price_missing || pr.source === "fallback") {
    return { text: "Current price unavailable", icon: "💰", dim: true };
  }

  // Actions that don't involve repricing
  if (["hold", "restock", "promote", "investigate"].includes(finalActionType)) {
    if (currentPrice != null) {
      return { text: `Current price: $${currentPrice.toFixed(2)} — no change recommended`, icon: "💰", dim: true };
    }
    return { text: "No price change recommended", icon: "💰", dim: true };
  }

  // Reprice with zero delta
  if (Math.abs(pct) < 0.01) {
    if (currentPrice != null) {
      return { text: `Current price: $${currentPrice.toFixed(2)} — no change`, icon: "💰", dim: true };
    }
    return { text: "No price change suggested", icon: "💰", dim: true };
  }

  // Reprice with non-zero delta
  const direction = pct > 0 ? "+" : "";
  if (currentPrice != null) {
    const newPrice = currentPrice * (1 + pct / 100);
    return {
      text: `Suggested: ${direction}${pct.toFixed(1)}%  ·  $${currentPrice.toFixed(2)} → $${newPrice.toFixed(2)}`,
      icon: "💰",
      dim: false,
    };
  }
  return {
    text: `Suggested price change: ${direction}${pct.toFixed(1)}%`,
    icon: "💰",
    dim: false,
  };
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

function actionLabel(actionType, pct) {
  const type = String(actionType || "reprice").trim().toLowerCase();
  if (type === "reprice") {
    if (pct > 0.001) return `Increase price by ${pct.toFixed(2)}%`;
    if (pct < -0.001) return `Decrease price by ${Math.abs(pct).toFixed(2)}%`;
    return "Hold price (0% change)";
  }
  if (type === "investigate") return "Investigate before changing price";
  if (type === "hold") return "Hold";
  if (type === "restock") return "Restock";
  if (type === "promote") return "Promote / increase visibility";
  return type.charAt(0).toUpperCase() + type.slice(1);
}

const ACTION_BADGE_COLORS = {
  reprice: "bg-cyan-500/20 text-cyan-200 border-cyan-500/40",
  investigate: "bg-amber-500/20 text-amber-200 border-amber-500/40",
  hold: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  restock: "bg-violet-500/20 text-violet-200 border-violet-500/40",
  promote: "bg-emerald-500/20 text-emerald-200 border-emerald-500/40",
};

function computeConfidence(a, pct) {
  const pr = a?.pricing || {};
  const sent = a?.sentiment || {};
  const retrieval = Number(a?.evidence?.retrieval_score ?? 0);
  const nRev = Number(sent?.n_reviews || 0);
  const pNeg = Number(sent?.p_neg || 0);

  // Low confidence conditions (any one is enough)
  if (pr.price_missing || pr.source === "fallback") return { level: "Low", color: "text-rose-300", bg: "bg-rose-500/15 border-rose-500/30" };
  if (retrieval < 0.80) return { level: "Low", color: "text-rose-300", bg: "bg-rose-500/15 border-rose-500/30" };
  if (nRev < 5 && nRev > 0) return { level: "Low", color: "text-rose-300", bg: "bg-rose-500/15 border-rose-500/30" };
  if (nRev === 0) return { level: "Low", color: "text-rose-300", bg: "bg-rose-500/15 border-rose-500/30" };

  // Medium confidence: have data but signals are mixed or uncertain
  if (pr.near_bound || pr.large_delta) return { level: "Medium", color: "text-amber-300", bg: "bg-amber-500/15 border-amber-500/30" };
  if (pNeg >= 0.25) return { level: "Medium", color: "text-amber-300", bg: "bg-amber-500/15 border-amber-500/30" };
  if (retrieval < 0.85) return { level: "Medium", color: "text-amber-300", bg: "bg-amber-500/15 border-amber-500/30" };

  // High: good retrieval, pricing cached and not near bound, reasonable sentiment
  return { level: "High", color: "text-emerald-300", bg: "bg-emerald-500/15 border-emerald-500/30" };
}

function buildWhyBullets(a, pct, finalActionType) {
  // Return exactly 1 primary reason + at most 1 supporting/cautionary note.
  const pr = a?.pricing || {};
  const sent = a?.sentiment || {};
  const inv = a?.inventory || {};
  const sig = a?.signals || {};
  const nRev = Number(sent?.n_reviews || 0);
  const pNeg = Number(sent?.p_neg || 0);
  const returns = Number(sig?.total_returns || 0);
  const stock = String(inv?.stock_status || "unknown");
  const primary = [];
  const secondary = [];

  if (finalActionType === "restock") {
    primary.push("Inventory is critically low — restocking prevents lost sales opportunities.");
    return primary;
  }

  if (finalActionType === "hold") {
    if (stock === "low_stock" || inv?.risk_flag) primary.push("Inventory risk detected — holding price avoids adding demand pressure to a supply-constrained SKU.");
    else primary.push("Multiple risk signals outweigh the pricing opportunity — holding is the safer path.");
    return primary;
  }

  if (finalActionType === "investigate") {
    if (returns >= 3) primary.push(`${returns} returns logged — root cause needs investigation before any price change.`);
    else if (pNeg >= 0.30) primary.push(`${Math.round(pNeg * 100)}% of ${nRev} reviews are negative — understand quality issues before repricing.`);
    else if (pr.price_missing) primary.push("Pricing data is unavailable — cannot form a confident recommendation without it.");
    else if (stock === "low_stock" || inv?.risk_flag) primary.push("Inventory risk present — assess supply situation before committing to a price change.");
    else primary.push("Mixed or uncertain signals — gather more data before acting.");
    return primary;
  }

  if (finalActionType === "promote") {
    primary.push("Overstocked SKU — a visibility boost moves inventory without sacrificing margin through price cuts.");
    return primary;
  }

  // reprice — pricing delta is shown separately in the pricing line, so focus the WHY
  // on the evidence quality behind the recommendation, not the number itself.
  if (pr.price_missing) {
    primary.push("No pricing data available — cannot form a confident recommendation.");
  } else if (pr.near_bound) {
    primary.push("Pricing signal exists but is near the policy cap — apply in stages.");
  } else if (pr.large_delta) {
    primary.push("Pricing model signals a large adjustment — verify before applying at full scale.");
  } else if (Math.abs(pct) < 0.01) {
    primary.push("Pricing model finds no strong signal — current price appears appropriate.");
  } else {
    primary.push("Pricing model finds a clear opportunity — signals support this adjustment.");
  }

  // Secondary: most important cautionary note
  if (pNeg >= 0.25 && pct > 0) {
    secondary.push(`Customer feedback is ${pNeg >= 0.40 ? "strongly" : "moderately"} negative (${Math.round(pNeg * 100)}%) — monitor closely after any increase.`);
  } else if (pr.near_bound) {
    secondary.push("Delta is near the policy cap — apply conservatively or in stages.");
  } else if (pr.large_delta) {
    secondary.push("Suggested delta is large — verify before applying at full scale.");
  } else if (returns >= 3) {
    secondary.push(`${returns} returns logged — watch return rate after repricing.`);
  } else if (nRev === 0) {
    secondary.push("No review data — sentiment confidence is very low.");
  } else if (nRev < 10) {
    secondary.push(`Only ${nRev} reviews — sentiment signal may not be representative.`);
  }

  return [...primary, ...secondary].slice(0, 2);
}

function buildOverrideNote(suggestedAction, finalActionType, a) {
  const pr = a?.pricing || {};
  const sent = a?.sentiment || {};
  const inv = a?.inventory || {};
  const sig = a?.signals || {};
  const pNeg = Number(sent?.p_neg || 0);
  const returns = Number(sig?.total_returns || 0);
  const stock = String(inv?.stock_status || "unknown");

  // Specific, seller-readable explanations for common transitions
  if (suggestedAction === "investigate" && finalActionType === "reprice") {
    if (pNeg >= 0.25) return "Pricing signal was strong enough to override the investigation flag — but negative feedback remains a concern.";
    return "Pricing signal overrode the initial caution flag — review this recommendation carefully.";
  }
  if (suggestedAction === "hold" && finalActionType === "reprice") {
    return "Pricing opportunity identified despite inventory caution — proceed with care.";
  }
  if (suggestedAction === "reprice" && finalActionType === "investigate") {
    if (returns >= 3) return "Recommendation changed to investigation — elevated returns suggest a quality issue to resolve first.";
    if (pNeg >= 0.30) return "Recommendation changed to investigation — high negative feedback outweighs the pricing opportunity.";
    return "Recommendation changed to investigation — risk signals outweighed the pricing signal.";
  }
  if (suggestedAction === "reprice" && finalActionType === "hold") {
    return "Recommendation reduced to hold — inventory or sentiment risk is too high to act on the pricing signal now.";
  }
  // Generic fallback
  return "Recommendation adjusted based on combined signal analysis — review before acting.";
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
    const finalActionType = String(a.action_type || "reprice").trim().toLowerCase();
    const actionBadgeColor = ACTION_BADGE_COLORS[finalActionType] || "bg-slate-500/20 text-slate-300 border-slate-500/40";
    const confidence = computeConfidence(a, pct);
    const whyBullets = buildWhyBullets(a, pct, finalActionType);

    // Caution notes: when the final action still carries notable contradictions
    const cautionNotes = [];
    if (finalActionType === "reprice" && pct > 0.001 && pNeg >= 0.30) {
      cautionNotes.push(`Customer feedback is ${pNeg >= 0.40 ? "strongly" : "notably"} negative (${Math.round(pNeg * 100)}%). Monitor closely if you increase price.`);
    }
    if (finalActionType === "reprice" && pct > 0.001 && (inv.risk_flag || ["low_stock", "stockout_risk"].includes(inv.stock_status))) {
      cautionNotes.push("Inventory is under pressure — a price increase may further reduce sell-through.");
    }
    if (finalActionType === "reprice" && pct < -0.001 && inv.stock_status === "stockout_risk") {
      cautionNotes.push("Reducing price may accelerate stockout — consider restocking before discounting.");
    }

    // Override note: context-aware seller language, not internal state machine language
    const overrideNote = (suggestedAction && finalActionType && suggestedAction !== finalActionType)
      ? buildOverrideNote(suggestedAction, finalActionType, a)
      : null;

    const rawTitle = productTitle || a.product_id;
    const shortTitle = rawTitle.length > 72 ? rawTitle.slice(0, 70).trimEnd() + "…" : rawTitle;

    return {
      id: String(a.product_id),
      title: shortTitle,
      fullTitle: rawTitle,
      productId: String(a.product_id),
      actionLabel: actionLabel(finalActionType, pct),
      actionBadgeColor,
      confidence,
      pricingLine: buildPricingLine(a, pct, finalActionType),
      whyBullets,
      cautionNotes,
      overrideNote,
      finalActionType,
      suggestedAction,
      retrievalSimilarity,
      retrievalScore: retrieval,
      evidenceSnippet: evidenceSnippet ? String(evidenceSnippet).slice(0, 240) : "",
      riskLevel,
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
      // Build judge summary from structured ranked_actions — far cleaner than raw LLM text
      const judgeMessage = (() => {
        const actions = (data.ranked_actions || []).slice(0, 3);
        if (!actions.length) return "Judge synthesis complete. No ranked actions returned.";
        const lines = actions.map((a, i) => {
          const pct = Number(a.recommended_price_change_pct || 0);
          const type = String(a.action_type || "reprice").toLowerCase();
          const pid = `…${String(a.product_id).slice(-6)}`;
          let decision = type;
          if (type === "reprice") decision = Math.abs(pct) < 0.01 ? "hold price" : (pct > 0 ? `increase price +${pct.toFixed(2)}%` : `decrease price ${Math.abs(pct).toFixed(2)}%`);
          if (type === "investigate") decision = "investigate before repricing";
          if (type === "hold") decision = "hold";
          if (type === "restock") decision = "restock";
          if (type === "promote") decision = "promote";
          return `${i + 1}. ${decision} (${pid})`;
        });
        return `Judge's final decisions:\n${lines.join("\n")}`;
      })();
      const judgeTurn = {
        id: `judge-${Date.now()}`,
        actor: "Judge",
        message: judgeMessage,
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
          portfolioSummary={computePortfolioSummary(pipelineResult?.enriched_candidates, objective)}
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
