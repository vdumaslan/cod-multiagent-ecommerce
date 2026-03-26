# Trends, web context, and multi-agent layout

## Your question: internet + business advice + inventory context

**Goal:** combine **(A)** live or recent **external** information (trends, news, seasonality) with **(B)** **your** product/inventory/sales context, so recommendations feel grounded.

### Recommended pattern

1. **Orchestrator (router)**  
   - Parses the owner’s goal (e.g. “reduce dead stock before Q4”).  
   - Decides which specialists to call and merges their outputs into 2–3 ranked plans with citations.

2. **Evidence / RAG agent**  
   - **Tools:** vector store over `retrieval_corpus.parquet` + `reviews.parquet` (and optional chunking).  
   - **Output:** product-level evidence, review themes, risks.

3. **Analytics / “store ops” agent**  
   - **Tools:** read Parquet / small SQL over **`inventory_skus`**, **`sales_daily`**, **`store_kpis_weekly`**, **`product_signals`**.  
   - **Output:** stock risk, margin pressure, which SKUs to promote or liquidate, simple forecasts.

4. **Market intelligence agent (single dedicated role is fine)**  
   - **Responsibility:** anything **external** and time-sensitive the static dataset cannot contain.  
   - **Tools (pick one in production):**  
     - Web search API (Tavily, SerpAPI, Bing, Google CSE), or  
     - Curated RSS / industry feeds, or  
     - LLM with **forced** tool use + citation URLs (never free-hallucinate “latest trends”).  
   - **Inputs:** summarized **internal** context passed by the orchestrator (categories you sell, price bands, inventory hotspots) so searches are **narrow** (“2025 home kitchen storage trends”, not generic “business tips”).  
   - **Output:** short bullet “external signals” + links + **explicit uncertainty** when data is thin.

5. **Human approval**  
   - Logs decision, optional feedback for re-ranking (aligns with your plan’s approval loop).

### Why one “trends” agent is enough

You can absolutely implement **one** **Market intelligence agent** whose job is *only* external retrieval + synthesis, while RAG and analytics stay separate. That keeps boundaries clear:  
- **RAG** = *your* evidence.  
- **Analytics** = *your* numbers (including synthetic ops tables).  
- **Trends agent** = *the world* since training cutoff / outside your files.

### Guardrails

- **Never** pass full 43GB JSONL to the LLM; only curated + synthetic Parquet summaries.  
- **Tag** synthetic fields (`unit_cost`, `sales_daily`) as synthetic in prompts so the model doesn’t confuse them with Amazon ground truth.  
- **Require** URLs or tool traces for “latest trend” claims in demo mode.

This file is **design guidance**; wire concrete tool names when you implement the agent runtime (e.g. LangGraph, AutoGen, or custom).
