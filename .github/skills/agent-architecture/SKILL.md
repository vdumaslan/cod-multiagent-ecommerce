---
name: agent-architecture
description: Use this skill when working with any of the 5 agents, the debate structure, orchestration logic, sentiment analysis, or any decisions about how the multi-agent system is designed or why it is designed that way
---

# Multi-Agent E-Commerce Architecture

## Project Overview
A Human-AI collaborative e-commerce recommendation system built on a Chain of Debate (CoD) framework.  Multiple specialized agents debate internally before delivering a consensus recommendation to the user.

- Team: Victor Dumaslan, Dongmei Han, Niranjan Rao, Thi Thao Tien Tran
- GitHub: https://github.com/vdumaslan/cod-multiagent-ecommerce

## Current Status
- Phase 1 (Data Engineering): COMPLETE
- Phase 2 (Cloud Pipeline): COMPLETE
- Phase 3 (Agent Development): IN PROGRESS - focus on Sentiment Agent and Product Discovery Agent first

## The 5 Agents

### 1. Sentiment & User Voice Agent (DEBATER)
- **Status:** To be built - start here
- **Role:** Real user voice via Amazon Reviews aspect-based sentiment analysis
- **Aspects tracked:** battery_life, performance, build_quality, price_value, etc.
- **Model:** TBD
- **Output format:** "X% of N users [praise/complain] [aspect] (avg Y)"

### 2. Product Discovery Agent (NON-DEBATER)
- **Status:** To be built
- **Role:** Parses natural language query -> retrieves ~50 candidate products via semantic search
- **Model:** TBD

### 3. Recommendation & Ranking Agent (DEBATER)
- **Status:** To be built
- **Role:** Argues for best quality/specs product
- **Model:** TBD
- **Debate style:** "Product A has best quality (4.5/5, best specs)"

### 4. Value & Pricing Agent (DEBATER)
- **Status:** To be built
- **Role:** Argues for best value-for-money product
- **Model:** TBD
- **Debate style:** "Product B has best value ($799, saves $100)"

### 5. Orchestrator Agent (NON-DEBATER)
- **Status:** To be built
- **Role:** Moderates debate, synthesizes final recommendation, delivers to user
- **Model:** TBD
- **Consensus rule:** 2/3 debaters must agree on final product recommendation

## Debate Flow
1. Discovery finds ~50 candidate products based on user query
2. Round 1: All 3 debaters argue for their top product choice from the candidate set simultaneously
3. Orchestrator idnetifies conflicts and critical points
4. Round 2: Debaters respond to each other
5. 2/3 majority wins - Orchestrator synthesizes final recommendation with rationale and delivers to user