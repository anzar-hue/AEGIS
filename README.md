# AEGIS-HIA — Guardrailed Multi-Agent Human Impact & Severity Assessment Framework

**Team TEMPEST** | Track 1: Agentic AI (*AI in Social Life*) — **AGENTIC X 2.0 (Woxsen University Future Tech Summit '26)**

* **Live Demo (Frontend):** https://aegis-v1-prototype.vercel.app/
* **Backend API (FastAPI):** https://aegis-kxs4.onrender.com/docs

> **Important Note for Reviewers & Judges (Cold Start & V1 Prototype Status):**
> 1. **Cold Start Delay:** The backend is hosted on Render's free tier, which sleeps after periods of inactivity. **The first load or pipeline run may take 30–60 seconds** while the server wakes up and trains the lightweight `scikit-learn` prototype models. If KPIs or 3D charts initially show `—`, please wait a moment and refresh.
> 2. **V1 Prototype & Make-Do Gemini Integration:** This deployment is a **V1 prototype** built for hackathon evaluation. The Gemini integration (`llm_fallback.py` + `agents.py`) currently operates as a **make-do / fallback-resilient setup** because Gemini model availability and demand limits caused intermittent errors and delays during testing. When external LLM calls are unavailable, rate-limited, or fail, the system gracefully falls back to local deterministic/rule-based handling so the pipeline and UI stay functional. We plan to **switch/upgrade the LLM provider layer to another API** in a subsequent iteration.

---

## Overview

**AEGIS-HIA** is a guardrailed, multi-agent decision-support framework designed to assess the societal impact of deployed AI systems across two distinct harm axes:
* **Employment Harm** (e.g., job displacement, termination, wage pressure, deskilling, workplace monitoring)
* **Human Sovereignty Harm** (e.g., automated decisioning without appeal, autonomy restriction, surveillance, privacy/data exposure, manipulation)

At the same time, **AEGIS-HIA** tracks **Independent Benefits** separately rather than netting benefit percentages against harm scores.

### Core Architectural Principle: Deterministic Math First
LLMs are never allowed to calculate severity scores or invent statistics:
1. **Agent 1 (Predicate & Dimension Extractor):** Parses incident text into structured predicates, controlled harm categories, confirmation statuses, and 1–5 dimension levels (with fallback support when LLM endpoints fail).
2. **Deterministic Python Math & ML Core (`math_engine.py` & `ml_predictor.py`):** Computes Weighted Power Mean ($q=2$) severity scores, confirmation-weighted burdens, $Q_{90}$ tail severity, and `scikit-learn` (`LogisticRegression` + `Ridge`) scenario predictions.
3. **Agent 2 (Impact Analyst & Advisor):** Translates deterministic outputs into institutional summaries and interactive policy/safeguard guidance.
4. **Agent 3 (Rule-Based Evaluator — 0 LLM Calls):** Runs deterministic validation checks across extracted fields, weights, and bounds.

---

## Key Features
* **AgenticX 2.0 HUD Interface:** Full-screen dark tactical UI with mouse-tracking spotlight effects, dataset switcher (`SYNTHETIC` vs. `DOCUMENTED`), and a toggleable **Developer's Dashboard** for inspecting live SSE execution traces and Agent 3 audit results.
* **Interactive 3D Analytics Lab (Plotly.js / WebGL):** Switch between and rotate three 3D views—*ML Risk Surface*, *Severity Landscape*, and *Harm–Benefit Cube*—while adjusting sliders for AI-only work percentage, safeguard score, and sector.
* **3-Agent Pipeline & AI Advisor Chat:** Streams stage-by-stage progress (`AGENT_1`, `MATH_ENGINE`, `AGENT_2`, `AGENT_3`) and provides an interactive chat interface grounded in current deterministic metrics.
* **Epistemic Honesty:** All metrics are statistical estimates with uncertainty bounds and are strictly dependent on the structured AEGIS-HIA schema (`data/*.csv`).

---

## Repository Structure

```text
AEGIS/
├── data/                        # Structured CSV datasets (deployments, cases, harms, attributes, benefits, validation)
├── backend/
│   ├── math_engine.py           # Deterministic Power Mean (q=2), Functions 1 & 4, Agent 3 Evaluator
│   ├── ml_predictor.py          # Scikit-learn Classifier + Ridge Regressor + 3D visualization data
│   ├── llm_fallback.py          # Make-do Gemini model router & fallback wrapper (planned API switch later)
│   ├── agents.py                # Agent 1, Agent 2, SSE pipeline stream, and Advisor Chat
│   ├── server.py                # FastAPI endpoints and CORS configuration
│   └── requirements.txt         # Python dependencies
└── frontend/
    ├── public/favicon.svg       # Custom AEGIS-HIA shield favicon
    ├── index.html               # App entry HTML & Google Fonts
    └── src/
        ├── api/client.js        # Backend REST & SSE client using VITE_API_BASE
        ├── components/          # PipelineVisual, Charts3D, DevDashboard
        ├── App.jsx              # Multi-view AgenticX 2.0 dashboard
        └── index.css            # Grid & HUD styling
```

---

## Running Locally

### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```
*(Optional)* Add a `backend/.env` file with `GEMINI_API_KEY=your_key_here` if testing live Gemini calls through your existing `llm_fallback.py` setup.

### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
By default, the frontend points to `http://127.0.0.1:8000` locally or `VITE_API_BASE` in production (`https://aegis-kxs4.onrender.com`).