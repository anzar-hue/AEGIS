import { useEffect, useState } from "react";
import {
  Terminal, Play, Sliders, ShieldAlert, Activity,
  Box, Bot, Lock, Send, MessageSquare, Sparkles
} from "lucide-react";
import {
  fetchFunction1, fetchFunction4, fetchCharts3D,
  predictScenario, streamReportAnalysis, sendChatMessage
} from "./api/client";
import PipelineVisual from "./components/PipelineVisual";
import Charts3D from "./components/Charts3D";
import DevDashboard from "./components/DevDashboard";

const SAMPLE_REPORT =
  "An automated AI recruitment screening system rejected a qualified applicant using a protected-group proxy and denied any opportunity for human appeal.";

const NAV_TABS = [
  { id: "overview", label: "OVERVIEW", badge: "LIVE" },
  { id: "chat", label: "AI ADVISOR", badge: "AGENT 2" },
  { id: "analytics3d", label: "3D LAB", badge: "WEBGL" },
  { id: "pipeline", label: "PIPELINE", badge: "3-AGENT" },
  { id: "roadmap", label: "ROADMAP", badge: "R2" },
];

// Card with local mouse-tracking blue glow
function SpotlightCard({ children, className = "" }) {
  const [pos, setPos] = useState({ x: -200, y: -200, active: false });

  return (
    <div
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setPos({ x: e.clientX - rect.left, y: e.clientY - rect.top, active: true });
      }}
      onMouseLeave={() => setPos((p) => ({ ...p, active: false }))}
      className={`hud-card rounded-md overflow-hidden ${className}`}
    >
      <div
        className="pointer-events-none absolute inset-0 transition-opacity duration-300"
        style={{
          opacity: pos.active ? 1 : 0,
          background: `radial-gradient(320px circle at ${pos.x}px ${pos.y}px, rgba(59, 130, 246, 0.16), transparent 70%)`,
        }}
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}

export default function App() {
  const [mouse, setMouse] = useState({ x: -500, y: -500 });
  const [activePage, setActivePage] = useState("overview");
  const [activeGraph, setActiveGraph] = useState("surface");

  const [f1, setF1] = useState(null);
  const [f4, setF4] = useState(null);
  const [charts, setCharts] = useState(null);
  const [recordType, setRecordType] = useState("SYNTHETIC");
  const [powerQ, setPowerQ] = useState(2.0);

  // 3-Agent Pipeline
  const [reportText, setReportText] = useState(SAMPLE_REPORT);
  const [running, setRunning] = useState(false);
  const [stageStates, setStageStates] = useState({});
  const [traceLog, setTraceLog] = useState([]);
  const [devOpen, setDevOpen] = useState(false);

  // ML Predictor
  const [aiPct, setAiPct] = useState(65);
  const [safeguardScore, setSafeguardScore] = useState(4);
  const [predField, setPredField] = useState("Recruitment & HR");
  const [livePred, setLivePred] = useState(null);

  // AI Chatbot
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Agent 2 online. Ask me how to reduce Employment or Sovereignty harm, evaluate safeguard trade-offs, or interpret your Q90 tail severity.",
    },
  ]);

  useEffect(() => {
    fetchFunction1(recordType, "ALL", powerQ).then(setF1).catch(console.error);
  }, [recordType, powerQ]);

  useEffect(() => {
    fetchFunction4().then(setF4).catch(console.error);
  }, []);

  useEffect(() => {
    fetchCharts3D(predField).then(setCharts).catch(console.error);
  }, [predField]);

  useEffect(() => {
    predictScenario({ ai_only_pct: aiPct, safeguard_score: safeguardScore, field: predField })
      .then(setLivePred)
      .catch(console.error);
  }, [aiPct, safeguardScore, predField]);

  const handleAnalyze = async () => {
    setRunning(true);
    setStageStates({});
    setTraceLog([]);
    try {
      await streamReportAnalysis(reportText, (ev) => {
        setTraceLog((prev) => [...prev, ev]);
        setStageStates((prev) => ({ ...prev, [ev.stage]: ev }));
      });
    } catch (e) {
      console.error(e);
    } finally {
      setRunning(false);
    }
  };

  const handleSendChat = async (customText) => {
    const q = (customText || chatInput).trim();
    if (!q || chatLoading) return;
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await sendChatMessage(q, {
        function1: f1?.result?.dimensions,
        benefits: f1?.result?.independent_benefits,
        function4: f4?.residual_risk_ratio_A_over_B,
        live_ml_prediction: livePred,
      });
      setMessages((prev) => [...prev, { role: "assistant", text: res.reply, mode: res.mode }]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", text: "Backend unreachable. Start uvicorn on port 8000." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const emp = f1?.result?.dimensions?.EMPLOYMENT;
  const sov = f1?.result?.dimensions?.SOVEREIGNTY;
  const bens = f1?.result?.independent_benefits;

  return (
    <div
      onMouseMove={(e) => setMouse({ x: e.clientX, y: e.clientY })}
      className="h-screen w-screen overflow-hidden flex flex-col bg-[#030612] tech-grid text-slate-100 relative"
    >
      {/* GLOBAL MOUSE-HOVER BLUE SPOTLIGHT */}
      <div
        className="pointer-events-none fixed inset-0 z-0 transition-opacity duration-200"
        style={{
          background: `radial-gradient(520px circle at ${mouse.x}px ${mouse.y}px, rgba(37, 99, 235, 0.14), transparent 65%)`,
        }}
      />

      {/* TOP AGENTICX 2.0 STYLE NAVBAR */}
      <header className="relative z-20 h-16 shrink-0 bg-[#030612]/90 backdrop-blur-md border-b border-blue-500/20 px-6 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div>
            <div className="text-base font-extrabold tracking-widest text-white leading-none">
              AEGIS-HIA
            </div>
            <div className="font-mono-tech text-[9px] tracking-[0.2em] text-blue-400/80 mt-1">
              TEAM TEMPEST • AGENTICX 2.0
            </div>
          </div>
        </div>

        {/* Center Horizontal Nav Links */}
        <nav className="hidden md:flex items-center gap-6 h-full">
          {NAV_TABS.map((tab) => {
            const active = activePage === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActivePage(tab.id)}
                className={`h-full flex items-center gap-2 font-mono-tech text-xs tracking-wider border-b-2 transition ${
                  active
                    ? "border-blue-500 text-white font-bold"
                    : "border-transparent text-slate-400 hover:text-blue-300"
                }`}
              >
                <span>{tab.label}</span>
                <span
                  className={`text-[9px] px-1.5 py-0.5 rounded-sm font-bold ${
                    active
                      ? "bg-blue-600 text-white shadow-[0_0_10px_rgba(37,99,235,0.7)]"
                      : "bg-blue-950/80 text-blue-400 border border-blue-500/30"
                  }`}
                >
                  {tab.badge}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Right Controls */}
        <div className="flex items-center gap-3">
          <select
            value={recordType}
            onChange={(e) => setRecordType(e.target.value)}
            className="font-mono-tech bg-[#070d1c] border border-blue-500/30 text-[11px] rounded px-2.5 py-1.5 text-blue-200 focus:outline-none focus:border-blue-400"
          >
            <option value="SYNTHETIC">SYNTHETIC (N=1,000)</option>
            <option value="DOCUMENTED">DOCUMENTED (N=15)</option>
          </select>

          <button
            onClick={() => setDevOpen(true)}
            className="font-mono-tech flex items-center gap-2 px-4 py-2 rounded-full bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-bold tracking-wider shadow-[0_0_20px_rgba(37,99,235,0.55)] transition"
          >
            <Terminal className="w-3.5 h-3.5" />
            DEV CONSOLE
          </button>
        </div>
      </header>

      {/* MAIN CONTENT VIEWPORT */}
      <main className="relative z-10 flex-1 overflow-y-auto px-8 py-6 max-w-[1440px] w-full mx-auto space-y-6">
        {/* Breadcrumb + Epistemic Tag */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="font-mono-tech text-[11px] tracking-[0.18em] text-slate-400">
            FTS '26 <span className="text-blue-500">/</span> TRACK 1: AGENTIC AI <span className="text-blue-500">/</span>{" "}
            <span className="text-blue-400 uppercase">{activePage}</span>
          </div>
          <div className="font-mono-tech text-[11px] px-3 py-1 bg-blue-950/50 border border-blue-500/30 text-blue-300 flex items-center gap-2">
            <ShieldAlert className="w-3.5 h-3.5 text-blue-400" />
            ESTIMATES ONLY • STRICTLY SCHEMA-DEPENDENT (AI IN SOCIAL LIFE)
          </div>
        </div>

        {/* ==================== PAGE 1: OVERVIEW ==================== */}
        {activePage === "overview" && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h1 className="text-4xl font-extrabold tracking-tight text-white leading-tight">
                  AEGIS-HIA <span className="text-blue-500">IMPACT ARENA</span>
                </h1>
                <p className="font-mono-tech text-xs tracking-wider text-blue-300/80 mt-1 uppercase">
                  GUARDRAILED INDIVIDUAL SEVERITY • INDEPENDENT BENEFITS • POWER MEAN q={powerQ}
                </p>
              </div>
              <div className="flex items-center gap-1 bg-[#070d1c] border border-blue-500/30 p-1 rounded">
                <span className="font-mono-tech text-[10px] text-slate-400 px-2">SENSITIVITY:</span>
                {[1.0, 2.0, 3.0].map((qVal) => (
                  <button
                    key={qVal}
                    onClick={() => setPowerQ(qVal)}
                    className={`font-mono-tech px-3 py-1 text-xs font-bold rounded-sm transition ${
                      powerQ === qVal ? "bg-blue-600 text-white" : "text-slate-400 hover:text-blue-300"
                    }`}
                  >
                    q={qVal}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              <SpotlightCard className="p-6">
                <div className="font-mono-tech text-[11px] font-bold tracking-widest text-blue-400 uppercase mb-4">
                  01 // EMPLOYMENT HARM AXIS
                </div>
                <div className="grid grid-cols-2 gap-5">
                  <div><div className="font-mono-tech text-[10px] text-slate-400">AFFECTED PEOPLE</div><div className="text-2xl font-extrabold mt-0.5">{emp?.affected_people ?? "—"}</div></div>
                  <div><div className="font-mono-tech text-[10px] text-slate-400">RATE / 1,000</div><div className="text-2xl font-extrabold mt-0.5">{emp?.event_rate_per_1000 ?? "—"}</div></div>
                  <div><div className="font-mono-tech text-[10px] text-slate-400">MEAN SEVERITY</div><div className="text-xl font-bold text-slate-200">{emp?.mean_severity ?? "—"}</div></div>
                  <div><div className="font-mono-tech text-[10px] text-rose-400">Q90 TAIL (USP)</div><div className="text-xl font-extrabold text-rose-400">{emp?.tail_severity_q90 ?? "—"}</div></div>
                </div>
              </SpotlightCard>

              <SpotlightCard className="p-6">
                <div className="font-mono-tech text-[11px] font-bold tracking-widest text-blue-400 uppercase mb-4">
                  02 // HUMAN SOVEREIGNTY AXIS
                </div>
                <div className="grid grid-cols-2 gap-5">
                  <div><div className="font-mono-tech text-[10px] text-slate-400">AFFECTED PEOPLE</div><div className="text-2xl font-extrabold mt-0.5">{sov?.affected_people ?? "—"}</div></div>
                  <div><div className="font-mono-tech text-[10px] text-slate-400">RATE / 1,000</div><div className="text-2xl font-extrabold mt-0.5">{sov?.event_rate_per_1000 ?? "—"}</div></div>
                  <div><div className="font-mono-tech text-[10px] text-slate-400">MEAN SEVERITY</div><div className="text-xl font-bold text-slate-200">{sov?.mean_severity ?? "—"}</div></div>
                  <div><div className="font-mono-tech text-[10px] text-blue-400">Q90 TAIL (USP)</div><div className="text-xl font-extrabold text-blue-400">{sov?.tail_severity_q90 ?? "—"}</div></div>
                </div>
              </SpotlightCard>

              <SpotlightCard className="p-6">
                <div className="font-mono-tech text-[11px] font-bold tracking-widest text-blue-400 uppercase mb-3">
                  03 // INDEPENDENT BENEFITS (UN-NETTED)
                </div>
                <div className="text-3xl font-extrabold text-white mb-3">
                  {bens?.total_benefit_records ?? 0}{" "}
                  <span className="font-mono-tech text-xs font-normal text-slate-400">RECORDS</span>
                </div>
                <div className="space-y-1.5 font-mono-tech text-xs text-slate-300">
                  {bens?.categories?.slice(0, 4).map((c) => (
                    <div key={c.category} className="flex justify-between border-b border-blue-500/15 pb-1">
                      <span className="truncate pr-2">{c.category}</span>
                      <span className="text-blue-400 font-bold">+{c.avg_raw_gain}%</span>
                    </div>
                  ))}
                </div>
              </SpotlightCard>
            </div>

            {f4 && (
              <SpotlightCard className="p-5 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <Activity className="w-5 h-5 text-blue-400" />
                  <div>
                    <div className="text-sm font-bold text-white uppercase tracking-wide">
                      FUNCTION 4 SAFEGUARD AUDIT — {f4.comparison}
                    </div>
                    <div className="font-mono-tech text-[11px] text-slate-400">{f4.note}</div>
                  </div>
                </div>
                <div className="flex gap-8 font-mono-tech text-xs">
                  <div>
                    <div className="text-[10px] text-slate-400">SOVEREIGNTY RESIDUAL RISK</div>
                    <div className="text-lg font-extrabold text-blue-400">{f4.residual_risk_ratio_A_over_B?.SOVEREIGNTY}x</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">BENEFIT GAIN (FULL vs NONE)</div>
                    <div className="text-lg font-extrabold text-white">
                      {f4.scenario_A?.avg_benefit_raw_gain}% vs {f4.scenario_B?.avg_benefit_raw_gain}%
                    </div>
                  </div>
                </div>
              </SpotlightCard>
            )}
          </div>
        )}

        {/* ==================== PAGE 2: AI ADVISOR CHAT ==================== */}
        {activePage === "chat" && (
          <SpotlightCard className="h-[calc(100vh-170px)] flex flex-col">
            <div className="px-6 py-4 border-b border-blue-500/20 flex flex-wrap items-center justify-between gap-3 bg-[#050a18]">
              <div className="flex items-center gap-3">
                <MessageSquare className="w-5 h-5 text-blue-400" />
                <div>
                  <h2 className="text-sm font-bold uppercase tracking-wider text-white">AGENT 2 // POLICY & SAFEGUARD ADVISOR</h2>
                  <p className="font-mono-tech text-[10px] text-slate-400">GROUNDED IN DETERMINISTIC PYTHON STATS • GEMINI API</p>
                </div>
              </div>
              <div className="flex gap-2">
                {["How do we lower Sovereignty burden?", "Explain the Q90 tail severity risk"].map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSendChat(q)}
                    className="font-mono-tech text-[10px] px-2.5 py-1 bg-blue-950/70 hover:bg-blue-900/70 text-blue-300 border border-blue-500/30 rounded-sm"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-2xl px-4 py-3 text-sm leading-relaxed rounded-md ${
                      m.role === "user"
                        ? "bg-blue-600/25 border border-blue-400/50 text-white"
                        : "bg-[#050a18] border border-blue-500/20 text-slate-200"
                    }`}
                  >
                    <div>{m.text}</div>
                    {m.mode && <div className="font-mono-tech text-[9px] text-blue-400/70 mt-1.5">MODE: {m.mode}</div>}
                  </div>
                </div>
              ))}
              {chatLoading && <div className="font-mono-tech text-xs text-blue-400 animate-pulse">&gt; AGENT 2 COMPUTING ADVISORY...</div>}
            </div>

            <div className="p-4 border-t border-blue-500/20 bg-[#050a18] flex gap-3">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendChat()}
                placeholder="Ask Agent 2 for safeguard recommendations or statistical interpretation..."
                className="flex-1 bg-[#030612] border border-blue-500/30 rounded px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-400"
              />
              <button
                onClick={() => handleSendChat()}
                disabled={chatLoading}
                className="font-mono-tech px-5 py-2.5 rounded bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs flex items-center gap-2 shadow-[0_0_15px_rgba(37,99,235,0.5)]"
              >
                <Send className="w-3.5 h-3.5" /> SEND
              </button>
            </div>
          </SpotlightCard>
        )}

        {/* ==================== PAGE 3: 3D ANALYTICS LAB ==================== */}
        {activePage === "analytics3d" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h1 className="text-2xl font-extrabold uppercase tracking-tight text-white">
                3D <span className="text-blue-500">ANALYTICS LAB</span>
              </h1>
              <div className="flex gap-1.5 bg-[#070d1c] p-1 border border-blue-500/30 rounded">
                {[
                  { id: "surface", label: "ML RISK SURFACE (F2)" },
                  { id: "landscape", label: "SEVERITY LANDSCAPE (F1)" },
                  { id: "pareto", label: "HARM–BENEFIT CUBE (F4)" },
                ].map((g) => (
                  <button
                    key={g.id}
                    onClick={() => setActiveGraph(g.id)}
                    className={`font-mono-tech px-3 py-1.5 rounded-sm text-[11px] font-bold transition ${
                      activeGraph === g.id ? "bg-blue-600 text-white" : "text-slate-400 hover:text-blue-300"
                    }`}
                  >
                    {g.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
              <SpotlightCard className="xl:col-span-8 p-4">
                <Charts3D activeGraph={activeGraph} chartsData={charts} livePrediction={livePred} />
              </SpotlightCard>

              <SpotlightCard className="xl:col-span-4 p-5 space-y-5">
                <div className="font-mono-tech flex items-center gap-2 text-xs font-bold tracking-widest text-blue-400 uppercase">
                  <Sliders className="w-4 h-4" /> LIVE ML PARAMETER TWEAKER
                </div>
                <div>
                  <div className="flex justify-between font-mono-tech text-xs mb-1">
                    <span className="text-slate-400">AI-ONLY WORK %</span>
                    <span className="font-bold text-blue-400">{aiPct}%</span>
                  </div>
                  <input type="range" min="10" max="90" value={aiPct} onChange={(e) => setAiPct(Number(e.target.value))} className="w-full accent-blue-500" />
                </div>
                <div>
                  <div className="flex justify-between font-mono-tech text-xs mb-1">
                    <span className="text-slate-400">SAFEGUARD SCORE</span>
                    <span className="font-bold text-blue-400">{safeguardScore} / 10</span>
                  </div>
                  <input type="range" min="0" max="10" value={safeguardScore} onChange={(e) => setSafeguardScore(Number(e.target.value))} className="w-full accent-blue-500" />
                </div>
                <div>
                  <label className="font-mono-tech text-[10px] text-slate-400 block mb-1">SECTOR</label>
                  <select value={predField} onChange={(e) => setPredField(e.target.value)} className="w-full bg-[#030612] border border-blue-500/30 rounded px-3 py-2 text-xs font-mono-tech">
                    <option>Recruitment & HR</option>
                    <option>Public Administration</option>
                    <option>Finance & Insurance</option>
                    <option>Healthcare</option>
                  </select>
                </div>
                <div className="grid grid-cols-3 gap-2 pt-3 border-t border-blue-500/20 text-center font-mono-tech">
                  <div className="bg-[#030612] border border-blue-500/20 rounded p-2"><div className="text-[9px] text-slate-400">P(HARM)</div><div className="text-sm font-bold text-blue-400">{livePred?.predicted_harm_probability ?? "—"}</div></div>
                  <div className="bg-[#030612] border border-blue-500/20 rounded p-2"><div className="text-[9px] text-slate-400">EXP. SEV</div><div className="text-sm font-bold text-white">{livePred?.dimensions?.SOVEREIGNTY?.expected_severity ?? "—"}</div></div>
                  <div className="bg-[#030612] border border-blue-500/20 rounded p-2"><div className="text-[9px] text-slate-400">BURDEN/1K</div><div className="text-sm font-bold text-rose-400">{livePred?.dimensions?.SOVEREIGNTY?.expected_burden_per_1000 ?? "—"}</div></div>
                </div>
                <button
                  onClick={() => setActivePage("chat")}
                  className="font-mono-tech w-full py-2.5 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold tracking-wider shadow-[0_0_15px_rgba(37,99,235,0.4)] transition"
                >
                  GET AGENT 2 ADVICE ON THIS SCENARIO →
                </button>
              </SpotlightCard>
            </div>
          </div>
        )}

        {/* ==================== PAGE 4: 3-AGENT PIPELINE ==================== */}
        {activePage === "pipeline" && (
          <div className="space-y-5">
            <SpotlightCard className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-blue-400" />
                  <h1 className="text-base font-extrabold uppercase tracking-wider text-white">LIVE 3-AGENT REPORT ANALYZER</h1>
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={running}
                  className="font-mono-tech flex items-center gap-2 px-5 py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold text-xs shadow-[0_0_15px_rgba(37,99,235,0.5)]"
                >
                  <Play className="w-3.5 h-3.5" /> {running ? "RUNNING..." : "EXECUTE PIPELINE"}
                </button>
              </div>
              <textarea
                rows={3}
                value={reportText}
                onChange={(e) => setReportText(e.target.value)}
                className="w-full bg-[#030612] border border-blue-500/30 rounded p-3.5 text-sm text-slate-200 focus:outline-none focus:border-blue-400"
              />
            </SpotlightCard>
            <PipelineVisual stageStates={stageStates} />
            <SpotlightCard className="p-5">
              <div className="font-mono-tech text-[11px] font-bold tracking-widest text-blue-400 uppercase mb-2">
                AGENT 2 // INSTITUTIONAL BRIEF
              </div>
              <p className="text-sm text-slate-200 leading-relaxed">
                {stageStates?.AGENT_2?.payload?.summary || "Execute the pipeline above to generate an audited institutional brief."}
              </p>
            </SpotlightCard>
          </div>
        )}

        {/* ==================== PAGE 5: ROADMAP ==================== */}
        {activePage === "roadmap" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SpotlightCard className="p-6">
              <span className="font-mono-tech inline-flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1 bg-blue-950 text-blue-300 border border-blue-500/40 mb-4">
                <Lock className="w-3 h-3" /> COMING SOON • LIVE DB SYNC
              </span>
              <h2 className="text-lg font-bold text-white mb-2">1. Live Report Ingestion & Delta Tracker</h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Type new real-world AI incident reports, approve through Agent 3, commit to PostgreSQL, and watch population-level rates and Q90 tail severity shift live.
              </p>
            </SpotlightCard>
            <SpotlightCard className="p-6">
              <span className="font-mono-tech inline-flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1 bg-blue-950 text-blue-300 border border-blue-500/40 mb-4">
                <Lock className="w-3 h-3" /> COMING SOON • COUNTERFACTUAL ENGINE
              </span>
              <h2 className="text-lg font-bold text-white mb-2">2. Deployed AI Scaling & Impact Simulator</h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Tweak deployed AI coverage across a sector to see whether deploying more hurts or benefits—and pinpoint exactly where it hurts (Employment vs. Sovereignty).
              </p>
            </SpotlightCard>
          </div>
        )}
      </main>

      <DevDashboard open={devOpen} onClose={() => setDevOpen(false)} traceLog={traceLog} stageStates={stageStates} />
    </div>
  );
}