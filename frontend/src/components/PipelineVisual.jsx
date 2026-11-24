import React from "react";
import { Cpu, FileSearch, Calculator, FileText, ShieldCheck, CheckCircle2, AlertTriangle, Loader2 } from "lucide-react";

const STAGES = [
  { key: "AGENT_1", label: "Agent 1: Extractor", sub: "LLM Predicate JSON", icon: FileSearch, color: "cyan" },
  { key: "MATH_ENGINE", label: "Python Math Core", sub: "Power Mean (q=2)", icon: Calculator, color: "indigo" },
  { key: "AGENT_2", label: "Agent 2: Analyst", sub: "Institutional Brief", icon: FileText, color: "emerald" },
  { key: "AGENT_3", label: "Agent 3: Evaluator", sub: "0-LLM Rule Audit", icon: ShieldCheck, color: "rose" },
];

export default function PipelineVisual({ stageStates }) {
  return (
    <div className="bg-slate-900/90 border border-slate-700/80 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <h3 className="text-sm font-bold tracking-wider uppercase text-slate-200">
            Live Guardrailed Multi-Agent Pipeline
          </h3>
        </div>
        <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-cyan-300 border border-cyan-500/30">
          Zero LLM Math Guarantee
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {STAGES.map((st) => {
          const Icon = st.icon;
          const info = stageStates[st.key] || { status: "IDLE" };
          const isRunning = info.status === "RUNNING";
          const isDone = info.status === "DONE";
          const isFlagged = st.key === "AGENT_3" && info.payload?.audit_status === "FLAGGED_FOR_HUMAN_REVIEW";

          let borderClass = "border-slate-700 bg-slate-800/60";
          if (isRunning) borderClass = "border-cyan-400 bg-cyan-950/40 shadow-[0_0_20px_rgba(34,211,238,0.25)] animate-pulse";
          else if (isFlagged) borderClass = "border-rose-500 bg-rose-950/40";
          else if (isDone) borderClass = "border-emerald-500/70 bg-emerald-950/20";

          return (
            <div key={st.key} className={`rounded-xl border p-4 transition-all duration-300 ${borderClass}`}>
              <div className="flex items-center justify-between mb-2">
                <Icon className={`w-5 h-5 ${isRunning ? "text-cyan-400" : isDone ? "text-emerald-400" : "text-slate-400"}`} />
                {isRunning && <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />}
                {isDone && !isFlagged && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                {isFlagged && <AlertTriangle className="w-4 h-4 text-rose-400" />}
              </div>
              <div className="text-sm font-bold text-white">{st.label}</div>
              <div className="text-xs text-slate-400 mb-2">{st.sub}</div>
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className={isRunning ? "text-cyan-300" : isFlagged ? "text-rose-300" : isDone ? "text-emerald-300" : "text-slate-500"}>
                  {isFlagged ? "FLAGGED" : info.status}
                </span>
                {info.latency_ms !== undefined && (
                  <span className="text-slate-400">{info.latency_ms} ms</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}