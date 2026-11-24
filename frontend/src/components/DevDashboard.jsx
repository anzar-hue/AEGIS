import React from "react";
import { Terminal, ShieldCheck, AlertOctagon, X } from "lucide-react";

export default function DevDashboard({ open, onClose, traceLog, stageStates }) {
  if (!open) return null;

  const audit = stageStates?.AGENT_3?.payload;

  return (
    <div className="fixed inset-y-0 right-0 w-full max-w-xl bg-slate-950 border-l border-cyan-500/40 shadow-2xl z-50 flex flex-col">
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 bg-slate-900">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Developer's Dashboard</h3>
            <p className="text-[11px] text-slate-400">Sanitized Agent Telemetry & Rule-Based Audit Log</p>
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4 font-mono text-xs">
        {audit && (
          <div
            className={`p-3.5 rounded-xl border ${
              audit.audit_status === "PASS"
                ? "bg-emerald-950/40 border-emerald-500/50 text-emerald-200"
                : "bg-rose-950/40 border-rose-500/50 text-rose-200"
            }`}
          >
            <div className="flex items-center gap-2 font-bold mb-1">
              {audit.audit_status === "PASS" ? <ShieldCheck className="w-4 h-4" /> : <AlertOctagon className="w-4 h-4" />}
              AGENT 3 EVALUATOR: {audit.audit_status} ({audit.flag_count} flags)
            </div>
            {audit.flags?.length > 0 && (
              <ul className="list-disc pl-5 mt-1 space-y-1">
                {audit.flags.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            )}
          </div>
        )}

        <div>
          <div className="text-slate-400 uppercase text-[10px] mb-2">Live Execution Trace (SSE Stream)</div>
          {traceLog.length === 0 ? (
            <div className="text-slate-500 italic">Run an incident report through the pipeline to inspect live agent JSON payloads.</div>
          ) : (
            <div className="space-y-3">
              {traceLog.map((ev, idx) => (
                <div key={idx} className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                  <div className="flex justify-between text-[11px] mb-1.5">
                    <span className="text-cyan-400 font-bold">{ev.stage} [{ev.status}]</span>
                    {ev.latency_ms !== undefined && <span className="text-slate-400">{ev.latency_ms} ms</span>}
                  </div>
                  {ev.message && <div className="text-slate-300 mb-1">{ev.message}</div>}
                  {ev.payload && (
                    <pre className="bg-slate-950 p-2.5 rounded-lg overflow-x-auto text-[11px] text-emerald-300">
                      {JSON.stringify(ev.payload, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}