const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function fetchFunction1(recordType = "SYNTHETIC", field = "ALL", q = 2.0) {
  const params = new URLSearchParams({ record_type: recordType, field, q });
  const res = await fetch(`${API_BASE}/api/function1?${params}`);
  if (!res.ok) throw new Error("Failed to fetch Function 1 data");
  return res.json();
}

export async function fetchFunction4(safeguardCol = "Human_Appeal_Mechanism", valA = "FULL", valB = "NONE") {
  const params = new URLSearchParams({ safeguard_col: safeguardCol, val_a: valA, val_b: valB });
  const res = await fetch(`${API_BASE}/api/function4?${params}`);
  if (!res.ok) throw new Error("Failed to fetch Function 4 comparison");
  return res.json();
}

export async function fetchCharts3D(field = "Recruitment & HR") {
  const params = new URLSearchParams({ field });
  const res = await fetch(`${API_BASE}/api/charts3d?${params}`);
  if (!res.ok) throw new Error("Failed to fetch 3D chart data");
  return res.json();
}

export async function predictScenario(payload) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Prediction request failed");
  return res.json();
}

export async function streamReportAnalysis(reportText, onEvent) {
  const res = await fetch(`${API_BASE}/api/analyze-report-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report_text: reportText }),
  });

  if (!res.ok) throw new Error("Stream request failed");
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith("data: ")) {
        try {
          const parsed = JSON.parse(line.slice(6));
          onEvent(parsed);
        } catch (err) {
          console.error("SSE parse error:", err);
        }
      }
    }
  }
}
export async function sendChatMessage(message, context) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, context }),
  });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}