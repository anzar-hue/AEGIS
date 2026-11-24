import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

function Plotly3DCanvas({ data, layout }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !data) return;
    Plotly.react(
      ref.current,
      data,
      {
        paper_bgcolor: "rgba(5, 8, 17, 0)",
        plot_bgcolor: "rgba(5, 8, 17, 0)",
        font: { color: "#cbd5e1", family: "Inter, sans-serif", size: 12 },
        margin: { l: 0, r: 0, b: 0, t: 10 },
        ...layout,
      },
      { responsive: true, displaylogo: false }
    );
  }, [data, layout]);

  return <div ref={ref} className="w-full h-[500px]" />;
}

export default function Charts3D({ activeGraph, chartsData, livePrediction }) {
  if (!chartsData) {
    return (
      <div className="h-[500px] flex items-center justify-center text-slate-400 text-sm">
        Loading interactive 3D WebGL canvas...
      </div>
    );
  }

  const { landscape, risk_surface, pareto_cube } = chartsData;

  const sceneStyle = (xTitle, yTitle, zTitle) => ({
    xaxis: { title: xTitle, gridcolor: "#1e293b", zerolinecolor: "#334155" },
    yaxis: { title: yTitle, gridcolor: "#1e293b", zerolinecolor: "#334155" },
    zaxis: { title: zTitle, gridcolor: "#1e293b", zerolinecolor: "#334155" },
    bgcolor: "rgba(15, 23, 42, 0.35)",
  });

  if (activeGraph === "landscape") {
    const trace = [
      {
        type: "scatter3d",
        mode: "markers",
        x: landscape.x_magnitude,
        y: landscape.y_duration,
        z: landscape.z_reversibility,
        text: landscape.case_id.map(
          (id, i) =>
            `<b>${id}</b> (${landscape.dimension[i]})<br>Category: ${landscape.category[i]}<br>Severity: ${landscape.severity[i]}<br>People: ${landscape.people[i]}`
        ),
        hoverinfo: "text",
        marker: {
          size: landscape.people.map((p) => Math.min(16, Math.max(5, Math.log10(p + 1) * 5.5))),
          color: landscape.severity,
          colorscale: "Turbo",
          cmin: 0,
          cmax: 1,
          opacity: 0.88,
          colorbar: { title: "Severity", thickness: 14, len: 0.7 },
        },
      },
    ];
    return (
      <Plotly3DCanvas
        data={trace}
        layout={{ scene: sceneStyle("Magnitude (M)", "Duration (D)", "Reversibility (R)") }}
      />
    );
  }

  if (activeGraph === "surface") {
    const traces = [
      {
        type: "surface",
        x: risk_surface.x_ai_only_pct,
        y: risk_surface.y_safeguard_score,
        z: risk_surface.z_expected_burden_1k,
        colorscale: "Viridis",
        opacity: 0.9,
        colorbar: { title: "Burden / 1k", thickness: 14, len: 0.7 },
      },
    ];

    if (livePrediction) {
      traces.push({
        type: "scatter3d",
        mode: "markers",
        name: "Live Scenario",
        x: [livePrediction.inputs.AI_Only_Work_Percentage],
        y: [livePrediction.inputs.Safeguard_Score],
        z: [livePrediction.dimensions.SOVEREIGNTY.expected_burden_per_1000],
        marker: { size: 11, color: "#f59e0b", line: { color: "#ffffff", width: 2.5 } },
        text: [
          `<b>Live Tweaked Scenario</b><br>AI-Only: ${livePrediction.inputs.AI_Only_Work_Percentage}%<br>Safeguard: ${livePrediction.inputs.Safeguard_Score}/10<br>P(Harm): ${livePrediction.predicted_harm_probability}<br>Exp. Burden/1k: ${livePrediction.dimensions.SOVEREIGNTY.expected_burden_per_1000}`,
        ],
        hoverinfo: "text",
      });
    }

    return (
      <Plotly3DCanvas
        data={traces}
        layout={{ scene: sceneStyle("AI-Only Work %", "Safeguard Score (0–10)", "Exp. Burden / 1k") }}
      />
    );
  }

  // Graph 3: Clean Hover-Only 3D Pareto Cube (no overlapping text clutter)
  const pts = pareto_cube.points || [];
  const paretoTrace = [
    {
      type: "scatter3d",
      mode: "markers",
      x: pts.map((p) => p.x_emp_burden),
      y: pts.map((p) => p.y_sov_burden),
      z: pts.map((p) => p.z_benefit_gain),
      text: pts.map(
        (p) =>
          `<b>${p.label}</b><br>Employment Burden/1k: ${p.x_emp_burden}<br>Sovereignty Burden/1k: ${p.y_sov_burden}<br>Independent Benefit Gain: +${p.z_benefit_gain}%<br>Deployments: ${p.deployments}`
      ),
      hoverinfo: "text",
      marker: {
        size: pts.map((p) => Math.max(9, p.deployments * 1.8)),
        color: pts.map((p) => p.z_benefit_gain),
        colorscale: "Emerald",
        opacity: 0.92,
        line: { color: "#e2e8f0", width: 1 },
        colorbar: { title: "Benefit %", thickness: 14, len: 0.7 },
      },
    },
  ];

  return (
    <Plotly3DCanvas
      data={paretoTrace}
      layout={{ scene: sceneStyle("Emp Burden / 1k", "Sov Burden / 1k", "Independent Benefit %") }}
    />
  );
}