import json
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
from math_engine import function_1_existing_impact, function_4_safeguard_comparison, agent_3_rule_based_evaluator
from ml_predictor import AEGISPredictor
from agents import agent_2_chat_advisor, run_single_report_pipeline

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")

app = FastAPI(title="AEGIS-HIA Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Train lightweight ML models once at startup (<1s)
predictor = AEGISPredictor(data_dir=DATA_DIR)


class ReportRequest(BaseModel):
    report_text: str = Field(..., min_length=10, max_length=3000)


class PredictRequest(BaseModel):
    ai_only_pct: float = Field(50.0, ge=0.0, le=100.0)
    safeguard_score: float = Field(5.0, ge=0.0, le=10.0)
    field: str = "Recruitment & HR"
    oversight: str = "ON_REQUEST"
    appeal: str = "NONE"
    capability: str = "ML_ASSISTED"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=1500)
    context: dict = {}

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    return agent_2_chat_advisor(req.message, req.context)

@app.get("/api/health")
def health():
    return {"status": "ok", "framework": "AEGIS-HIA", "agents": 3}


@app.get("/api/function1")
def api_function1(record_type: str = "SYNTHETIC", field: str = "ALL", q: float = 2.0):
    res = function_1_existing_impact(data_dir=DATA_DIR, record_type=record_type, field=field, q=q)
    audit = agent_3_rule_based_evaluator(math_output=res)
    return {"result": res, "evaluator_audit": audit}


@app.get("/api/function4")
def api_function4(safeguard_col: str = "Human_Appeal_Mechanism", val_a: str = "FULL", val_b: str = "NONE"):
    return function_4_safeguard_comparison(
        data_dir=DATA_DIR, safeguard_col=safeguard_col, val_a=val_a, val_b=val_b
    )


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    return predictor.predict_scenario(
        ai_only_pct=req.ai_only_pct,
        safeguard_score=req.safeguard_score,
        field=req.field,
        oversight=req.oversight,
        appeal=req.appeal,
        capability=req.capability,
    )


@app.get("/api/charts3d")
def api_charts3d(field: str = "Recruitment & HR"):
    return {
        "landscape": predictor.get_3d_severity_landscape(limit=350),
        "risk_surface": predictor.get_3d_risk_surface(field=field),
        "pareto_cube": predictor.get_3d_pareto_cube(),
    }


@app.post("/api/analyze-report-stream")
def api_analyze_report_stream(req: ReportRequest):
    """SSE endpoint streaming live 4-stage agent execution traces."""
    def event_generator():
        for event in run_single_report_pipeline(req.report_text):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")