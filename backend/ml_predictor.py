import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from math_engine import load_datasets, CONFIRMATION_WEIGHTS


def compute_safeguard_score(df: pd.DataFrame) -> np.ndarray:
    """Maps deployment safeguard columns into a transparent 0–10 composite score."""
    score = np.zeros(len(df), dtype=float)
    oversight_map = {
        "NONE": 0, "ON_REQUEST": 1, "HUMAN_ON_LOOP": 2,
        "HUMAN_IN_LOOP": 3, "HUMAN_FINAL_DECISION": 4
    }
    appeal_map = {"NONE": 0, "LIMITED": 1, "FULL": 2}
    priv_map = {"NONE": 0, "BASIC": 1, "STRONG": 2}
    yes_no = {"YES": 1, "NO": 0}

    score += df["Human_Oversight_Level"].map(oversight_map).fillna(0).values
    score += df["Human_Appeal_Mechanism"].map(appeal_map).fillna(0).values
    score += df["Privacy_Safeguards"].map(priv_map).fillna(0).values
    score += df["Independent_Audit"].map(yes_no).fillna(0).values
    score += df["Continuous_Monitoring"].map(yes_no).fillna(0).values
    return np.clip(score, 0, 10)


class AEGISPredictor:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.num_features = ["AI_Only_Work_Percentage", "Safeguard_Score"]
        self.cat_features = [
            "Field", "Human_Oversight_Level", "Human_Appeal_Mechanism", "AI_Capability_Level"
        ]
        self.clf_pipe = None
        self.reg_models = {}
        self.ds = load_datasets(data_dir)
        self._fit_models()

    def _fit_models(self):
        deps = self.ds["deployments"]
        cases = self.ds["cases"]
        harms = self.ds["harms"]
        attrs = self.ds["attributes"]

        # Train strictly on SYNTHETIC active/pilot records (never fit on DOCUMENTED)
        deps_syn = deps[
            (deps["Record_Type"] == "SYNTHETIC")
            & (deps["Deployment_Status"].isin(["ACTIVE", "PILOT"]))
        ].copy()
        deps_syn["Safeguard_Score"] = compute_safeguard_score(deps_syn)

        cases_syn = cases[cases["Record_Type"] == "SYNTHETIC"].copy()
        harms_syn = harms.merge(attrs, on="Case_Harm_ID").merge(
            cases_syn[["Case_ID", "Deployment_ID", "Field", "Represented_Person_Count"]],
            on="Case_ID",
        )

        # 1. Deployment-level Harm Occurrence Classifier
        confirmed_dep_ids = set(
            harms_syn[harms_syn["Harm_Confirmation_Status"] == "CONFIRMED"]["Deployment_ID"].dropna()
        )
        deps_syn["Has_Confirmed_Harm"] = deps_syn["Deployment_ID"].isin(confirmed_dep_ids).astype(int)

        X_clf = deps_syn[self.num_features + self.cat_features].fillna({
            "AI_Only_Work_Percentage": 0.0, "Safeguard_Score": 0.0,
            "Field": "UNKNOWN", "Human_Oversight_Level": "NONE",
            "Human_Appeal_Mechanism": "NONE", "AI_Capability_Level": "UNKNOWN",
        })
        y_clf = deps_syn["Has_Confirmed_Harm"].values

        pre_clf = ColumnTransformer([
            ("num", StandardScaler(), self.num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
        ])
        self.clf_pipe = Pipeline([
            ("pre", pre_clf),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ])
        self.clf_pipe.fit(X_clf, y_clf)

        # 2. Conditional Severity Regressors (Ridge) for EMPLOYMENT & SOVEREIGNTY
        merged_reg = harms_syn.merge(
            deps_syn[[
                "Deployment_ID", "AI_Only_Work_Percentage", "Safeguard_Score",
                "Human_Oversight_Level", "Human_Appeal_Mechanism", "AI_Capability_Level"
            ]],
            on="Deployment_ID",
            how="inner",
        )

        for dim, col in [("EMPLOYMENT", "Employment_Severity"), ("SOVEREIGNTY", "Sovereignty_Severity")]:
            sub = merged_reg[(merged_reg["Harm_Dimension"] == dim) & (merged_reg[col].notna())].copy()
            X_reg = sub[self.num_features + self.cat_features].fillna("UNKNOWN")
            y_reg = sub[col].values
            pre_reg = ColumnTransformer([
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
            ])
            pipe = Pipeline([("pre", pre_reg), ("reg", Ridge(alpha=1.0))])
            pipe.fit(X_reg, y_reg)
            preds = np.clip(pipe.predict(X_reg), 0.0, 1.0)
            resid_std = float(np.std(y_reg - preds))
            self.reg_models[dim] = (pipe, resid_std)

    def predict_scenario(
        self,
        ai_only_pct: float,
        safeguard_score: float,
        field: str = "Recruitment & HR",
        oversight: str = "ON_REQUEST",
        appeal: str = "NONE",
        capability: str = "ML_ASSISTED",
    ) -> dict:
        row = pd.DataFrame([{
            "AI_Only_Work_Percentage": float(ai_only_pct),
            "Safeguard_Score": float(safeguard_score),
            "Field": field,
            "Human_Oversight_Level": oversight,
            "Human_Appeal_Mechanism": appeal,
            "AI_Capability_Level": capability,
        }])
        assert self.clf_pipe is not None

        p_harm = float(self.clf_pipe.predict_proba(row)[0, 1])
        out = {
            "record_type": "MODEL_ESTIMATE (SYNTHETIC_PROTOTYPE)",
            "inputs": row.iloc[0].to_dict(),
            "predicted_harm_probability": round(p_harm, 4),
            "dimensions": {},
            "disclaimer": "Probabilistic estimate trained on synthetic prototype data; not a guaranteed real-world forecast.",
        }

        for dim, (pipe, rstd) in self.reg_models.items():
            s_hat = float(np.clip(pipe.predict(row)[0], 0.0, 1.0))
            ci_low = float(np.clip(s_hat - 1.96 * rstd, 0.0, 1.0))
            ci_high = float(np.clip(s_hat + 1.96 * rstd, 0.0, 1.0))
            exp_burden_1k = p_harm * s_hat * 1000.0
            out["dimensions"][dim] = {
                "expected_severity": round(s_hat, 4),
                "severity_95_ci": [round(ci_low, 4), round(ci_high, 4)],
                "expected_burden_per_1000": round(exp_burden_1k, 2),
            }
        return out

    # --- 3D GRAPH 1: 3D SEVERITY LANDSCAPE (Function 1) ---
    def get_3d_severity_landscape(self, limit: int = 400) -> dict:
        cases = self.ds["cases"]
        harms = self.ds["harms"]
        attrs = self.ds["attributes"]

        df = harms.merge(attrs, on="Case_Harm_ID").merge(
            cases[["Case_ID", "Field", "AI_System", "Represented_Person_Count", "Record_Type"]],
            on="Case_ID",
        )
        df = df[df["Record_Type"] == "SYNTHETIC"].copy()
        df["Severity"] = df["Employment_Severity"].fillna(df["Sovereignty_Severity"])
        df = df[df["Severity"].notna()].head(limit)

        return {
            "x_magnitude": df["Magnitude_x"].round(3).tolist(),
            "y_duration": df["Duration_x"].round(3).tolist(),
            "z_reversibility": df["Reversibility_x"].round(3).tolist(),
            "severity": df["Severity"].round(4).tolist(),
            "dimension": df["Harm_Dimension"].tolist(),
            "category": df["Harm_Category"].tolist(),
            "case_id": df["Case_ID"].tolist(),
            "field": df["Field"].tolist(),
            "people": df["Represented_Person_Count"].fillna(1).astype(int).tolist(),
        }

    # --- 3D GRAPH 2: 3D PREDICTIVE RISK SURFACE (Function 2 ML) ---
    def get_3d_risk_surface(self, field: str = "Recruitment & HR") -> dict:
        ai_grid = np.linspace(10, 90, 15)
        sg_grid = np.linspace(0, 10, 15)
        z_surface = []

        for sg in sg_grid:
            row_vals = []
            for ai_pct in ai_grid:
                pred = self.predict_scenario(ai_only_pct=ai_pct, safeguard_score=sg, field=field)
                # Combined expected risk surface (Sovereignty expected burden per 1k)
                row_vals.append(pred["dimensions"]["SOVEREIGNTY"]["expected_burden_per_1000"])
            z_surface.append(row_vals)

        return {
            "x_ai_only_pct": [round(float(v), 1) for v in ai_grid],
            "y_safeguard_score": [round(float(v), 1) for v in sg_grid],
            "z_expected_burden_1k": z_surface,
            "field": field,
        }

    # --- 3D GRAPH 3: 3D HARM-BENEFIT PARETO CUBE (Function 4) ---
    def get_3d_pareto_cube(self) -> dict:
        deps = self.ds["deployments"]
        cases = self.ds["cases"]
        harms = self.ds["harms"]
        bens = self.ds["benefits"]

        deps_syn = deps[(deps["Record_Type"] == "SYNTHETIC") & (deps["Exposure_Count"].notna())].copy()
        points = []
        for appeal_val in ["FULL", "LIMITED", "NONE"]:
            for oversight_val in ["HUMAN_FINAL_DECISION", "HUMAN_IN_LOOP", "ON_REQUEST", "NONE"]:
                sub_d = deps_syn[
                    (deps_syn["Human_Appeal_Mechanism"] == appeal_val)
                    & (deps_syn["Human_Oversight_Level"] == oversight_val)
                ]
                if len(sub_d) == 0:
                    continue
                exp = float(sub_d["Exposure_Count"].sum())
                d_ids = set(sub_d["Deployment_ID"])
                sub_c = cases[cases["Deployment_ID"].isin(d_ids)]
                sub_h = harms.merge(sub_c[["Case_ID", "Represented_Person_Count"]], on="Case_ID")
                sub_h["conf_w"] = sub_h["Harm_Confirmation_Status"].map(CONFIRMATION_WEIGHTS).fillna(0.1)

                emp_df = sub_h[sub_h["Harm_Dimension"] == "EMPLOYMENT"]
                sov_df = sub_h[sub_h["Harm_Dimension"] == "SOVEREIGNTY"]

                emp_b = float((emp_df["Employment_Severity"].fillna(0) * emp_df["conf_w"] * emp_df["Represented_Person_Count"].fillna(1)).sum() / exp * 1000)
                sov_b = float((sov_df["Sovereignty_Severity"].fillna(0) * sov_df["conf_w"] * sov_df["Represented_Person_Count"].fillna(1)).sum() / exp * 1000)

                sub_b = bens[bens["Deployment_ID"].isin(d_ids) | bens["Case_ID"].isin(set(sub_c["Case_ID"]))]
                ben_gain = float(sub_b["Benefit_Raw_Value"].mean()) if len(sub_b) > 0 else 0.0

                points.append({
                    "label": f"Appeal:{appeal_val} | Oversight:{oversight_val}",
                    "x_emp_burden": round(emp_b, 4),
                    "y_sov_burden": round(sov_b, 4),
                    "z_benefit_gain": round(ben_gain, 2),
                    "deployments": int(len(sub_d)),
                })
        return {"points": points}