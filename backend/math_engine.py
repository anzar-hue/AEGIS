import json
import numpy as np
import pandas as pd
from pathlib import Path
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# ==============================================================================
# AEGIS-HIA: DETERMINISTIC MATH ENGINE & PURE RULE-BASED EVALUATOR (AGENT 3)
# ==============================================================================

DEFAULT_WEIGHTS = (0.20, 0.20, 0.20, 0.20, 0.20)
DEFAULT_Q = 2.0

CONFIRMATION_WEIGHTS = {
    "CONFIRMED": 1.00,
    "ALLEGED": 0.25,
    "SUSPECTED": 0.20,
    "UNKNOWN": 0.10,
    "CONTRADICTED": 0.00,
}

CONTROLLED_HARM_CATEGORIES = {
    "JOB_DISPLACEMENT", "EMPLOYMENT_TERMINATION", "WAGE_REDUCTION", "DESKILLING",
    "EMPLOYMENT_INSTABILITY", "WORKPLACE_MONITORING", "OPPORTUNITY_REDUCTION",
    "WORKLOAD_OR_ROLE_CHANGE", "UNAUTHORIZED_DATA_COLLECTION", "DATA_EXPOSURE",
    "PRIVACY_VIOLATION", "DISCRIMINATION", "UNJUSTIFIED_AUTOMATED_DECISION",
    "LOSS_OF_HUMAN_CONTROL", "DENIAL_OF_APPEAL", "LACK_OF_ACCOUNTABILITY",
    "LACK_OF_TRANSPARENCY", "SURVEILLANCE", "AUTONOMY_RESTRICTION", "MANIPULATION"
}


def load_datasets(data_dir="."):
    return {
        "deployments": pd.read_csv(f"{data_dir}/AI_DEPLOYMENT.csv"),
        "cases": pd.read_csv(f"{data_dir}/INCIDENT_CASES.csv"),
        "harms": pd.read_csv(f"{data_dir}/CASE_HARMS.csv"),
        "attributes": pd.read_csv(f"{data_dir}/CASE_HARM_ATTRIBUTES.csv"),
        "benefits": pd.read_csv(f"{data_dir}/CASE_BENEFITS.csv"),
    }


def map_level_to_unit(level: float) -> float:
    """Maps ordinal level L in {1,2,3,4,5} to [0, 1] via (L - 1) / 4."""
    return (float(level) - 1.0) / 4.0


def compute_individual_severity(levels, weights=DEFAULT_WEIGHTS, q=DEFAULT_Q):
    """Deterministic Weighted Power Mean severity over 5 dimensions [N, M, P, D, R]."""
    if any(pd.isna(v) for v in levels):
        return None
    x = np.array([map_level_to_unit(v) for v in levels], dtype=float)
    w = np.array(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {w.sum()}")
    return round(float((np.sum(w * (x ** q))) ** (1.0 / q)), 6)


def combine_case_harms(df_dim):
    """
    Combines multiple harms per Case_ID within one Harm_Dimension:
    - Same Overlap_Group_ID -> take maximum severity (no double counting).
    - Across distinct groups -> 1 - prod(1 - S_g).
    """
    case_rows = []
    for case_id, grp in df_dim.groupby("Case_ID"):
        group_sevs = []
        grouped = grp[grp["Overlap_Group_ID"].notna()]
        standalone = grp[grp["Overlap_Group_ID"].isna()]

        for _, og in grouped.groupby("Overlap_Group_ID"):
            group_sevs.append(og["effective_severity"].max())
        for _, row in standalone.iterrows():
            group_sevs.append(row["effective_severity"])

        combined = 1.0 - float(np.prod([1.0 - s for s in group_sevs]))
        rep_count = grp["Represented_Person_Count"].iloc[0]
        rep_count = 1.0 if pd.isna(rep_count) else float(rep_count)
        case_rows.append({
            "Case_ID": case_id,
            "Deployment_ID": grp["Deployment_ID"].iloc[0],
            "Field": grp["Field"].iloc[0],
            "combined_severity": combined,
            "Represented_Person_Count": rep_count,
        })
    return pd.DataFrame(case_rows)


def weighted_quantile(values, quantiles, sample_weight):
    values = np.array(values, dtype=float)
    sample_weight = np.array(sample_weight, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight) / np.sum(sample_weight)
    return [float(np.interp(q, cdf, values)) for q in quantiles]


def function_1_existing_impact(
    data_dir=".",
    record_type="SYNTHETIC",
    field=None,
    q=DEFAULT_Q,
    weights=DEFAULT_WEIGHTS,
    use_confirmation_weights=True,
):
    """PRIMARY FUNCTION 1: Analysis of Existing AI Impacts (7 core metrics + Benefits)."""
    ds = load_datasets(data_dir)
    cases, harms, attrs, deps, bens = (
        ds["cases"].copy(), ds["harms"].copy(), ds["attributes"].copy(),
        ds["deployments"].copy(), ds["benefits"].copy()
    )

    if record_type != "ALL":
        cases = cases[cases["Record_Type"] == record_type]
        deps = deps[deps["Record_Type"] == record_type]
        bens = bens[bens["Record_Type"] == record_type]

    if field and field != "ALL":
        cases = cases[cases["Field"] == field]
        deps = deps[deps["Field"] == field]

    merged = (
        harms.merge(attrs, on="Case_Harm_ID", how="inner")
             .merge(cases[["Case_ID", "Deployment_ID", "Field", "Represented_Person_Count", "Record_Type"]],
                    on="Case_ID", how="inner")
    )

    dim_cols = ["Nature_Score", "Magnitude_Score", "Scope_Score", "Duration_Score", "Reversibility_Score"]
    merged["recomputed_severity"] = merged[dim_cols].apply(
        lambda r: compute_individual_severity(r.values, weights=weights, q=q), axis=1
    )
    scored = merged[merged["recomputed_severity"].notna()].copy()

    if use_confirmation_weights:
        scored["conf_w"] = scored["Harm_Confirmation_Status"].map(CONFIRMATION_WEIGHTS).fillna(0.10)
        scored["effective_severity"] = scored["recomputed_severity"] * scored["conf_w"]
        scored = scored[scored["conf_w"] > 0]
    else:
        scored["effective_severity"] = scored["recomputed_severity"]

    valid_deps = deps[deps["Exposure_Count"].notna() & (deps["Exposure_Count"] > 0)]
    total_exposure = float(valid_deps["Exposure_Count"].sum()) if len(valid_deps) > 0 else None

    results = {
        "metadata": {
            "record_type": record_type,
            "field": field or "ALL",
            "formula": f"PowerMean(q={q})",
            "weights": list(weights),
            "confirmation_weighted": use_confirmation_weights,
            "total_exposure_units": total_exposure,
            "disclaimer": "Statistical estimates dependent on structured schema; not guaranteed real-world truth.",
        },
        "dimensions": {},
        "independent_benefits": {},
    }

    for dim in ["EMPLOYMENT", "SOVEREIGNTY"]:
        df_dim = scored[scored["Harm_Dimension"] == dim]
        if len(df_dim) == 0:
            results["dimensions"][dim] = {"status": "INSUFFICIENT_EVIDENCE", "affected_people": 0}
            continue

        case_df = combine_case_harms(df_dim)
        w = case_df["Represented_Person_Count"].values
        s = case_df["combined_severity"].values

        affected_people = float(np.sum(w))
        mean_sev = float(np.average(s, weights=w))
        q50, q90 = weighted_quantile(s, [0.50, 0.90], w)
        total_burden = float(np.sum(s * w))

        event_rate = (affected_people / total_exposure) * 1000.0 if total_exposure else None
        norm_burden = (total_burden / total_exposure) * 1000.0 if total_exposure else None

        results["dimensions"][dim] = {
            "affected_people": round(affected_people, 1),
            "distinct_cases": int(len(case_df)),
            "event_rate_per_1000": round(event_rate, 4) if event_rate is not None else "UNKNOWN_EXPOSURE",
            "mean_severity": round(mean_sev, 4),
            "median_severity_q50": round(q50, 4),
            "tail_severity_q90": round(q90, 4),
            "total_severity_burden": round(total_burden, 2),
            "normalized_burden_per_1000": round(norm_burden, 4) if norm_burden is not None else "UNKNOWN_EXPOSURE",
        }

    if field and field != "ALL":
        bens = bens[bens["Case_ID"].isin(set(cases["Case_ID"])) | bens["Deployment_ID"].isin(set(deps["Deployment_ID"]))]

    if len(bens) > 0:
        top_cats = bens.groupby("Benefit_Category").agg(
            count=("Benefit_ID", "count"), avg_raw_value=("Benefit_Raw_Value", "mean")
        ).reset_index().to_dict(orient="records")
        results["independent_benefits"] = {
            "total_benefit_records": int(len(bens)),
            "categories": [
                {
                    "category": r["Benefit_Category"],
                    "count": int(r["count"]),
                    "avg_raw_gain": round(float(r["avg_raw_value"]), 2) if pd.notna(r["avg_raw_value"]) else None,
                }
                for r in top_cats
            ],
        }
    else:
        results["independent_benefits"] = {"total_benefit_records": 0, "categories": []}

    return results


def function_4_safeguard_comparison(
    data_dir=".",
    safeguard_col="Human_Appeal_Mechanism",
    val_a="FULL",
    val_b="NONE",
    record_type="SYNTHETIC",
):
    """SUPPORTING FUNCTION 4: Safeguard & Deployment Scenario Comparison."""
    ds = load_datasets(data_dir)
    deps, cases, harms, attrs, bens = ds["deployments"], ds["cases"], ds["harms"], ds["attributes"], ds["benefits"]

    if record_type != "ALL":
        deps = deps[deps["Record_Type"] == record_type]
        cases = cases[cases["Record_Type"] == record_type]
        bens = bens[bens["Record_Type"] == record_type]

    def eval_scenario(val):
        sub_deps = deps[deps[safeguard_col] == val]
        dep_ids = set(sub_deps["Deployment_ID"])
        exp = float(sub_deps["Exposure_Count"].dropna().sum())
        sub_cases = cases[cases["Deployment_ID"].isin(dep_ids)]
        sub_harms = harms.merge(attrs, on="Case_Harm_ID").merge(
            sub_cases[["Case_ID", "Deployment_ID", "Field", "Represented_Person_Count"]], on="Case_ID"
        )
        dim_cols = ["Nature_Score", "Magnitude_Score", "Scope_Score", "Duration_Score", "Reversibility_Score"]
        sub_harms["recomputed_severity"] = sub_harms[dim_cols].apply(
            lambda r: compute_individual_severity(r.values), axis=1
        )
        sub_harms = sub_harms[sub_harms["recomputed_severity"].notna()].copy()
        sub_harms["conf_w"] = sub_harms["Harm_Confirmation_Status"].map(CONFIRMATION_WEIGHTS).fillna(0.10)
        sub_harms["effective_severity"] = sub_harms["recomputed_severity"] * sub_harms["conf_w"]
        sub_harms = sub_harms[sub_harms["conf_w"] > 0]

        out = {"deployments": int(len(sub_deps)), "exposure": exp}
        for dim in ["EMPLOYMENT", "SOVEREIGNTY"]:
            d_df = sub_harms[sub_harms["Harm_Dimension"] == dim]
            if len(d_df) == 0 or exp == 0:
                out[dim] = {"burden_per_1000": 0.0, "tail_q90": 0.0, "affected_people": 0.0}
            else:
                cdf = combine_case_harms(d_df)
                w, s = cdf["Represented_Person_Count"].values, cdf["combined_severity"].values
                out[dim] = {
                    "burden_per_1000": round((float(np.sum(s * w)) / exp) * 1000.0, 4),
                    "tail_q90": round(weighted_quantile(s, [0.90], w)[0], 4),
                    "affected_people": round(float(np.sum(w)), 1),
                }
        sub_bens = bens[bens["Deployment_ID"].isin(dep_ids) | bens["Case_ID"].isin(set(sub_cases["Case_ID"]))]
        out["benefit_count"] = int(len(sub_bens))
        out["avg_benefit_raw_gain"] = round(float(sub_bens["Benefit_Raw_Value"].mean()), 2) if len(sub_bens) > 0 else None
        return out

    scen_a, scen_b = eval_scenario(val_a), eval_scenario(val_b)
    ratio = lambda a, b: round(a / b, 4) if b else None

    return {
        "comparison": f"{safeguard_col}: {val_a} (Scenario A) vs {val_b} (Scenario B)",
        "scenario_A": scen_a,
        "scenario_B": scen_b,
        "residual_risk_ratio_A_over_B": {
            "EMPLOYMENT": ratio(scen_a["EMPLOYMENT"]["burden_per_1000"], scen_b["EMPLOYMENT"]["burden_per_1000"]),
            "SOVEREIGNTY": ratio(scen_a["SOVEREIGNTY"]["burden_per_1000"], scen_b["SOVEREIGNTY"]["burden_per_1000"]),
        },
        "note": "Association-based scenario comparison on synthetic prototype data; causal claims require causal adjustment.",
    }


def agent_3_rule_based_evaluator(extraction_json=None, math_output=None):
    """AGENT 3: PURE RULE-BASED EVALUATOR (ZERO LLM CALLS)."""
    flags = []

    if extraction_json is not None:
        cat = extraction_json.get("harm_category")
        if cat not in CONTROLLED_HARM_CATEGORIES:
            flags.append(f"TAXONOMY_ERROR: '{cat}' is not in controlled harm taxonomy.")

        status = extraction_json.get("harm_confirmation_status")
        if status not in CONFIRMATION_WEIGHTS:
            flags.append(f"STATUS_ERROR: Invalid confirmation status '{status}'.")

        levels = extraction_json.get("levels", {})
        for k in ["Nature", "Magnitude", "Scope", "Duration", "Reversibility"]:
            v = levels.get(k)
            if v is None or not isinstance(v, (int, float)) or not (1 <= v <= 5):
                flags.append(f"LEVEL_OUT_OF_RANGE: Dimension '{k}' has invalid value '{v}' (must be 1–5).")

        preds = extraction_json.get("predicates", [])
        if status == "CONFIRMED":
            if any(p.get("assertion_status") in {"NEGATED", "HYPOTHETICAL"} for p in preds):
                flags.append("ASSERTION_CONFLICT: Confirmed harm cannot rely on NEGATED or HYPOTHETICAL predicates.")

    if math_output is not None:
        w = math_output.get("metadata", {}).get("weights", [])
        if w and not np.isclose(sum(w), 1.0, atol=1e-4):
            flags.append(f"WEIGHT_SUM_ERROR: Weights sum to {sum(w)}, must equal 1.0.")
        for dim, vals in math_output.get("dimensions", {}).items():
            for metric in ["mean_severity", "median_severity_q50", "tail_severity_q90"]:
                v = vals.get(metric)
                if isinstance(v, (int, float)) and not (0.0 <= v <= 1.0):
                    flags.append(f"SEVERITY_BOUND_ERROR: {dim}.{metric} = {v} outside [0, 1].")

    return {
        "audit_status": "PASS" if len(flags) == 0 else "FLAGGED_FOR_HUMAN_REVIEW",
        "flag_count": len(flags),
        "flags": flags,
    }


if __name__ == "__main__":
    print(json.dumps(function_1_existing_impact(str(DEFAULT_DATA_DIR)), indent=2))