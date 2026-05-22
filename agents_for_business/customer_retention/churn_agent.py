"""
LangGraph Churn Monitoring Agent

Graph flow:
  START → predict → identify_churners ──(none)──→ END
                                       ──(found)──→ explain_shap → draft_emails → output → END

Each churner gets:
  - SHAP top-5 features explaining why they will churn
  - A personalised retention email written by Claude
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = "churn_model.pkl"
META_PATH     = "churn_metadata.pkl"
DATA_PATH     = "telco_customer_churn_train.csv"
CHURN_THRESHOLD = 0.55   # flag customers above this probability
TOP_N_FEATURES  = 5      # SHAP features to include in the email prompt
DEMO_SAMPLE     = 200    # rows to score (set None to score all)

# ── State ─────────────────────────────────────────────────────────────────────

class ChurnState(TypedDict):
    customer_df: Any                              # raw DataFrame
    pipeline: Any                                 # trained sklearn Pipeline
    metadata: Dict[str, Any]                      # numeric_cols, cat_cols, feature_names
    predictions: Dict[str, float]                 # customerID → churn probability
    churners: List[str]                           # high-risk customer IDs
    shap_explanations: Dict[str, List[Dict]]      # customerID → [{feature, value, shap}]
    emails: List[Dict[str, str]]                  # [{customer_id, subject, body}]
    status: str

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "12-24 mo", "24-48 mo", "48-72 mo"], include_lowest=True,
    )
    df["charges_ratio"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


def _readable_feature(name: str) -> str:
    """Convert OHE feature name to human-readable text."""
    name = name.replace("_", " ").replace("  ", " ")
    # e.g. "Contract Month-to-month" → already readable
    return name.title()


def _feature_impact_desc(feature: str, shap_val: float, raw_val: Any) -> str:
    direction = "increases" if shap_val > 0 else "decreases"
    return f"{_readable_feature(feature)} = '{raw_val}'  ({direction} churn risk, impact {shap_val:+.3f})"


# ── Node 1 – predict ──────────────────────────────────────────────────────────

def predict_node(state: ChurnState) -> ChurnState:
    print("\n[Agent] Scoring customers…")
    pipeline = state["pipeline"]
    df = state["customer_df"]
    meta = state["metadata"]

    X = df.drop(columns=["customerID", "Churn"], errors="ignore")
    probs = pipeline.predict_proba(X)[:, 1]

    predictions = {row["customerID"]: float(prob)
                   for row, prob in zip(df.to_dict("records"), probs)}
    print(f"[Agent] Scored {len(predictions)} customers.")
    return {**state, "predictions": predictions}


# ── Node 2 – identify churners ────────────────────────────────────────────────

def identify_churners_node(state: ChurnState) -> ChurnState:
    threshold = CHURN_THRESHOLD
    churners = [cid for cid, prob in state["predictions"].items() if prob >= threshold]
    print(f"[Agent] {len(churners)} customers above {threshold:.0%} churn threshold.")
    status = "found" if churners else "none"
    return {**state, "churners": churners, "status": status}


# ── Node 3 – SHAP explanations ────────────────────────────────────────────────

def shap_node(state: ChurnState) -> ChurnState:
    print(f"[Agent] Computing SHAP for {len(state['churners'])} churners…")
    pipeline  = state["pipeline"]
    meta      = state["metadata"]
    df        = state["customer_df"]
    churners  = state["churners"]
    feature_names = meta["feature_names"]

    prep = pipeline.named_steps["prep"]
    clf  = pipeline.named_steps["clf"]

    # Filter to churners only
    churn_df = df[df["customerID"].isin(churners)].reset_index(drop=True)
    X_churn  = churn_df.drop(columns=["customerID", "Churn"], errors="ignore")
    X_transformed = prep.transform(X_churn)

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)  # (n, features)

    explanations: Dict[str, List[Dict]] = {}
    for i, cid in enumerate(churn_df["customerID"]):
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:TOP_N_FEATURES]

        row = churn_df.iloc[i]
        top_feats = []
        for fi in top_idx:
            fname = feature_names[fi]
            raw_val = X_transformed[i, fi]
            # Try to recover original value for categoricals
            if fi >= len(meta["numeric_cols"]):
                # OHE column — value is 1/0; label is embedded in the name
                raw_val = "Yes" if raw_val == 1.0 else "No"
            top_feats.append({
                "feature": fname,
                "raw_value": raw_val,
                "shap_value": float(sv[fi]),
            })
        explanations[cid] = top_feats

    print("[Agent] SHAP complete.")
    return {**state, "shap_explanations": explanations}


# ── Node 4 – draft emails ─────────────────────────────────────────────────────

def draft_emails_node(state: ChurnState) -> ChurnState:
    print("[Agent] Drafting retention emails with OpenAI…")
    client = OpenAI()   # uses OPENAI_API_KEY from env
    df     = state["customer_df"]
    emails = []

    for cid in state["churners"]:
        prob  = state["predictions"][cid]
        feats = state["shap_explanations"][cid]
        row   = df[df["customerID"] == cid].iloc[0].to_dict()

        # Build SHAP factor bullet list
        factor_lines = "\n".join(
            f"  {j+1}. {_feature_impact_desc(f['feature'], f['shap_value'], f['raw_value'])}"
            for j, f in enumerate(feats)
        )

        prompt = f"""You are a customer retention specialist at a telecom company called TelConnect.
A valued customer has been flagged as at high risk of leaving.

Customer Profile
----------------
Customer ID    : {cid}
Gender         : {row.get('gender', 'N/A')}
Senior Citizen : {'Yes' if row.get('SeniorCitizen') == 1 else 'No'}
Partner        : {row.get('Partner', 'N/A')}
Dependents     : {row.get('Dependents', 'N/A')}
Tenure         : {row.get('tenure', 'N/A')} months
Contract       : {row.get('Contract', 'N/A')}
Internet       : {row.get('InternetService', 'N/A')}
Monthly Charges: ${row.get('MonthlyCharges', 'N/A')}
Payment Method : {row.get('PaymentMethod', 'N/A')}

Churn Probability: {prob:.1%}

Top factors driving churn risk (SHAP analysis):
{factor_lines}

Task
----
Write a warm, personalised retention email that:
1. Opens with empathy — do NOT mention "churn" or "churn model"
2. Addresses the specific pain points revealed by the SHAP factors above
3. Offers a concrete, relevant incentive (discount, upgrade, contract flexibility, etc.)
4. Ends with a clear call-to-action (reply, call, link)
5. Is professional yet friendly — max 200 words in the body

Return your answer in this exact JSON format (no markdown fences):
{{
  "subject": "<email subject line>",
  "body": "<full email body>"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract JSON block if Claude wrapped it in text
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {"subject": "We miss you!", "body": raw}

        emails.append({
            "customer_id": cid,
            "churn_probability": f"{prob:.1%}",
            "subject": parsed.get("subject", ""),
            "body": parsed.get("body", ""),
        })
        print(f"  ✓ Email drafted for {cid}  (churn prob {prob:.1%})")

    return {**state, "emails": emails}


# ── Node 5 – output ───────────────────────────────────────────────────────────

def output_node(state: ChurnState) -> ChurnState:
    emails = state["emails"]
    out_path = "churn_emails.json"
    with open(out_path, "w") as f:
        json.dump(emails, f, indent=2)
    print(f"\n[Agent] {len(emails)} retention emails saved → {out_path}")

    # Print preview of first email
    if emails:
        e = emails[0]
        print(f"\n── Preview: {e['customer_id']} (prob {e['churn_probability']}) ──")
        print(f"Subject : {e['subject']}")
        print(f"Body    :\n{e['body']}")

    return {**state, "status": "done"}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_identify(state: ChurnState) -> str:
    return state["status"]   # "found" or "none"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(ChurnState)
    g.add_node("predict",          predict_node)
    g.add_node("identify_churners", identify_churners_node)
    g.add_node("explain_shap",     shap_node)
    g.add_node("draft_emails",     draft_emails_node)
    g.add_node("output",           output_node)

    g.set_entry_point("predict")
    g.add_edge("predict", "identify_churners")
    g.add_conditional_edges(
        "identify_churners",
        route_after_identify,
        {"found": "explain_shap", "none": END},
    )
    g.add_edge("explain_shap", "draft_emails")
    g.add_edge("draft_emails", "output")
    g.add_edge("output", END)

    return g.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model & metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found — run churn_prediction.py first to train and save the model."
        )
    pipeline = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)

    # Load and prepare customer data
    df_raw = pd.read_csv(DATA_PATH)
    if DEMO_SAMPLE:
        # Sample from customers who actually churned + random non-churners for a realistic demo
        churned    = df_raw[df_raw["Churn"] == 1].sample(min(50, (df_raw["Churn"] == 1).sum()), random_state=1)
        non_churned = df_raw[df_raw["Churn"] == 0].sample(DEMO_SAMPLE - len(churned), random_state=1)
        df_raw = pd.concat([churned, non_churned]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_clean = _clean(df_raw)

    # Initial state
    initial_state: ChurnState = {
        "customer_df":      df_clean,
        "pipeline":         pipeline,
        "metadata":         metadata,
        "predictions":      {},
        "churners":         [],
        "shap_explanations": {},
        "emails":           [],
        "status":           "",
    }

    app = build_graph()
    print("=" * 60)
    print("  TelConnect Churn Monitoring Agent")
    print("=" * 60)
    final_state = app.invoke(initial_state)
    print("\n[Agent] Complete. Status:", final_state["status"])
