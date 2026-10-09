"""Streamlit dashboard. Talks to the FastAPI service only (no direct DB / model access).

    API_URL=http://localhost:8000 streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

COLORS = {"LOW": "#2a9d8f", "MEDIUM": "#e9c46a", "HIGH": "#e63946",
          "LEGITIMATE": "#2a6f97", "FRAUD": "#e63946"}

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    pass


def _request(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL} ({exc.__class__.__name__}). "
                       "Is the backend running?") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"API error {resp.status_code}: {detail}")
    return resp.json()


@st.cache_data(ttl=15, show_spinner=False)
def get_stats() -> dict:
    return _request("GET", "/stats")


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info() -> dict:
    return _request("GET", "/model-info")


@st.cache_data(ttl=15, show_spinner=False)
def get_history(params: tuple) -> dict:
    return _request("GET", "/history", params=dict(params))


def history_df(params: dict | None = None, limit: int = 5000) -> tuple[int, pd.DataFrame]:
    q = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    q["limit"] = limit
    data = get_history(tuple(sorted(q.items())))
    df = pd.DataFrame(data["items"])
    if df.empty:
        return data["total"], df
    df["transaction_time"] = pd.to_datetime(df["transaction_time"].fillna(df["timestamp"]))
    df["status"] = df["prediction"].map({0: "LEGITIMATE", 1: "FRAUD"})
    return data["total"], df


def guard(fn):
    """Run a page; show API problems as a friendly message instead of a traceback."""
    try:
        fn()
    except ApiError as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.header("Dashboard Overview")
    s = get_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total transactions", f"{s['total_transactions']:,}")
    c2.metric("Fraud transactions", f"{s['fraud_transactions']:,}")
    c3.metric("Fraud %", f"{s['fraud_percentage']:.2f}%")
    c4.metric("Avg risk score", f"{s['average_risk_score']:.1f}")
    c5.metric("High-risk", f"{s['high_risk_transactions']:,}")

    if s["total_transactions"] == 0:
        st.info("No transactions analysed yet. Use **Transaction Prediction**, or run "
                "`python scripts/seed_demo.py` to load demo transactions.")
        return

    left, right = st.columns(2)
    dist = pd.DataFrame({"risk_level": list(s["risk_distribution"]),
                         "count": list(s["risk_distribution"].values())})
    fig = px.pie(dist, names="risk_level", values="count", hole=0.5, color="risk_level",
                 color_discrete_map=COLORS, title="Risk level mix")
    left.plotly_chart(fig, use_container_width=True)

    trend = pd.DataFrame(s["daily_trend"])
    if not trend.empty:
        fig = go.Figure()
        fig.add_bar(x=trend["day"], y=trend["total"], name="All", marker_color="#8ecae6")
        fig.add_bar(x=trend["day"], y=trend["fraud"], name="Flagged fraud", marker_color=COLORS["FRAUD"])
        fig.update_layout(barmode="overlay", title="Daily volume", legend_orientation="h")
        right.plotly_chart(fig, use_container_width=True)


PRESETS = {
    "Typical purchase": dict(amount=42.5, merchant_category="grocery", payment_channel="pos",
                             device_type="pos_terminal", is_new_device=False, account_age_days=900,
                             distance_from_home_km=4.0, country_mismatch=False, txn_count_24h=1,
                             avg_amount_30d=55.0, time_since_last_txn_min=600.0, hour=14),
    "Suspicious purchase": dict(amount=1850.0, merchant_category="electronics", payment_channel="online",
                                device_type="desktop", is_new_device=True, account_age_days=9,
                                distance_from_home_km=830.0, country_mismatch=True, txn_count_24h=8,
                                avg_amount_30d=70.0, time_since_last_txn_min=2.0, hour=3),
}
MERCHANTS = ["grocery", "restaurant", "fuel", "travel", "electronics", "clothing",
             "entertainment", "health", "online_services"]
CHANNELS = ["pos", "online", "mobile_app", "atm"]
DEVICES = ["mobile", "desktop", "tablet", "pos_terminal"]


def risk_gauge(score: int, thresholds: dict) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": "Risk score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#222"},
               "steps": [{"range": [0, thresholds["low_max"]], "color": COLORS["LOW"]},
                         {"range": [thresholds["low_max"], thresholds["medium_max"]], "color": COLORS["MEDIUM"]},
                         {"range": [thresholds["medium_max"], 100], "color": COLORS["HIGH"]}]}))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def shap_chart(result: dict) -> go.Figure:
    items = result["top_increasing_features"] + result["top_decreasing_features"]
    items = sorted(items, key=lambda c: c["shap_value"])
    fig = go.Figure(go.Bar(
        x=[c["shap_value"] for c in items],
        y=[f"{c['label']} = {c['value']}" for c in items], orientation="h",
        marker_color=[COLORS["FRAUD"] if c["shap_value"] > 0 else COLORS["LEGITIMATE"] for c in items]))
    fig.update_layout(title="SHAP: what moved this prediction (log-odds)", height=70 * max(len(items), 4),
                      xaxis_title="← lowers risk | raises risk →", margin=dict(l=10))
    return fig


def page_predict() -> None:
    st.header("Transaction Prediction")
    info = get_model_info()
    preset_name = st.radio("Start from", list(PRESETS), horizontal=True)
    p = PRESETS[preset_name]

    with st.form("txn_form"):
        k = preset_name  # changing preset changes widget keys -> defaults refresh
        a, b, c = st.columns(3)
        amount = a.number_input("Amount", 0.01, 1_000_000.0, float(p["amount"]), key=f"amt{k}")
        merchant = b.selectbox("Merchant category", MERCHANTS, MERCHANTS.index(p["merchant_category"]), key=f"m{k}")
        channel = c.selectbox("Payment channel", CHANNELS, CHANNELS.index(p["payment_channel"]), key=f"c{k}")
        d, e, f = st.columns(3)
        device = d.selectbox("Device type", DEVICES, DEVICES.index(p["device_type"]), key=f"d{k}")
        tx_date = e.date_input("Transaction date", date.today(), key=f"dt{k}")
        hour = f.slider("Hour of day", 0, 23, int(p["hour"]), key=f"h{k}")
        g, h, i = st.columns(3)
        account_age = g.number_input("Account age (days)", 0, 50_000, int(p["account_age_days"]), key=f"a{k}")
        distance = h.number_input("Distance from home (km)", 0.0, 25_000.0, float(p["distance_from_home_km"]), key=f"di{k}")
        avg30 = i.number_input("Customer 30-day avg amount", 0.0, 1_000_000.0, float(p["avg_amount_30d"]), key=f"av{k}")
        j, l, m = st.columns(3)
        txn24 = j.number_input("Transactions in last 24h", 0, 10_000, int(p["txn_count_24h"]), key=f"t{k}")
        since = l.number_input("Minutes since previous txn", 0.0, 1_000_000.0, float(p["time_since_last_txn_min"]), key=f"s{k}")
        n1, n2 = m.columns(2)
        new_dev = n1.checkbox("New device", p["is_new_device"], key=f"nd{k}")
        mismatch = n2.checkbox("Country mismatch", p["country_mismatch"], key=f"cm{k}")
        submitted = st.form_submit_button("Analyse transaction", type="primary")

    if not submitted:
        return

    payload = {
        "timestamp": datetime.combine(tx_date, datetime.min.time()).replace(hour=hour).isoformat(),
        "amount": amount, "merchant_category": merchant, "payment_channel": channel,
        "device_type": device, "is_new_device": new_dev, "account_age_days": int(account_age),
        "distance_from_home_km": distance, "country_mismatch": mismatch,
        "txn_count_24h": int(txn24), "avg_amount_30d": avg30, "time_since_last_txn_min": since,
    }
    result = _request("POST", "/predict", json=payload)
    get_stats.clear()
    get_history.clear()

    level = result["risk_level"]
    st.markdown(f"### Result: <span style='color:{COLORS[result['prediction_label']]}'>"
                f"{result['prediction_label']}</span>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
    m2.metric("Risk score", f"{result['risk_score']} / 100")
    m3.metric("Risk level", level)
    if not result["stored"]:
        st.warning("Prediction was not saved to the database (see API logs).")

    gcol, scol = st.columns([1, 2])
    gcol.plotly_chart(risk_gauge(result["risk_score"], info["risk_thresholds"]), use_container_width=True)
    scol.info(result["explanation"])
    st.plotly_chart(shap_chart(result), use_container_width=True)
    st.caption(f"Transaction {result['transaction_id']} · model {result['model_version']}")


def page_analytics() -> None:
    st.header("Analytics")
    info = get_model_info()
    total, df = history_df()

    t1, t2 = st.tabs(["Transactions", "Model"])
    with t1:
        if df.empty:
            st.info("No stored transactions yet.")
        else:
            c1, c2 = st.columns(2)
            counts = df["status"].value_counts().reset_index()
            c1.plotly_chart(px.bar(counts, x="status", y="count", color="status", color_discrete_map=COLORS,
                                   title="Fraud vs legitimate"), use_container_width=True)
            risk = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).reset_index()
            c2.plotly_chart(px.bar(risk, x="risk_level", y="count", color="risk_level", color_discrete_map=COLORS,
                                   title="Risk distribution"), use_container_width=True)
            st.plotly_chart(px.histogram(df, x="amount", color="status", nbins=50, log_y=True, barmode="overlay",
                                         color_discrete_map=COLORS, title="Transaction amount distribution"),
                            use_container_width=True)
            daily = (df.assign(day=df["transaction_time"].dt.date)
                     .groupby("day").agg(total=("prediction", "size"), fraud=("prediction", "sum")).reset_index())
            daily["fraud_rate_%"] = 100 * daily["fraud"] / daily["total"]
            st.plotly_chart(px.line(daily, x="day", y=["fraud", "fraud_rate_%"], markers=True,
                                    title="Fraud trend over time"), use_container_width=True)
            if total > len(df):
                st.caption(f"Charts use the most recent {len(df):,} of {total:,} stored transactions.")

    with t2:
        comp = pd.DataFrame(info["comparison"])
        test = comp[comp["Split"] == "test"]
        metrics = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        long = test.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                               title="Model performance (held-out test set)"), use_container_width=True)
        st.dataframe(test[["Model", "Threshold", *metrics, "TP", "FP", "FN", "TN"]].round(4), hide_index=True)

        tm = info["test_metrics"]
        cm = [[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                                  color_continuous_scale="Blues", title=f"Confusion matrix – {info['model_type']}"),
                        use_container_width=True)
        imp = pd.DataFrame(info["global_importance"]).head(12).iloc[::-1]
        c2.plotly_chart(px.bar(imp, x="mean_abs_shap", y="label", orientation="h",
                               title="Top fraud-related features (mean |SHAP|)"), use_container_width=True)


def page_history() -> None:
    st.header("Transaction History")
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        use_dates = c1.checkbox("Filter by date")
        d_from = d_to = None
        if use_dates:
            rng = c1.date_input("Date range", (date.today() - timedelta(days=30), date.today()))
            if isinstance(rng, tuple) and len(rng) == 2:
                d_from, d_to = rng
        risk = c2.selectbox("Risk level", ["All", "LOW", "MEDIUM", "HIGH"])
        status_ = c3.selectbox("Fraud status", ["All", "FRAUD", "LEGITIMATE"])
        min_amt = c4.number_input("Min amount", 0.0, value=0.0)
        max_amt = c4.number_input("Max amount (0 = no limit)", 0.0, value=0.0)

    params = {
        "date_from": d_from.isoformat() if d_from else None,
        "date_to": d_to.isoformat() if d_to else None,
        "risk_level": None if risk == "All" else risk,
        "prediction": {"All": None, "FRAUD": 1, "LEGITIMATE": 0}[status_],
        "min_amount": min_amt or None,
        "max_amount": max_amt or None,
    }
    total, df = history_df(params, limit=1000)
    st.write(f"**{total:,}** matching transactions" + (f" (showing latest {len(df):,})" if total > len(df) else ""))
    if df.empty:
        return
    view = df[["transaction_id", "transaction_time", "amount", "fraud_probability", "risk_score",
               "risk_level", "status", "model_version"]]
    st.dataframe(view, hide_index=True, use_container_width=True,
                 column_config={"fraud_probability": st.column_config.ProgressColumn(
                     "Fraud probability", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download CSV", view.to_csv(index=False), "transactions.csv", "text/csv")


# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio("Navigate", ["Overview", "Transaction Prediction", "Analytics", "Transaction History"])
try:
    health = _request("GET", "/health")
    st.sidebar.success(f"API online · model {health.get('model_version') or 'not loaded'}")
    if not health["database_connected"]:
        st.sidebar.warning("Database not connected")
except ApiError as exc:
    st.sidebar.error("API offline")
    st.error(str(exc))
    st.stop()

PAGES = {"Overview": page_overview, "Transaction Prediction": page_predict,
         "Analytics": page_analytics, "Transaction History": page_history}
guard(PAGES[page])
