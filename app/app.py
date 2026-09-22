from pathlib import Path
import json, sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import engineer_features

st.set_page_config(page_title="Fleet Predictive Maintenance", page_icon="🔧", layout="wide")
model = joblib.load(ROOT / "models/rul_model.joblib")
anomaly_model = joblib.load(ROOT / "models/anomaly_model.joblib")
metadata = json.loads((ROOT / "models/metadata.json").read_text())

st.title("Fleet Predictive Maintenance")
st.caption("Remaining useful life and anomaly monitoring using NASA C-MAPSS")
st.warning("Educational prototype only—not for safety-critical maintenance decisions.")
uploaded = st.file_uploader("Upload an engine trajectory CSV", type="csv")
source = uploaded if uploaded else ROOT / "data/sample_engine.csv"
data = pd.read_csv(source)
engineered = engineer_features(data)
latest = engineered.groupby("unit_number", as_index=False).tail(1).copy()
latest["predicted_rul"] = model.predict(latest[metadata["features"]]).clip(0, metadata["rul_cap"])
latest["anomaly_score"] = -anomaly_model.decision_function(latest[metadata["sensors"]])
latest["priority"] = pd.cut(latest["predicted_rul"], [-1, 15, 30, 60, 126], labels=["Critical", "Warning", "Monitor", "Normal"])
st.dataframe(latest[["unit_number", "cycle", "predicted_rul", "priority", "anomaly_score"]].sort_values("predicted_rul"), use_container_width=True)
selected = st.selectbox("Inspect engine", sorted(engineered["unit_number"].unique()))
engine = engineered[engineered["unit_number"] == selected]
prediction = latest.loc[latest["unit_number"] == selected].iloc[0]
a, b, c = st.columns(3)
a.metric("Predicted RUL", f"{prediction['predicted_rul']:.1f} cycles")
b.metric("Maintenance priority", str(prediction["priority"]))
c.metric("Anomaly score", f"{prediction['anomaly_score']:.3f}")
sensor = st.selectbox("Sensor trajectory", metadata["sensors"])
st.line_chart(engine.set_index("cycle")[[sensor, f"{sensor}_mean_5"]])
