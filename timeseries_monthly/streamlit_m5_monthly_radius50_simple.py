import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_m5_monthly_radius50 import (
    ARTIFACT_DIR,
    DATA_PATH,
    predict_next_month_point,
    predict_next_month_region,
    train_and_save,
)


st.set_page_config(page_title="M5 Monthly Radius50 - Simple Test", layout="wide")


def default_artifact_dir() -> Path:
    return ARTIFACT_DIR


def model_bundle_path(artifact_dir: Path) -> Path:
    return artifact_dir / "saved_models" / "m5_monthly_radius50_bundle.joblib"


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def load_metrics(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


st.title("Gempa M>=5 Bulanan - Simple Testing UI")
st.caption("Mode titik (proxy cell radius 50km) dan mode region, horizon 1 bulan.")

with st.sidebar:
    st.header("Konfigurasi")
    artifact_dir_text = st.text_input(
        "Artifact directory", value=str(default_artifact_dir())
    )
    artifact_dir = Path(artifact_dir_text)
    model_path = model_bundle_path(artifact_dir)
    metrics_path = artifact_dir / "metrics.csv"
    metadata_path = artifact_dir / "metadata.json"

    st.write("Data path:", str(DATA_PATH))
    st.write("Model path:", str(model_path))

    if st.button("Train / Retrain Model", type="primary"):
        with st.spinner("Training model, mohon tunggu..."):
            summary = train_and_save(data_path=DATA_PATH, artifact_dir=artifact_dir)
        st.success("Training selesai")
        st.json(summary)
        st.cache_resource.clear()


if not model_path.exists():
    st.warning("Model belum tersedia. Klik 'Train / Retrain Model' di sidebar.")
    st.stop()


bundle = load_bundle(str(model_path))
metadata = bundle.get("metadata", {})

col_meta, col_metrics = st.columns(2)
with col_meta:
    st.subheader("Metadata")
    st.json(metadata)

with col_metrics:
    st.subheader("Metrics")
    metrics_df = load_metrics(metrics_path)
    if metrics_df.empty:
        st.info("metrics.csv belum ada")
    else:
        st.dataframe(metrics_df, use_container_width=True)


st.divider()
mode = st.radio("Pilih mode inferensi", ["Point", "Region"], horizontal=True)

if mode == "Point":
    st.subheader("Inferensi Mode Point")
    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.number_input("Latitude", value=-6.2, format="%.6f")
    with c2:
        lon = st.number_input("Longitude", value=106.8, format="%.6f")
    with c3:
        ref_month = st.text_input("Reference Month (opsional, YYYY-MM)", value="")

    if st.button("Prediksi Point", type="primary"):
        ref_arg = ref_month.strip() or None
        result = predict_next_month_point(
            bundle=bundle,
            lat=float(lat),
            lon=float(lon),
            reference_month=ref_arg,
        )
        st.success("Prediksi berhasil")
        st.json(result)

if mode == "Region":
    st.subheader("Inferensi Mode Region")
    region_map = bundle["region"]["region_map"]
    all_regions = sorted(region_map["entity"].unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        region_name = st.selectbox("Pilih region", options=all_regions, index=0)
    with c2:
        ref_month = st.text_input("Reference Month (opsional, YYYY-MM)", value="")

    if st.button("Prediksi Region", type="primary"):
        ref_arg = ref_month.strip() or None
        result = predict_next_month_region(
            bundle=bundle,
            region_name=region_name,
            reference_month=ref_arg,
        )
        st.success("Prediksi berhasil")
        st.json(result)


with st.expander("Raw metadata.json"):
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="json")
    else:
        st.info("metadata.json belum ada")
