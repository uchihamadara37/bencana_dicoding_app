import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spatial Click M3 Simulator", layout="wide")

DEFAULT_ARTIFACT_DIR = Path(
    r"D:\Projects\bencana_dicoding_app\timeseries_triwulan\artifacts\model_spatial_click_m3"
)


def resolve_artifact_dir() -> Path:
    if DEFAULT_ARTIFACT_DIR.exists():
        return DEFAULT_ARTIFACT_DIR

    local_candidate = (
        Path(__file__).resolve().parent / "artifacts" / "model_spatial_click_m3"
    )
    if local_candidate.exists():
        return local_candidate

    raise FileNotFoundError(
        "Artifact directory tidak ditemukan. Pastikan folder model_spatial_click_m3 sudah ada."
    )


ARTIFACT_DIR = resolve_artifact_dir()
MODEL_PATH = ARTIFACT_DIR / "saved_models" / "spatial_click_m3_twostage.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics_spatial.csv"
PRED_TEST_PATH = ARTIFACT_DIR / "predictions_test_spatial.csv"
TRAIN_DATASET_PATH = ARTIFACT_DIR / "train_dataset_spatial.csv"


@st.cache_resource
def load_model_bundle(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_metadata(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data
def load_train_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data
def build_map_candidates(train_df: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    # Keep only base columns in case duplicated columns from export are present.
    base = train_df[["cell_id", "lat_bin", "lon_bin", "count_m3"]].copy()
    base["is_nonzero"] = (base["count_m3"] > 0).astype(int)

    agg = (
        base.groupby(["cell_id", "lat_bin", "lon_bin"], as_index=False)
        .agg(
            total_events_train=("count_m3", "sum"),
            nonzero_months_train=("is_nonzero", "sum"),
            mean_count_train=("count_m3", "mean"),
            max_count_train=("count_m3", "max"),
        )
        .sort_values("total_events_train", ascending=False)
        .reset_index(drop=True)
    )

    agg["lat_center"] = (agg["lat_bin"] + 0.5) * grid_deg
    agg["lon_center"] = (agg["lon_bin"] + 0.5) * grid_deg
    return agg


@st.cache_data
def build_monthly_eval_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        pred_df.groupby("month", as_index=False)
        .agg(
            actual_total=("count_m3", "sum"),
            pred_expected_total=("pred_expected_count", "sum"),
        )
        .sort_values("month")
    )
    table["abs_error"] = (table["actual_total"] - table["pred_expected_total"]).abs()
    return table


def predict_total_next_month(bundle: dict, reference_month=None) -> dict:
    history = bundle["history_pivot"]

    if reference_month is None:
        ref_month = (
            pd.Timestamp(bundle["last_observed_month"]).to_period("M").to_timestamp()
        )
    else:
        ref_month = pd.Timestamp(reference_month).to_period("M").to_timestamp()

    target_month = (ref_month.to_period("M") + 1).to_timestamp()

    expected_counts = []
    active_cells = 0
    cell_lookup = bundle["cell_lookup"]

    for row in cell_lookup.itertuples(index=False):
        cell_id = row.cell_id
        lat_bin = int(row.lat_bin)
        lon_bin = int(row.lon_bin)

        if cell_id in history.columns:
            series = history[cell_id]
        else:
            series = pd.Series(0.0, index=history.index)

        x_row = build_single_feature(
            lat_bin=lat_bin,
            lon_bin=lon_bin,
            target_month=target_month,
            history_series=series,
            feature_cols=bundle["feature_cols"],
            lags=bundle["lags"],
            roll_windows=bundle["roll_windows"],
        )

        p_event = float(bundle["classifier"].predict_proba(x_row)[0, 1])
        pred_if_event = float(np.clip(bundle["regressor"].predict(x_row)[0], 0.0, None))
        expected_count = float(p_event * pred_if_event)

        expected_counts.append(expected_count)
        if p_event >= 0.5:
            active_cells += 1

    total_expected = float(np.sum(expected_counts))
    n_cells = int(len(expected_counts))
    mean_expected = float(total_expected / n_cells) if n_cells > 0 else 0.0

    return {
        "reference_month": str(ref_month.date()),
        "target_month": str(target_month.date()),
        "total_expected_count": total_expected,
        "active_cells_prob_ge_0_5": int(active_cells),
        "n_cells": n_cells,
        "mean_expected_per_cell": mean_expected,
    }


def latlon_to_bin(lat: float, lon: float, grid_deg: float):
    return int(np.floor(lat / grid_deg)), int(np.floor(lon / grid_deg))


def build_single_feature(
    lat_bin: int,
    lon_bin: int,
    target_month,
    history_series: pd.Series,
    feature_cols,
    lags,
    roll_windows,
):
    target_month = pd.Timestamp(target_month).to_period("M").to_timestamp()
    feat = {}

    for lag in lags:
        lag_month = (target_month.to_period("M") - lag).to_timestamp()
        feat[f"lag_{lag}"] = float(history_series.get(lag_month, 0.0))

    for w in roll_windows:
        vals = []
        for k in range(1, w + 1):
            m = (target_month.to_period("M") - k).to_timestamp()
            vals.append(float(history_series.get(m, 0.0)))
        feat[f"roll_mean_{w}"] = float(np.mean(vals))
        feat[f"roll_std_{w}"] = float(np.std(vals))

    month_num = target_month.month
    feat["month_sin"] = float(np.sin(2 * np.pi * month_num / 12.0))
    feat["month_cos"] = float(np.cos(2 * np.pi * month_num / 12.0))
    feat["lat_bin"] = int(lat_bin)
    feat["lon_bin"] = int(lon_bin)

    return pd.DataFrame([feat])[feature_cols]


def predict_click_one_month(
    lat: float, lon: float, bundle: dict, reference_month=None
) -> dict:
    grid_deg = bundle["grid_deg"]
    lat_bin, lon_bin = latlon_to_bin(lat, lon, grid_deg)
    cell_id = f"{lat_bin}_{lon_bin}"

    history = bundle["history_pivot"]

    if reference_month is None:
        ref_month = (
            pd.Timestamp(bundle["last_observed_month"]).to_period("M").to_timestamp()
        )
    else:
        ref_month = pd.Timestamp(reference_month).to_period("M").to_timestamp()

    target_month = (ref_month.to_period("M") + 1).to_timestamp()

    if cell_id in history.columns:
        series = history[cell_id]
        known_cell = True
    else:
        series = pd.Series(0.0, index=history.index)
        known_cell = False

    X_row = build_single_feature(
        lat_bin=lat_bin,
        lon_bin=lon_bin,
        target_month=target_month,
        history_series=series,
        feature_cols=bundle["feature_cols"],
        lags=bundle["lags"],
        roll_windows=bundle["roll_windows"],
    )

    p_event = float(bundle["classifier"].predict_proba(X_row)[0, 1])
    pred_if_event = float(np.clip(bundle["regressor"].predict(X_row)[0], 0.0, None))
    expected_count = float(p_event * pred_if_event)

    return {
        "input_lat": float(lat),
        "input_lon": float(lon),
        "lat_bin": int(lat_bin),
        "lon_bin": int(lon_bin),
        "cell_id": cell_id,
        "known_cell_in_training": known_cell,
        "reference_month": str(ref_month.date()),
        "target_month": str(target_month.date()),
        "prob_event_ge_1": p_event,
        "pred_count_if_event": pred_if_event,
        "expected_count": expected_count,
    }


def metadata_to_table(meta: dict) -> pd.DataFrame:
    rows = []
    for key, value in meta.items():
        if isinstance(value, (dict, list)):
            val = json.dumps(value, ensure_ascii=True)
        else:
            val = str(value)
        rows.append({"key": key, "value": val})
    return pd.DataFrame(rows)


# Load artifacts
bundle = load_model_bundle(MODEL_PATH)
meta = load_metadata(METADATA_PATH)
metrics_df = load_metrics(METRICS_PATH)
pred_test_df = load_predictions(PRED_TEST_PATH)
train_df = load_train_dataset(TRAIN_DATASET_PATH)
map_candidates = build_map_candidates(train_df, grid_deg=float(bundle["grid_deg"]))
monthly_eval_df = build_monthly_eval_table(pred_test_df)
meta_table = metadata_to_table(meta)

st.title("Simulasi Streamlit - Spatial Click Gempa M>=3")
st.caption("Data sumber: artifact model_spatial_click_m3 dari notebook")

with st.sidebar:
    st.header("Input Simulasi")
    input_mode = st.radio(
        "Sumber input", ["Pilih dari daftar peta", "Input manual lat/lon"]
    )

    max_points = int(len(map_candidates))
    if max_points < 20:
        min_points = 1
        step_points = 1
    else:
        min_points = 20
        step_points = 20

    default_points = min(200, max_points)
    if default_points < min_points:
        default_points = max_points

    top_n = st.slider(
        "Jumlah titik yang ditampilkan di peta",
        min_value=min_points,
        max_value=max_points,
        value=default_points,
        step=step_points,
    )
    map_view_df = map_candidates.head(top_n).copy()

    if input_mode == "Pilih dari daftar peta":
        selected_cell_id = st.selectbox(
            "Pilih cell", options=map_view_df["cell_id"].tolist()
        )
        selected_row = map_candidates.loc[
            map_candidates["cell_id"] == selected_cell_id
        ].iloc[0]
        input_lat = float(selected_row["lat_center"])
        input_lon = float(selected_row["lon_center"])
        st.write("Lat center:", round(input_lat, 4))
        st.write("Lon center:", round(input_lon, 4))
    else:
        input_lat = st.number_input("Latitude", value=-6.2, format="%.6f")
        input_lon = st.number_input("Longitude", value=106.8, format="%.6f")

    ref_month = st.date_input(
        "Reference month (akhir data historis)",
        value=pd.Timestamp(bundle["last_observed_month"]).date(),
    )

    run_btn = st.button("Jalankan Prediksi")

st.subheader("Daftar Input dari Peta (berdasarkan aktivitas train)")
map_table = map_view_df[
    [
        "cell_id",
        "lat_center",
        "lon_center",
        "total_events_train",
        "nonzero_months_train",
        "max_count_train",
    ]
].copy()
st.dataframe(map_table, use_container_width=True)

st.map(
    map_view_df.rename(columns={"lat_center": "lat", "lon_center": "lon"})[
        ["lat", "lon"]
    ]
)

if run_btn:
    pred_result = predict_click_one_month(
        lat=float(input_lat),
        lon=float(input_lon),
        bundle=bundle,
        reference_month=pd.Timestamp(ref_month),
    )

    total_result = predict_total_next_month(
        bundle=bundle,
        reference_month=pd.Timestamp(ref_month),
    )

    st.markdown("## Hasil Utama Prediksi")
    st.markdown(
        f"""
        <div style="border:2px solid #0f766e;border-radius:14px;padding:18px;background:#ecfeff;">
            <div style="font-size:16px;font-weight:600;color:#115e59;">Estimasi total gempa M&gt;=3 untuk bulan {total_result['target_month']}</div>
            <div style="font-size:56px;font-weight:800;line-height:1.1;color:#0f766e;">{total_result['total_expected_count']:.1f}</div>
            <div style="font-size:14px;color:#334155;">Agregat seluruh cell pada grid spasial</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)
    col_kpi_1.metric("Bulan Referensi", total_result["reference_month"])
    col_kpi_2.metric("Cell aktif (p>=0.5)", total_result["active_cells_prob_ge_0_5"])
    col_kpi_3.metric(
        "Rata-rata expected/cell", f"{total_result['mean_expected_per_cell']:.3f}"
    )

    st.subheader("Metadata Hasil Prediksi (1 Bulan ke Depan)")
    st.table(pd.DataFrame([pred_result]))

    selected_cell = pred_result["cell_id"]
    cell_hist = pred_test_df.loc[
        pred_test_df["cell_id"] == selected_cell,
        [
            "month",
            "count_m3",
            "pred_expected_count",
            "p_event",
            "pred_count_if_event",
        ],
    ].copy()
    cell_hist = cell_hist.sort_values("month")

    st.subheader("Tabel Actual vs Prediksi (Cell Terpilih pada Periode Test)")
    if cell_hist.empty:
        st.info(
            "Cell belum muncul pada periode test, jadi tabel actual-vs-pred per-cell tidak tersedia."
        )
    else:
        st.dataframe(cell_hist, use_container_width=True)

st.subheader("Tabel Actual vs Prediksi Agregat Bulanan (Periode Test)")
st.dataframe(monthly_eval_df, use_container_width=True)

st.subheader("Tabel Eval Metrik")
st.table(metrics_df)

st.subheader("Tabel Metadata Model")
st.dataframe(meta_table, use_container_width=True)
