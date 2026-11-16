"""Cached loaders shared by every page (Streamlit caching wraps the plain functions in core.py)."""

import numpy as np
import streamlit as st

import core


@st.cache_resource(show_spinner="Rebuilding the Phase 1 feature transform from the raw data…")
def scaler_and_raw():
    train, heldout = core.load_raw()
    scaler = core.fit_rssi_scaler(core.internal_train_rows(train))
    positions = train[["BUILDINGID", "LONGITUDE", "LATITUDE"]].round(3).drop_duplicates()
    return scaler, heldout, positions


@st.cache_resource(show_spinner="Loading the trained networks…")
def models():
    return core.load_models()


@st.cache_data(show_spinner=False)
def saved_predictions():
    return core.load_saved_predictions()


@st.cache_data(show_spinner=False)
def table(name: str):
    return core.load_table(name)


@st.cache_data(show_spinner=False)
def default_scan_id() -> int:
    """A representative held-out scan: multitask building and floor correct, 15-40 detected access points,
    position error closest to the median of such scans."""
    _, heldout, _ = scaler_and_raw()
    mt = saved_predictions()["DL multitask"].set_index("ROW_ID")
    n_det = (heldout[core.AP] != core.MISSING).sum(axis=1)
    ok = mt[(mt.BUILDINGID_TRUE == mt.BUILDINGID_PRED) & (mt.FLOOR_TRUE == mt.FLOOR_PRED) & n_det.between(15, 40).to_numpy()]
    return int((ok.EUCLIDEAN_ERROR - ok.EUCLIDEAN_ERROR.median()).abs().idxmin())


def score(raw: np.ndarray) -> dict:
    scaler, _, _ = scaler_and_raw()
    return core.predict(models(), core.features_from_raw(raw, scaler))
