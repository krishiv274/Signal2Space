import io

import numpy as np
import pandas as pd
import streamlit as st

import charts
import core
import loaders

st.title("Live prediction")
st.markdown("The trained Phase 2 networks score the scan below **live**. Hide access points or add noise to see "
            "how the answer changes when the fingerprint gets worse.")

scaler, heldout, positions = loaders.scaler_and_raw()
saved = loaders.saved_predictions()

source = st.segmented_control("Scan source", ["Held-out scan", "Upload your own"], default="Held-out scan",
                              key="source")

# ---------------------------------------------------------------- choose a scan
truth = None
row_id = None
if source == "Upload your own":
    st.markdown("Upload a CSV with columns `WAP001` … `WAP520` holding RSSI in dBm (−110 to 0), with **100** for an "
                "access point that was not detected. Optional columns `BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE` "
                "are used to score the prediction.")
    example = heldout.sample(5, random_state=7)[core.AP + ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]]
    st.download_button("Download an example file (5 held-out scans)", example.to_csv(index=False),
                       file_name="signal2space_example_scans.csv", mime="text/csv")
    up = st.file_uploader("Scan file", type=["csv"])
    if up is None:
        st.info("Upload a file to score it, or switch back to a held-out scan.")
        st.stop()
    frame = pd.read_csv(io.BytesIO(up.getvalue()))
    raw_all, problems = core.parse_upload(frame)
    if problems:
        for p in problems:
            st.error(p)
        st.stop()
    idx = st.number_input(f"Row to inspect (file has {len(frame)} rows)", 1, len(frame), 1) - 1
    raw0 = raw_all[idx]
    if all(c in frame.columns for c in ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]):
        r = frame.iloc[idx]
        truth = dict(building=int(r.BUILDINGID), floor=int(r.FLOOR), x=float(r.LONGITUDE), y=float(r.LATITUDE))
    with st.expander(f"Predictions for all {len(frame)} rows"):
        p_all = loaders.score(raw_all)
        st.dataframe(pd.DataFrame({"row": np.arange(1, len(frame) + 1), "building": p_all["building"], "floor": p_all["floor"],
                                   "longitude (multitask)": p_all["coords_multitask"][:, 0].round(1),
                                   "latitude (multitask)": p_all["coords_multitask"][:, 1].round(1),
                                   "longitude (coordinate net)": p_all["coords_single"][:, 0].round(1),
                                   "latitude (coordinate net)": p_all["coords_single"][:, 1].round(1)}),
                     hide_index=True, width="stretch")
else:
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
    b_sel = f1.selectbox("Building", ["Any", 0, 1, 2])
    fl_opts = ["Any"] + sorted(heldout.FLOOR.unique().tolist() if b_sel == "Any" else heldout[heldout.BUILDINGID == b_sel].FLOOR.unique().tolist())
    fl_sel = f2.selectbox("Floor", fl_opts)
    ph_sel = f3.selectbox("Phone (PHONEID)", ["Any"] + sorted(heldout.PHONEID.unique().tolist()))
    pool = heldout
    if b_sel != "Any":
        pool = pool[pool.BUILDINGID == b_sel]
    if fl_sel != "Any":
        pool = pool[pool.FLOOR == fl_sel]
    if ph_sel != "Any":
        pool = pool[pool.PHONEID == ph_sel]
    if pool.empty:
        st.warning("No held-out scan matches these filters.")
        st.stop()
    ids = pool.ROW_ID.tolist()
    if "row_id" not in st.session_state:
        st.session_state.row_id = loaders.default_scan_id()
    if st.session_state.row_id not in ids:
        st.session_state.row_id = ids[0]
    if f4.button("Random scan", icon=":material/casino:", width="stretch"):
        st.session_state.row_id = int(np.random.default_rng().choice(ids))
    row_id = st.selectbox(f"Held-out scan ({len(ids)} match)", ids, index=ids.index(st.session_state.row_id),
                          format_func=lambda i: f"scan {i} · building {heldout.at[i, 'BUILDINGID']}, floor {heldout.at[i, 'FLOOR']}, phone {heldout.at[i, 'PHONEID']}")
    st.session_state.row_id = row_id
    r = heldout.loc[row_id]
    raw0 = r[core.AP].to_numpy(dtype=np.float64)
    truth = dict(building=int(r.BUILDINGID), floor=int(r.FLOOR), x=float(r.LONGITUDE), y=float(r.LATITUDE))

# ---------------------------------------------------------------- stress test
n_det = int((raw0 != core.MISSING).sum())
with st.expander("Stress-test the scan", icon=":material/tune:", expanded=False):
    s1, s2, s3 = st.columns(3)
    hide = s1.slider("Hide the strongest access points", 0, min(10, n_det), 0,
                     help="Simulates the nearest routers being switched off.")
    drop = s2.slider("Randomly drop detected access points (%)", 0, 80, 0, step=5)
    noise = s3.slider("Add signal noise (± dB, standard deviation)", 0.0, 10.0, 0.0, step=0.5)
raw, removed = core.perturb_scan(raw0, hide, drop / 100, noise, seed=int(row_id or 0))
modified = bool(hide or drop or noise)

pred = loaders.score(raw)
pb, pf = pred["building_proba"][0], pred["floor_proba"][0]
b_hat, f_hat = int(pred["building"][0]), int(pred["floor"][0])
xm, ym = pred["coords_multitask"][0]
xs, ys = pred["coords_single"][0]

# ---------------------------------------------------------------- headline
m1, m2, m3, m4 = st.columns(4)
if truth:
    m1.metric("Building", f"{b_hat}", "correct" if b_hat == truth["building"] else f"true: {truth['building']}",
              delta_color="normal" if b_hat == truth["building"] else "inverse", delta_arrow="off")
    m2.metric("Floor", f"{f_hat}", "correct" if f_hat == truth["floor"] else f"true: {truth['floor']}",
              delta_color="normal" if f_hat == truth["floor"] else "inverse", delta_arrow="off")
    m3.metric("Position error · multitask", f"{np.hypot(xm - truth['x'], ym - truth['y']):.1f}", "dataset units", delta_color="off",
              delta_arrow="off")
    m4.metric("Position error · coordinate net", f"{np.hypot(xs - truth['x'], ys - truth['y']):.1f}", "dataset units",
              delta_color="off", delta_arrow="off")
else:
    m1.metric("Building", b_hat, f"{pb.max():.0%} confident", delta_color="off", delta_arrow="off")
    m2.metric("Floor", f_hat, f"{pf.max():.0%} confident", delta_color="off", delta_arrow="off")
    m3.metric("Position · multitask", f"{xm:.0f}, {ym:.0f}")
    m4.metric("Position · coordinate net", f"{xs:.0f}, {ys:.0f}")
if modified:
    st.caption(f"Scan modified: {int(removed.sum())} of {n_det} detected access points removed"
               + (f", ±{noise:g} dB noise" if noise else "") + ".")
if n_det == 0 or int((raw != core.MISSING).sum()) == 0:
    st.warning("This scan has no detected access points left, so the prediction carries no location information.")

# ---------------------------------------------------------------- map + probabilities
left, right = st.columns([1.7, 1])
points = {}
if truth:
    points["True position"] = (truth["x"], truth["y"])
points["DL multitask"] = (xm, ym)
points["DL coordinate network"] = (xs, ys)
aml_row = None
if row_id is not None and not modified:
    a = saved["AML Random Forest"].set_index("ROW_ID").loc[row_id]
    points["AML Random Forest"] = (a.LONGITUDE_PRED, a.LATITUDE_PRED)
    aml_row = a
with left:
    zoom = st.toggle("Zoom to the predicted building", value=True)
    st.plotly_chart(charts.radio_map(positions, points, building=b_hat if zoom else None), width="stretch")
    if aml_row is not None:
        st.caption(f"Random Forest (from saved Phase 1 predictions): building {int(aml_row.BUILDINGID_PRED)}, "
                   f"floor {int(aml_row.FLOOR_PRED)}, position error {aml_row.EUCLIDEAN_ERROR:.1f}. "
                   "Phase 1 models were not saved as files, so the Random Forest is shown only for unmodified held-out scans.")
    elif row_id is not None:
        st.caption("The Random Forest is hidden for modified scans: Phase 1 models were not saved as files, so only "
                   "their saved predictions for the original scans are available.")
with right:
    st.plotly_chart(charts.probabilities(pred["building_classes"], pb, truth["building"] if truth else None,
                                         "Building probability"), width="stretch")
    st.plotly_chart(charts.probabilities(pred["floor_classes"], pf, truth["floor"] if truth else None,
                                         "Floor probability"), width="stretch")

st.plotly_chart(charts.fingerprint(raw0, raw), width="stretch")
st.caption("Blue bars are access points the model used; grey bars were removed by the stress test. "
           "Every other access point (of 520) was not detected.")
