import numpy as np
import pandas as pd
import streamlit as st

import charts
import core
import loaders

st.title("Results explorer")
st.markdown("Every model is scored on the same 1,111 held-out scans. Coordinate errors are in the dataset's recorded units.")

saved = loaders.saved_predictions()
comp = loaders.table("dl_05_multitask_aml_vs_dl_comparison.csv")
split = st.segmented_control("Partition", ["Held-out", "Internal validation"], default="Held-out", key="split")
c = comp[comp.split == ("heldout_validation" if split != "Internal validation" else "internal_validation")]

tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Floor errors", "Position errors", "Robustness"])

with tab1:
    a, b = st.columns(2)
    a.plotly_chart(charts.metric_bars(c, [("building_accuracy", "Building"), ("floor_accuracy", "Floor"),
                                          ("building_and_floor_accuracy", "Building + floor")],
                                      "Accuracy (higher is better; axis starts at 80%)", pct=True), width="stretch")
    b.plotly_chart(charts.metric_bars(c, [("mae_longitude", "MAE longitude"), ("mae_latitude", "MAE latitude"),
                                          ("mean_euclidean_error", "Mean error"), ("median_euclidean_error", "Median error")],
                                      "Position error (lower is better)", pct=False), width="stretch")
    show = c[["model", "building_accuracy", "floor_accuracy", "floor_macro_f1", "building_and_floor_accuracy",
              "mean_euclidean_error", "median_euclidean_error"]].rename(columns={
        "model": "Model", "building_accuracy": "Building acc.", "floor_accuracy": "Floor acc.", "floor_macro_f1": "Floor macro F1",
        "building_and_floor_accuracy": "Building + floor", "mean_euclidean_error": "Mean error", "median_euclidean_error": "Median error"})
    show = show.copy()
    for k in ["Building acc.", "Floor acc.", "Floor macro F1", "Building + floor"]:
        show[k] = show[k].map(lambda v: "—" if pd.isna(v) else f"{v:.2%}")
    for k in ["Mean error", "Median error"]:
        show[k] = show[k].map(lambda v: f"{v:.2f}")
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("DL single-task combines three separate one-head networks, so it has no joint building + floor value. "
               "Paired tests on held-out scans: the multitask floor gain over the Random Forest is significant "
               "(37 vs 18 discordant scans, p = 0.014); the mean position difference is not (+0.30, 95% CI −0.42 to +1.02).")

with tab2:
    if split == "Internal validation":
        st.info("Confusion matrices and error maps are shown for the held-out scans.")
    m = st.radio("Model", list(saved), horizontal=True, key="cm_model")
    d = saved[m]
    l, r = st.columns([1.2, 1])
    scale = {"AML Random Forest": "Oranges", "DL multitask": "Blues", "DL single-task": "Greens"}[m]
    l.plotly_chart(charts.confusion(d.FLOOR_TRUE, d.FLOOR_PRED, [0, 1, 2, 3, 4], f"Held-out floor confusion · {m}", scale),
                   width="stretch")
    rec = (pd.DataFrame({"true": d.FLOOR_TRUE, "ok": d.FLOOR_TRUE == d.FLOOR_PRED}).groupby("true").ok.agg(["size", "mean"])
           .rename(columns={"size": "Scans", "mean": "Recall"}))
    rec.index.name = "True floor"
    r.markdown("**Recall by floor**")
    r.dataframe(rec.style.format({"Recall": "{:.2%}"}), width="stretch")
    off = d[d.FLOOR_TRUE != d.FLOOR_PRED]
    r.markdown(f"**{len(off)}** floor errors; **{int(((off.FLOOR_TRUE - off.FLOOR_PRED).abs() == 1).sum())}** "
               "of them are one floor away.")
    hier = pd.Series([core.hierarchy_label(bb, ff) for bb, ff in zip(d.BUILDINGID_TRUE == d.BUILDINGID_PRED,
                                                                     d.FLOOR_TRUE == d.FLOOR_PRED)])
    h = d.assign(outcome=hier.values).groupby("outcome").agg(Scans=("ROW_ID", "size"), **{"Mean position error": ("EUCLIDEAN_ERROR", "mean")})
    r.markdown("**Building/floor outcome and position error**")
    r.dataframe(h.style.format({"Mean position error": "{:.2f}"}), width="stretch")

with tab3:
    m2 = st.radio("Model", list(saved), horizontal=True, key="map_model")
    bsel = st.selectbox("Building", ["All", 0, 1, 2], key="map_b")
    d2 = saved[m2] if bsel == "All" else saved[m2][saved[m2].BUILDINGID_TRUE == bsel]
    l, r = st.columns([1.4, 1])
    l.plotly_chart(charts.error_map(d2, f"Position error at each true location · {m2}"), width="stretch")
    r.plotly_chart(charts.error_cdf({k: (v if bsel == "All" else v[v.BUILDINGID_TRUE == bsel]).EUCLIDEAN_ERROR.to_numpy()
                                     for k, v in saved.items()}, "Share of scans within a given error"), width="stretch")
    e = d2.EUCLIDEAN_ERROR
    r.markdown(f"**{m2}**: mean {e.mean():.2f}, median {e.median():.2f}, 95th percentile {np.percentile(e, 95):.2f}, "
               f"worst {e.max():.1f}; {(e < 5).mean():.0%} of scans within 5 units.")

with tab4:
    seeds = loaders.table("dl_05_multitask_seed_variability.csv")
    st.markdown("**Training seeds.** The frozen multitask configuration retrained with three seeds, without re-selection.")
    st.dataframe(seeds[["seed", "best_epoch", "building_accuracy", "floor_accuracy", "building_and_floor_accuracy",
                        "mean_euclidean_error"]].rename(columns={"seed": "Seed", "best_epoch": "Best epoch",
                                                                 "building_accuracy": "Building", "floor_accuracy": "Floor",
                                                                 "building_and_floor_accuracy": "Building + floor",
                                                                 "mean_euclidean_error": "Mean error"})
                 .style.format({"Building": "{:.2%}", "Floor": "{:.2%}", "Building + floor": "{:.2%}", "Mean error": "{:.2f}"}),
                 hide_index=True, width="stretch")
    st.markdown("**Phones.** Held-out results by `PHONEID` for each model. Only phones 13 and 14 also appear in training; "
                "groups are small and tied to particular times and places, so differences are descriptive.")
    rows = []
    for name, d in saved.items():
        g = d.assign(ok=d.FLOOR_TRUE == d.FLOOR_PRED).groupby("PHONEID").agg(Scans=("ROW_ID", "size"), Floor=("ok", "mean"),
                                                                             Error=("EUCLIDEAN_ERROR", "mean"))
        g["Model"] = name
        rows.append(g.reset_index())
    ph = pd.concat(rows)
    pv = ph.pivot_table(index=["PHONEID", "Scans"], columns="Model", values="Floor").reset_index().sort_values("Scans", ascending=False)
    st.dataframe(pv.style.format({k: "{:.1%}" for k in saved}), hide_index=True, width="stretch")
    st.caption("Values are held-out floor accuracy per phone.")
