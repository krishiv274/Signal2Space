import streamlit as st

import loaders

st.title("Signal2Space")
st.markdown("#### Which building, which floor, and where? Predicted from one Wi-Fi scan.")
st.markdown(
    "A phone indoors hears some of 520 Wi-Fi access points, each at a different strength. That list of signal "
    "strengths is a **fingerprint**. Signal2Space learns to map a fingerprint to the device's building, floor and "
    "position, using the UJIIndoorLoc dataset (19,937 training scans and 1,111 held-out scans in three buildings)."
)

comp = loaders.table("dl_05_multitask_aml_vs_dl_comparison.csv")
held = comp[comp.split == "heldout_validation"].set_index("model")
aml, mt, st_ = held.loc["AML Random Forest"], held.loc["DL multitask"], held.loc["DL single-task (separate nets)"]

st.subheader("Held-out results")
st.caption("All models are scored on the same 1,111 scans, collected months after the training data, mostly with phones "
           "never seen in training. Coordinate errors are in the dataset's recorded units.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Building accuracy · DL multitask", f"{mt.building_accuracy:.2%}",
          f"{(mt.building_accuracy - aml.building_accuracy) * 100:+.2f} pp vs Random Forest")
c2.metric("Floor accuracy · DL multitask", f"{mt.floor_accuracy:.2%}",
          f"{(mt.floor_accuracy - aml.floor_accuracy) * 100:+.2f} pp vs Random Forest")
c3.metric("Building and floor both right", f"{mt.building_and_floor_accuracy:.2%}",
          f"{(mt.building_and_floor_accuracy - aml.building_and_floor_accuracy) * 100:+.2f} pp vs Random Forest")
c4.metric("Lowest mean position error", f"{st_.mean_euclidean_error:.2f}",
          f"{st_.mean_euclidean_error - aml.mean_euclidean_error:+.2f} vs Random Forest", delta_color="inverse",
          help="DL coordinate network (single task). Dataset coordinate units.")

st.subheader("Two phases, one fair comparison")
a, b = st.columns(2)
with a:
    with st.container(border=True):
        st.markdown("**Phase 1 · Machine learning**")
        st.markdown("Three independent Random Forests, one each for building, floor and coordinates, trained on 1,046 "
                    "engineered features: scaled signal strengths, detected/not-detected flags and six signal summaries.")
        st.markdown(f"Floor **{aml.floor_accuracy:.2%}** · mean position error **{aml.mean_euclidean_error:.2f}**")
with b:
    with st.container(border=True):
        st.markdown("**Phase 2 · Deep learning**")
        st.markdown("One neural network with a shared trunk and three heads, trained on the same features, split and "
                    "metrics, plus one-head networks to test what sharing does.")
        st.markdown(f"Floor **{mt.floor_accuracy:.2%}** (significant, p = 0.014) · mean position error "
                    f"**{mt.mean_euclidean_error:.2f}** (no real difference)")

st.subheader("Explore")
e1, e2, e3 = st.columns(3)
with e1:
    st.page_link("views/live_prediction.py", label="Live prediction", icon=":material/wifi:")
    st.caption("Pick a real held-out scan or upload your own, hide access points, and watch the network's answer change.")
with e2:
    st.page_link("views/results_explorer.py", label="Results explorer", icon=":material/insights:")
    st.caption("Confusion matrices, error maps and error distributions for every model.")
with e3:
    st.page_link("views/how_it_works.py", label="How it works", icon=":material/schema:")
    st.caption("Features, architecture, training curves and model selection.")

st.divider()
st.caption("Team: Krishiv (krishiv274) · Gargi · Bhawana · Agrima · Abhiman (Koray-afk). "
           "Data: UJIIndoorLoc, UCI Machine Learning Repository, dataset 310.")
