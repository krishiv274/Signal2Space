import streamlit as st

import charts
import loaders

st.title("How it works")

st.subheader("1 · From a Wi-Fi scan to 1,046 features")
st.markdown(
    "Each scan has 520 readings. A reading between −104 and 0 dBm is a signal strength; the value **100** means the "
    "access point was not detected (about 97% of readings). Phase 1 turned every scan into 1,046 features, using "
    "statistics from the internal training split only:"
)
a, b, c = st.columns(3)
with a:
    with st.container(border=True):
        st.metric("Scaled signal strength", "520")
        st.caption("Each access point standardised with its own training mean and spread; undetected readings filled "
                   "just below the weakest training reading.")
with b:
    with st.container(border=True):
        st.metric("Detected / not detected", "520")
        st.caption("A 0/1 flag per access point, so absence is explicit information.")
with c:
    with st.container(border=True):
        st.metric("Signal summaries", "6")
        st.caption("How many were heard, strongest, mean and spread of strength, how many are strong, and the "
                   "strongest access point.")

st.subheader("2 · Shared-trunk multitask network")
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#8493A1", fontcolor="#243A50"];
  edge [color="#64877F"];
  x  [label="1,046 features", fillcolor="#E8EFF4"];
  t1 [label="Linear 512\\nBatchNorm · ReLU · Dropout", fillcolor="#C7DBD1"];
  t2 [label="Linear 256\\nBatchNorm · ReLU · Dropout", fillcolor="#C7DBD1"];
  hb [label="Building head\\n3 probabilities", fillcolor="#F1E9DB"];
  hf [label="Floor head\\n5 probabilities", fillcolor="#F1E9DB"];
  hc [label="Coordinate head\\n2 values", fillcolor="#F1E9DB"];
  x -> t1 -> t2; t2 -> hb; t2 -> hf; t2 -> hc;
}
""")
st.markdown(
    "All three heads read the same 256-number summary of the scan (the **shared trunk**). Training minimises "
    "`CE(building) + CE(floor) + 2 × SmoothL1(coordinates)`, with AdamW and early stopping on internal validation. "
    "The same network family trained with one head at a time gives the **single-task** comparison models."
)

st.subheader("3 · Training and model selection")
hist = loaders.table("dl_05_multitask_training_history.csv")
sel = loaders.table("dl_05_multitask_selection.csv")
chosen = sel.loc[sel.selected, "config"].iloc[0]
st.plotly_chart(charts.training_history(hist[hist.config == chosen]), width="stretch")
st.markdown("Seven candidates were compared on internal validation only, with the score "
            "`mean(building error rate, floor error rate, mean position error ÷ 125.9)`. The chosen configuration was "
            "saved before the held-out scans were ever read.")
st.dataframe(sel[["config", "hidden_sizes", "dropout", "w_floor", "w_coord", "coord_beta", "best_epoch", "val_selection_score",
                  "val_floor_acc", "val_mean_euclidean", "selected"]]
             .rename(columns={"config": "Candidate", "hidden_sizes": "Trunk", "dropout": "Dropout", "w_floor": "Floor weight",
                              "w_coord": "Coordinate weight", "coord_beta": "Smooth-L1 β", "best_epoch": "Best epoch",
                              "val_selection_score": "Score", "val_floor_acc": "Floor acc.", "val_mean_euclidean": "Mean error",
                              "selected": "Selected"})
             .style.format({"Score": "{:.4f}", "Floor acc.": "{:.2%}", "Mean error": "{:.2f}", "Dropout": "{:g}",
                            "Floor weight": "{:g}", "Coordinate weight": "{:g}", "Smooth-L1 β": "{:g}"}),
             hide_index=True, width="stretch")

st.subheader("4 · What it means")
st.markdown(
    "- **Building** is near-perfect for every model, so it does not separate them.\n"
    "- **Floor** benefits from sharing: the multitask network is the best floor classifier, and the gain is significant "
    "and stable across training seeds.\n"
    "- **Position** suffers: the shared trunk is pulled toward separating floors, so a network trained only for "
    "coordinates places scans most accurately.\n"
    "- Results describe this dataset's split (a later campaign in the same buildings), and positions are in the "
    "dataset's own units because the coordinate system is undocumented."
)
