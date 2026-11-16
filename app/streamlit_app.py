"""Signal2Space interactive app.  Run from the repository root:

    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Signal2Space", page_icon="📡", layout="wide")

pages = st.navigation([
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/live_prediction.py", title="Live prediction", icon=":material/wifi:"),
    st.Page("views/results_explorer.py", title="Results explorer", icon=":material/insights:"),
    st.Page("views/how_it_works.py", title="How it works", icon=":material/schema:"),
])
with st.sidebar:
    st.caption("Signal2Space · Project 9: indoor location prediction from Wi-Fi signals. "
               "Coordinates are in the dataset's recorded units.")
    st.caption("[GitHub repository](https://github.com/krishiv274/Signal2Space)")
pages.run()
