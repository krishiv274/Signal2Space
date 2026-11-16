"""Every page of the app runs without an exception (Streamlit AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
PAGES = ["views/overview.py", "views/live_prediction.py", "views/results_explorer.py", "views/how_it_works.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_live_prediction_interactions():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page("views/live_prediction.py").run()
    assert not at.exception
    at.selectbox[0].select(2).run()          # building 2
    at.button[0].click().run()               # random scan
    at.slider[0].set_value(3).run()          # hide the 3 strongest APs
    at.slider[2].set_value(4.0).run()        # add noise
    assert not at.exception, [e.value for e in at.exception]
    assert any("Scan modified" in c.value for c in at.caption)
