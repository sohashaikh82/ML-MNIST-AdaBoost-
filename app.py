"""MNIST AI LAB — entry point.

Run with:  streamlit run app.py

This file only sets up the page, styling and navigation. Each page lives in
``views/`` and all machine-learning code lives in ``src/``.
"""

import streamlit as st

from ui.components import inject_css, render_sidebar
from ui.nav import PAGES

st.set_page_config(
    page_title="MNIST AI LAB · AdaBoost",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [st.Page(path, title=title, icon=icon, default=(i == 0)) for i, (path, title, icon, _s) in enumerate(PAGES)]
navigation = st.navigation(pages, position="hidden")  # we render our own sidebar with st.page_link

inject_css()
render_sidebar(active_url_path=navigation.url_path)
navigation.run()
