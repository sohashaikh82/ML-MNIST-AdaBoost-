"""Reusable UI pieces: CSS, sidebar, header, metric cards, empty states."""

from __future__ import annotations

import base64
import html
from datetime import date
from functools import lru_cache

import streamlit as st

from src.config import ASSETS_DIR
from ui.nav import PAGES, PATH_BY_TITLE
from ui.state import first_time, get_data, get_result


@lru_cache(maxsize=1)
def _css() -> str:
    return (ASSETS_DIR / "styles.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _logo_b64() -> str:
    return base64.b64encode((ASSETS_DIR / "logo.png").read_bytes()).decode()


def inject_css() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def render_sidebar(active_url_path: str = "") -> None:
    """Dark sidebar: brand, real page links, model status and info card.

    Streamlit gives page links no 'active' attribute, so we highlight the current
    page by its URL path (the default page's link has an empty href).
    """
    safe = active_url_path.replace('"', "")
    st.markdown(
        f"""<style>[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{safe}"] {{
        background: var(--lime) !important; }}
        [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{safe}"] * {{
        color: var(--charcoal) !important; font-weight: 700; }}</style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(
            f"""<div class="brand"><img src="data:image/png;base64,{_logo_b64()}" alt="logo"/>
            <div><div class="brand-title">MNIST AI LAB</div><div class="brand-sub">AdaBoost digit recognition</div></div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        for path, title, icon, section in PAGES:
            if section == "main":
                st.page_link(path, label=title, icon=icon)
        st.markdown('<div class="nav-label">Learn &amp; submit</div>', unsafe_allow_html=True)
        for path, title, icon, section in PAGES:
            if section == "learn":
                st.page_link(path, label=title, icon=icon)

        result = get_result()
        if result is not None:
            status = (f'<span class="status-dot on"></span><span><b>Model ready</b><br/>'
                      f'{result.n_estimators_fitted} learners · {result.test_accuracy:.1%} test acc.</span>')
        else:
            status = '<span class="status-dot off"></span><span><b>No model yet</b><br/>Train one on Model Training</span>'
        st.markdown(f'<div class="status-pill">{status}</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="promo"><h4>Explore Machine Learning</h4>
            <p>Draw a digit, train AdaBoost on real MNIST images and see how many weak decision trees
            vote together to recognise handwriting.</p></div>""",
            unsafe_allow_html=True,
        )


def page_header(title: str, subtitle: str, key: str) -> None:
    """Title row with project icon, live chips (date, dataset, model) and quick navigation."""
    data = get_data()
    result = get_result()
    left, right = st.columns([3, 1.1], vertical_alignment="center")
    with left:
        ds = f"MNIST · {data.n_train:,} train / {data.n_test:,} test" if data else "Dataset unavailable"
        model_chip = ('<span class="chip ok">● Model trained</span>' if result
                      else '<span class="chip warn">● No model</span>')
        st.markdown(
            f"""<div class="topbar"><div class="hello"><div class="avatar">AI</div>
            <div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div></div></div>
            <div class="chips" style="margin:.6rem 0 1rem">
              <span class="chip dark">📅 {date.today().strftime('%a, %d %b %Y')}</span>
              <span class="chip">🗂️ {ds}</span>{model_chip}</div>""",
            unsafe_allow_html=True,
        )
    with right:
        widget_key = f"quicknav_{key}"

        def _on_pick():
            st.session_state["_nav_target"] = st.session_state.get(widget_key)
            st.session_state[widget_key] = None  # reset so returning to this page doesn't re-jump

        st.selectbox("Quick navigation", list(PATH_BY_TITLE), index=None, placeholder="🔎 Jump to a page…",
                     key=widget_key, label_visibility="collapsed", on_change=_on_pick)
        target = st.session_state.pop("_nav_target", None)
        if target:
            st.switch_page(PATH_BY_TITLE[target])


def metric_card(label: str, value: str, sub: str = "", icon: str = "", variant: str = "",
                badge: str | None = None, badge_kind: str = "up", counter: int | None = None,
                anim_tag: str | None = None, delay: int = 0) -> None:
    """A dashboard metric card. ``counter`` animates an integer from 0 once per session."""
    animate = anim_tag is not None and first_time(anim_tag)
    cls = f"metric-card {variant} {'anim d' + str(delay) if animate else ''}"
    if counter is not None and animate:
        value_html = (f'<span class="counter anim-count" style="--target:{int(counter)}">'
                      f'<span class="static">{html.escape(value)}</span></span>')
    else:
        value_html = html.escape(value)
    badge_html = f'<span class="badge {badge_kind}">{html.escape(badge)}</span> ' if badge else ""
    st.markdown(
        f"""<div class="{cls}"><div class="m-top"><span class="m-label">{html.escape(label)}</span>
        <span class="m-icon">{icon}</span></div><div class="m-value">{value_html}</div>
        <div class="m-sub">{badge_html}{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def card_title(title: str, sub: str = "") -> None:
    st.markdown(f'<p class="card-title">{html.escape(title)}</p>'
                + (f'<p class="card-sub">{sub}</p>' if sub else ""), unsafe_allow_html=True)


def empty_state(title: str, body: str, icon: str = "🧪", link: str | None = None, link_label: str = "") -> None:
    st.markdown(f'<div class="empty"><div class="e-icon">{icon}</div><h4>{html.escape(title)}</h4>'
                f'<p>{body}</p></div>', unsafe_allow_html=True)
    if link:
        st.page_link(link, label=link_label, icon=":material/arrow_forward:")


def callout(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def require_model():
    """Return the active model or show an empty state and stop the page."""
    result = get_result()
    if result is None:
        empty_state("No trained model yet",
                    "Train an AdaBoost model first — it takes under a minute with the default settings.",
                    icon="🤖", link="views/model_training.py", link_label="Go to Model Training")
        st.stop()
    return result


PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
