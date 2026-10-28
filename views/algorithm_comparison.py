"""Algorithm Comparison: Decision Tree vs AdaBoost vs Random Forest under identical conditions."""

import streamlit as st

from src import visualization as viz
from src.comparison import ComparisonConfig, best_per_metric, compare_models, load_comparison, save_comparison
from ui.components import PLOTLY_CONFIG, callout, card_title, empty_state, page_header
from ui.state import get_result, require_data

page_header("Algorithm Comparison", "Same training images, same test images, three tree-based models.", "cmp")
data = require_data()
current = get_result()
base = current.config if current else None

with st.container(border=True):
    card_title("Comparison settings", "All three models use the identical stratified training subset and the full "
                                      "official 10,000-image test set.")
    with st.form("cmp_form"):
        c1, c2, c3, c4 = st.columns(4)
        train_size = c1.select_slider("Training sample size", [2000, 5000, 10000, 20000],
                                      value=base.train_size if base and base.train_size in (2000, 5000, 10000, 20000) else 10000)
        seed = c1.number_input("Random seed", 0, 99999, base.random_state if base else 42)
        dt_depth = c2.slider("Decision Tree max depth", 2, 30, 12)
        rf_trees = c3.slider("Random Forest trees", 10, 300, 100, step=10)
        rf_depth = c3.slider("Random Forest max depth (0 = unlimited)", 0, 30, 0)
        ada_n = c4.slider("AdaBoost estimators", 10, 200, base.n_estimators if base else 50)
        ada_depth = c4.slider("AdaBoost tree depth", 1, 12, base.max_depth if base else 6)
        ada_lr = c4.select_slider("AdaBoost learning rate", [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
                                  value=base.learning_rate if base and base.learning_rate in
                                  (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5) else 0.5)
        run = st.form_submit_button("Run comparison", type="primary", icon=":material/play_arrow:")

cfg = ComparisonConfig(int(train_size), int(seed), int(dt_depth), int(rf_trees), int(rf_depth),
                       int(ada_n), float(ada_lr), int(ada_depth))
df = load_comparison(cfg)

if run:
    if df is not None:
        st.toast("These exact settings were already compared — showing saved results.", icon="♻️")
    else:
        with st.status("Training three models on the same data…", expanded=True) as status:
            try:
                df = compare_models(cfg, data.x_train, data.y_train, data.x_test, data.y_test,
                                    progress=lambda m: st.write(f"• Training and testing {m}"))
                save_comparison(cfg, df)
                status.update(label="Comparison complete", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Comparison failed", state="error")
                st.error(f"{exc}. Try a smaller training sample size.")
                df = None

if df is None:
    empty_state("No comparison for these settings yet",
                "Press <b>Run comparison</b>. With 10,000 training images it takes about a minute.", icon="⚖️")
    st.stop()

best = best_per_metric(df)
st.write("")
with st.container(border=True):
    card_title("Results", "Green = best in that column")
    styled = (df.set_index("Model").style
              .format({"Test accuracy": "{:.2%}", "Macro precision": "{:.2%}", "Macro recall": "{:.2%}",
                       "Macro F1": "{:.2%}", "Training time (s)": "{:.2f}", "Inference time (s)": "{:.3f}"})
              .highlight_max(subset=["Test accuracy", "Macro precision", "Macro recall", "Macro F1"], color="#DDF3A1")
              .highlight_min(subset=["Training time (s)", "Inference time (s)"], color="#DDF3A1"))
    st.dataframe(styled, width="stretch")
    st.caption(f"Training images: {cfg.train_size:,} (stratified, seed {cfg.random_state}) · test images: {data.n_test:,} "
               "· precision/recall/F1 use macro averaging.")

c1, c2, c3 = st.columns(3)
for col, metric, fmt in ((c1, "Test accuracy", ".2%"), (c2, "Macro F1", ".2%"), (c3, "Training time (s)", "s")):
    with col, st.container(border=True):
        st.plotly_chart(viz.comparison_bars(df, metric, height=300, fmt=fmt), width="stretch", config=PLOTLY_CONFIG)

r = df.set_index("Model")
ada, dt, rf = r.loc["AdaBoost"], r.loc["Decision Tree"], r.loc["Random Forest"]
with st.container(border=True):
    card_title("Analysis of this run")
    st.markdown(f"""
- **Most accurate:** {best['Test accuracy']} ({r['Test accuracy'].max():.2%}). **Fastest to train:** {best['Training time (s)']}.
- **AdaBoost vs a single Decision Tree:** {ada['Test accuracy']:.2%} vs {dt['Test accuracy']:.2%}
  ({ada['Test accuracy'] - dt['Test accuracy']:+.1%}). Boosting many shallow trees fixes errors a single tree keeps making.
- **AdaBoost vs Random Forest:** {ada['Test accuracy']:.2%} vs {rf['Test accuracy']:.2%}
  ({ada['Test accuracy'] - rf['Test accuracy']:+.1%}); training {ada['Training time (s)']:.1f}s vs {rf['Training time (s)']:.1f}s.
""")
    st.markdown("""
| Model | Strengths | Weaknesses |
|---|---|---|
| **Decision Tree** | Very fast, easy to visualise and explain | One tree overfits or underfits easily; lowest accuracy |
| **AdaBoost** | Turns weak trees into a strong model; focuses on hard examples; few, small trees | Trains sequentially (one core); sensitive to noisy/mislabelled images |
| **Random Forest** | Strong out of the box; trees train in parallel on all cores; robust to noise | Many deep trees → larger model and slower predictions |

**Why can Random Forest beat AdaBoost on MNIST?** Random Forest grows many *deep* trees in parallel on bootstrap
samples and averages them, which reduces variance — very effective for 784 noisy pixel features. AdaBoost with shallow
trees reduces bias step by step; it needs deeper weak learners or more rounds to match. Neither model sees the 2-D
shape of the digit, which is why convolutional neural networks reach ~99%.
""")
    callout("<b>Disclaimer:</b> these numbers depend on the training subset size, hyper-parameters, random seed and the "
            "computer used (especially the timings). They are measured results for <i>this</i> configuration, not "
            "universal rankings.", "warn")
