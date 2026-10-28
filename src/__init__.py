"""Core machine-learning package for the MNIST AdaBoost project.

Nothing in ``src`` imports Streamlit: data loading, preprocessing, model
training, evaluation, comparison and the manual AdaBoost demo are plain
Python so they can be unit-tested and reused outside the web app.
The Streamlit layer lives in ``ui/`` (shared components) and ``views/`` (pages).
"""

__version__ = "1.0.0"
