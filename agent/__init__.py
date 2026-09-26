"""Datacenter Agent."""

import os

# Quiet MLflow's startup banner and its git probing (the container has no git).
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
