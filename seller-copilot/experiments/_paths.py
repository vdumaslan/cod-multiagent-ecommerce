"""Repo paths shared by experiment scripts."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COPILOT_ROOT = Path(__file__).resolve().parents[1]
DATA_AGENT = COPILOT_ROOT / "data" / "agent_dataset"
DATA_SYN = COPILOT_ROOT / "data" / "synthetic"
ARTIFACTS = COPILOT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS / "splits"
EVALS_DIR = ARTIFACTS / "evals"
FAISS_DIR = ARTIFACTS / "faiss"
LOGS_DIR = ARTIFACTS / "logs"
