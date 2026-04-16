from __future__ import annotations

"""
Compatibility alias.

`Docs/DATA_PROCESSING_REPORT.md` references `python -m copilot_v2.scripts.tune_retrieval`,
but the maintained implementation lives in `tune_retrieval_dense.py`.
"""

from copilot_v2.scripts.tune_retrieval_dense import main


if __name__ == "__main__":
    main()

