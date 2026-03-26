from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_acquisition.flows.stage1_amazon_hk_flow import run_stage1


def _bootstrap_gcp_credentials_from_env() -> None:
    # Supports passing full service-account JSON directly via env.
    raw_json = os.getenv("GCP_SA_KEY_JSON", "").strip()
    if raw_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        tmp_dir = Path("seller-copilot/.secrets")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "gcp-sa-from-env.json"
        tmp_path.write_text(raw_json, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp_path.resolve())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv("seller-copilot/.env")
    _bootstrap_gcp_credentials_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="seller-copilot/config/stage1_amazon_hk.yaml")
    args = parser.parse_args()
    out = run_stage1(config_path=args.config)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

