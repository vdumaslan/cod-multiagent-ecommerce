from __future__ import annotations

import argparse

from prefect_flow import seller_copilot_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="seller-copilot/config/pipeline.yaml")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    seller_copilot_pipeline(config_path=args.config, max_rows=args.max_rows)


if __name__ == "__main__":
    main()

