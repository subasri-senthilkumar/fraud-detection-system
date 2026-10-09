"""Load demo transactions through the API so the dashboard has data to show.

    python scripts/seed_demo.py --n 1500 --api-url http://localhost:8000

Uses the held-out test split written by training (data/processed/test.csv); the
true labels are dropped, so everything stored is the model's own prediction.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED_DIR, RAW_REQUIRED_COLUMNS  # noqa: E402

FIELDS = ["transaction_id", "customer_id", *RAW_REQUIRED_COLUMNS]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    p.add_argument("--data", type=Path, default=DATA_PROCESSED_DIR / "test.csv")
    p.add_argument("--chunk", type=int, default=250)
    args = p.parse_args()

    if not args.data.exists():
        sys.exit(f"{args.data} not found - run `python -m src.train` first.")
    df = pd.read_csv(args.data).sample(frac=1, random_state=0).head(args.n)
    headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}

    sent = 0
    for start in range(0, len(df), args.chunk):
        chunk = df.iloc[start:start + args.chunk][FIELDS]
        records = [{k: (None if pd.isna(v) else v) for k, v in row.items()}
                   for row in chunk.to_dict("records")]
        for r in records:  # numpy -> plain python types, 0/1 -> bool for the schema
            r["is_new_device"] = bool(r["is_new_device"])
            r["country_mismatch"] = bool(r["country_mismatch"])
            r["account_age_days"] = int(r["account_age_days"])
            r["txn_count_24h"] = int(r["txn_count_24h"])
        resp = requests.post(f"{args.api_url}/batch-predict", headers=headers, timeout=120,
                             json={"transactions": records, "include_explanations": False})
        resp.raise_for_status()
        sent += resp.json()["count"]
        print(f"scored {sent}/{len(df)}")


if __name__ == "__main__":
    main()
