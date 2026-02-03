from __future__ import annotations
import argparse
from a_top10.main import run_pipeline

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--date", default="", help="trade_date YYYYMMDD, empty => auto")
    r.add_argument("--config", default="configs/default.yml")
    r.add_argument("--dry", action="store_true", help="dry run: no write outputs")

    args = p.parse_args()
    if args.cmd == "run":
        run_pipeline(config_path=args.config, trade_date=args.date, dry_run=args.dry)

if __name__ == "__main__":
    main()
