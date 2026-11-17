from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="credit-risk")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch the UCI dataset into data/raw")
    sub.add_parser("train", help="fit, cross-validate and score the hold-out; writes reports/")
    args = parser.parse_args()
    if args.command == "download":
        from .data import download

        print(download())
    else:
        from .train import run

        report = run()
        report.pop("deciles")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
