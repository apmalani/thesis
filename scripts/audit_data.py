#!/usr/bin/env python3
"""Run data integrity audit."""

import argparse

from redistricting.utils.data_audit import audit_data_directory, print_audit_report
from redistricting.utils.paths import get_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data directory")
    parser.add_argument("--state", type=str, default="az")
    args = parser.parse_args()

    basepath = str(get_data_dir(None, "processed"))
    results = audit_data_directory(args.state, basepath)
    print_audit_report(results)
    raise SystemExit(0 if results["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

