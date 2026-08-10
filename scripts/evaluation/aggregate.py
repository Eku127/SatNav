#!/usr/bin/env python3
"""Validate rank-local outputs and write a merged evaluation summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make direct execution from a source checkout work before editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from satnav.evaluation import ResultValidationError, aggregate_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a diagnostic summary without requiring every done marker",
    )
    parser.add_argument(
        "--fail-on-episode-error",
        action="store_true",
        help="return non-zero when any episode has an explicit error record",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = aggregate_run(
            args.output_dir,
            strict=True,
            require_done=not args.allow_incomplete,
            fail_on_episode_error=args.fail_on_episode_error,
        )
    except ResultValidationError as error:
        if error.summary is not None:
            print(json.dumps(error.summary, indent=2, sort_keys=True))
        print(f"evaluation validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
