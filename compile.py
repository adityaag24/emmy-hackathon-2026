#!/usr/bin/env python3
"""Submission entry point: python compile.py kernel.ir.json > kernel.ptx."""

import argparse
import json
import sys
from pathlib import Path

from emmy_ptx import CompileError, compile_file


def worker() -> int:
    """Compile JSON-line requests in one process, amortizing Emmy imports."""
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("ir"), str):
                raise ValueError('request must contain a string "ir" path')
            response = {"ok": True, "ptx": compile_file(Path(request["ir"]))}
        except (CompileError, OSError, ValueError, KeyError, TypeError) as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path, nargs="?")
    parser.add_argument("--worker", action="store_true", help="serve JSON-line compile requests on stdin/stdout")
    args = parser.parse_args()
    if args.worker:
        if args.ir is not None:
            parser.error("--worker cannot be combined with an IR path")
        return worker()
    if args.ir is None:
        parser.error("an IR path is required")
    try:
        ptx = compile_file(args.ir)
    except (CompileError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
