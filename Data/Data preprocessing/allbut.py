#!/usr/bin/env python3
"""
Python conversion of allbut.pl.

Usage:
    python allbut.py base_name start stop max_test [ratings ...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, TextIO


def iter_lines(files: list[str]) -> Iterable[str]:
    if files:
        for file_path in files:
            with open(file_path, "r", encoding="latin-1") as f:
                for line in f:
                    yield line
    else:
        for line in sys.stdin:
            yield line


def main() -> int:
    if len(sys.argv) < 5:
        print(f"Usage: {Path(sys.argv[0]).name} base_name start stop max_test [ratings ...]", file=sys.stderr)
        return 1

    basename = sys.argv[1]
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    maxtest = int(sys.argv[4])
    rating_files = sys.argv[5:]

    test_path = f"{basename}.test"
    base_path = f"{basename}.base"

    ratingcnt: dict[str, int] = {}
    testcnt = 0

    with open(test_path, "w", encoding="utf-8") as testfile, open(base_path, "w", encoding="utf-8") as basefile:
        for line in iter_lines(rating_files):
            if not line.strip():
                continue
            user = line.split()[0]
            ratingcnt[user] = ratingcnt.get(user, 0) + 1

            if (testcnt < maxtest or maxtest <= 0) and start <= ratingcnt[user] <= stop:
                testcnt += 1
                testfile.write(line)
            else:
                basefile.write(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
