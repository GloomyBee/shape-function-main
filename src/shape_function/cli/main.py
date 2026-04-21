from __future__ import annotations

import argparse
import sys


from typing import Sequence

from shape_function.cli.config import ConfigError
from shape_function.cli.train import add_train_subparser
from shape_function.eval.ood_eval import add_ood_eval_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shape-function", description="shape_function experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_subparser(subparsers)
    add_ood_eval_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except (ConfigError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
