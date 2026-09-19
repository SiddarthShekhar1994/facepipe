"""Command-line entry point. The only module that knows about argv."""

import argparse
import dataclasses
import pprint
import sys

from facepipe.config import ConfigError, load_config
from facepipe.run import run


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.toml", help="TOML config file (default: %(default)s)")

    parser = argparse.ArgumentParser(prog="facepipe", description="Local face detection and recognition pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-config", parents=[common], help="load and validate the config, print the resolved values")
    p_run = sub.add_parser("run", parents=[common], help="live webcam loop with per-stage timing; q or Esc quits")
    p_run.add_argument("--frames", type=int, default=None, help="stop after this many frames (default: run until quit)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"error: {args.config}: {e}", file=sys.stderr)
        return 1

    if args.command == "show-config":
        print(f"# {args.config}")
        pprint.pprint(dataclasses.asdict(cfg))
    elif args.command == "run":
        run(cfg, max_frames=args.frames)
    return 0
