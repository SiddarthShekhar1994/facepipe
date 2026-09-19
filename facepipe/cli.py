"""Command-line entry point. The only module that knows about argv."""

import argparse
import dataclasses
import pprint
import sys

from facepipe.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.toml", help="TOML config file (default: %(default)s)")

    parser = argparse.ArgumentParser(prog="facepipe", description="Local face detection and recognition pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-config", parents=[common], help="load and validate the config, print the resolved values")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"error: {args.config}: {e}", file=sys.stderr)
        return 1

    if args.command == "show-config":
        print(f"# {args.config}")
        pprint.pprint(dataclasses.asdict(cfg))
    return 0
