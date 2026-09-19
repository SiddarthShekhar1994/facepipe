"""Command-line entry point. The only module that knows about argv."""

import argparse
import dataclasses
import pprint
import sys

from facepipe.bench import bench
from facepipe.config import ConfigError, load_config
from facepipe.dashboard import serve
from facepipe.enroll import enroll
from facepipe.evaluate import evaluate
from facepipe.run import run


def main(argv: list[str] | None = None) -> int:
    # Timing lines are printed once a second from long-running loops; keep
    # them flowing when stdout is a pipe or a log file, not just a terminal.
    sys.stdout.reconfigure(line_buffering=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.toml", help="TOML config file (default: %(default)s)")

    parser = argparse.ArgumentParser(prog="facepipe", description="Local face detection and recognition pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-config", parents=[common], help="load and validate the config, print the resolved values")
    p_run = sub.add_parser("run", parents=[common], help="live webcam loop with per-stage timing; q or Esc quits")
    p_run.add_argument("--frames", type=int, default=None, help="stop after this many frames (default: run until quit)")
    p_enroll = sub.add_parser("enroll", parents=[common], help="enroll one person from a folder of images")
    p_enroll.add_argument("name", help="letters, digits, '_' or '-'; also the label drawn on the video")
    p_enroll.add_argument("folder", help="directory of .jpg/.png images, one face each")
    p_serve = sub.add_parser("serve", parents=[common], help="web dashboard: live feed, enrolled list, enroll from webcam")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only; the feed has no auth)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_bench = sub.add_parser("bench", parents=[common], help="per-stage latency and FPS over a fixed video or image directory")
    p_bench.add_argument("--video", help="video file to run through the pipeline")
    p_bench.add_argument("--images", help="directory of images to run through the pipeline")
    p_bench.add_argument("--frames", type=int, default=300, help="frames to measure (default: %(default)s)")
    p_bench.add_argument("--warmup", type=int, default=30, help="frames to run and discard first (default: %(default)s)")
    p_bench.add_argument("--repeat", type=int, default=1, help="repeat the whole run and report ranges (default: %(default)s)")
    p_bench.add_argument("--input-size", type=int, default=None, help="override detector.input_size for this run")
    p_eval = sub.add_parser("eval", parents=[common], help="similarity distributions, TAR/FAR and the threshold, on LFW plus webcam probes")
    p_eval.add_argument("--lfw", default="data/eval/lfw", help="LFW root: one directory per person (default: %(default)s)")
    p_eval.add_argument("--pairs", default="data/eval/pairs.txt", help="the official LFW pairs.txt (default: %(default)s)")
    p_eval.add_argument("--probes", default="data/eval/probes", help="<dir>/<name>/*.jpg probes of enrolled people (default: %(default)s)")
    p_eval.add_argument("--folds", type=int, default=None, help="use only the first N LFW folds (default: all 10)")
    p_eval.add_argument("--out", default="data/eval/results", help="where the embedding cache and curves go (default: %(default)s)")
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
    elif args.command == "enroll":
        return enroll(cfg, args.name, args.folder)
    elif args.command == "serve":
        serve(cfg, args.host, args.port)
    elif args.command == "bench":
        return bench(cfg, args.video, args.images, args.frames, args.warmup, args.repeat, args.input_size)
    elif args.command == "eval":
        return evaluate(cfg, args.lfw, args.pairs, args.probes, args.folds, args.out)
    return 0
