"""Command-line entry point for object detection and tracking."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.pipeline import Line, PipelineConfig, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and track objects in video.")
    parser.add_argument("--source", required=True, help="Video path, webcam index, or RTSP/HTTP URL")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML defaults file")
    parser.add_argument("--model", help="Ultralytics model weights (default: yolov8n.pt)")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort"))
    parser.add_argument("--conf", type=float, help="Minimum detection confidence")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    parser.add_argument("--classes", nargs="+", help="Class names or IDs to keep, e.g. person car")
    parser.add_argument("--frame-skip", type=int, help="Run inference once per N source frames")
    parser.add_argument("--output-dir", help="Directory for annotated video and tracking CSV")
    parser.add_argument("--trail-length", type=int, help="Maximum number of centers in each motion trail")
    parser.add_argument(
        "--line-crossing",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional finite line segment for unique-ID crossing counts",
    )
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _line_from_config(value: Any) -> Line | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError("line_crossing must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    return ((x1, y1), (x2, y2))


def _load_options(args: argparse.Namespace) -> PipelineConfig:
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    options: dict[str, Any] = {
        "model": raw.get("model", "yolov8n.pt"),
        "tracker": raw.get("tracker", "bytetrack"),
        "confidence": raw.get("conf", 0.4),
        "image_size": raw.get("imgsz", 640),
        "classes": tuple(str(value) for value in raw.get("classes", [])),
        "frame_skip": raw.get("frame_skip", 1),
        "save": raw.get("save", True),
        "show": raw.get("show", False),
        "output_dir": raw.get("output_dir", "outputs"),
        "trail_length": raw.get("trail_length", 24),
        "line_crossing": _line_from_config(raw.get("line_crossing")),
    }
    overrides = {
        "model": args.model,
        "tracker": args.tracker,
        "confidence": args.conf,
        "image_size": args.imgsz,
        "classes": tuple(args.classes) if args.classes is not None else None,
        "frame_skip": args.frame_skip,
        "save": args.save,
        "show": args.show,
        "output_dir": args.output_dir,
        "trail_length": args.trail_length,
        "line_crossing": _line_from_config(args.line_crossing),
    }
    options.update({key: value for key, value in overrides.items() if value is not None})
    return PipelineConfig(source=args.source, **options)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options, run the video pipeline, and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_options(args)
        run_pipeline(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
