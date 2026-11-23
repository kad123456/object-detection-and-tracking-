"""Streaming video pipeline for detection, tracking, annotation, and output."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.csv_writer import CsvTrackingWriter
from src.detector import Detector
from src.tracker import ObjectTracker, TrackObservation
from src.utils import CrossingCounter, FPSCounter, Line, TrailStore, draw_status, draw_tracks

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved options needed to process one video source."""

    source: str
    model: str = "yolov8n.pt"
    tracker: str = "bytetrack"
    confidence: float = 0.4
    image_size: int = 640
    classes: tuple[str, ...] = ()
    frame_skip: int = 1
    save: bool = True
    show: bool = False
    output_dir: str = "outputs"
    trail_length: int = 24
    line_crossing: Line | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.image_size < 1:
            raise ValueError("image size must be positive")
        if self.frame_skip < 1:
            raise ValueError("frame skip must be at least 1")
        if self.trail_length < 1:
            raise ValueError("trail length must be at least 1")


def _source_value(source: str) -> int | str:
    """Convert digit-only source values into webcam indices."""
    stripped = source.strip()
    return int(stripped) if stripped.isdecimal() else stripped


def _is_stream(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _output_stem(source: str) -> str:
    """Create a safe, readable filename stem from a file, stream, or device."""
    source_value = _source_value(source)
    if isinstance(source_value, int):
        candidate = f"webcam_{source_value}"
    elif _is_stream(source):
        candidate = "stream"
    else:
        candidate = Path(source_value).stem or "video"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate)


def run_pipeline(
    config: PipelineConfig,
    detector: Any | None = None,
    tracker: Any | None = None,
) -> dict[str, Any]:
    """Process a source one frame at a time and finalize all outputs on exit.

    Optional injected detector and tracker objects make the video pipeline easy to
    exercise without downloading model weights.
    """
    source = _source_value(config.source)
    if isinstance(source, str) and not _is_stream(source) and not Path(source).is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        if isinstance(source, int):
            raise RuntimeError(f"Could not open webcam {source}.")
        raise RuntimeError(f"Could not open video or stream: {config.source}")

    try:
        active_detector = detector or Detector(config.model)
        active_tracker = tracker or ObjectTracker(config.tracker, active_detector.names)
        class_ids = active_detector.resolve_classes(config.classes) if config.classes else None
    except BaseException:
        capture.release()
        raise

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0

    output_directory = Path(config.output_dir)
    stem = _output_stem(config.source)
    video_path = output_directory / f"{stem}_tracked.mp4" if config.save else None
    csv_path = output_directory / f"{stem}_tracks.csv" if config.save else None

    video_writer: cv2.VideoWriter | None = None
    csv_writer: CsvTrackingWriter | None = None
    if config.save:
        output_directory.mkdir(parents=True, exist_ok=True)
        assert csv_path is not None
        csv_writer = CsvTrackingWriter(csv_path)

    trails = TrailStore(config.trail_length)
    crossing_counter = CrossingCounter(config.line_crossing) if config.line_crossing else None
    latest_tracks: list[TrackObservation] = []
    fps_counter = FPSCounter()
    started_at = time.perf_counter()
    inference_seconds = 0.0
    tracking_seconds = 0.0
    drawing_seconds = 0.0
    frame_count = 0
    processed_count = 0

    LOGGER.info(
        "Processing source=%s model=%s tracker=%s confidence=%.2f frame_skip=%d",
        config.source,
        config.model,
        config.tracker,
        config.confidence,
        config.frame_skip,
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_number = frame_count
            should_infer = frame_number % config.frame_skip == 0
            if should_infer:
                inference_started = time.perf_counter()
                result = active_detector.predict(
                    frame,
                    confidence=config.confidence,
                    image_size=config.image_size,
                    classes=class_ids,
                )
                inference_seconds += time.perf_counter() - inference_started

                tracking_started = time.perf_counter()
                latest_tracks = active_tracker.update(result, frame)
                tracking_seconds += time.perf_counter() - tracking_started
                processed_count += 1
                trails.update(latest_tracks)
                if crossing_counter is not None:
                    crossing_counter.update(latest_tracks)
                if csv_writer is not None:
                    csv_writer.write_frame(frame_number, latest_tracks)

            draw_started = time.perf_counter()
            now = time.perf_counter()
            display_fps = fps_counter.tick(now)
            annotated = draw_tracks(frame, latest_tracks, trails)
            crossing_count = crossing_counter.count if crossing_counter is not None else 0
            annotated = draw_status(
                annotated,
                display_fps,
                crossing_count,
                config.line_crossing,
            )
            drawing_seconds += time.perf_counter() - draw_started

            if config.save and video_writer is None:
                height, width = annotated.shape[:2]
                assert video_path is not None
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {video_path}")

            if video_writer is not None:
                video_writer.write(annotated)
            frame_count += 1
            if config.show:
                cv2.imshow("Object Detection and Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        LOGGER.info("Stopping on Ctrl+C; finalizing output files.")
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        if csv_writer is not None:
            csv_writer.close()
        if config.show:
            cv2.destroyAllWindows()

    if frame_count == 0:
        raise RuntimeError(f"The source produced no readable frames: {config.source}")

    elapsed = time.perf_counter() - started_at
    metrics = {
        "frames": frame_count,
        "inference_frames": processed_count,
        "average_fps": frame_count / elapsed if elapsed else 0.0,
        "inference_ms": 1000.0 * inference_seconds / processed_count if processed_count else 0.0,
        "tracking_ms": 1000.0 * tracking_seconds / processed_count if processed_count else 0.0,
        "drawing_ms": 1000.0 * drawing_seconds / frame_count if frame_count else 0.0,
        "video_path": str(video_path) if video_path is not None else None,
        "csv_path": str(csv_path) if csv_path is not None else None,
        "crossings": crossing_counter.count if crossing_counter is not None else None,
    }
    LOGGER.info(
        "Finished %d frames (%.2f average FPS); mean latency: inference %.2f ms, "
        "tracking %.2f ms, drawing %.2f ms",
        metrics["frames"],
        metrics["average_fps"],
        metrics["inference_ms"],
        metrics["tracking_ms"],
        metrics["drawing_ms"],
    )
    if video_path is not None:
        LOGGER.info("Annotated video: %s", video_path)
        LOGGER.info("Tracking CSV: %s", csv_path)
    if metrics["crossings"] is not None:
        LOGGER.info("Unique line crossings: %d", metrics["crossings"])
    return metrics
