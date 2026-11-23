import csv

import cv2
import numpy as np

from src.pipeline import PipelineConfig, run_pipeline
from src.tracker import TrackObservation


class FakeDetector:
    names = {0: "person"}

    def predict(self, frame, confidence, image_size, classes=None):
        return frame


class FakeTracker:
    def __init__(self):
        self.updates = 0

    def update(self, result, frame):
        self.updates += 1
        center_x = 5 + (self.updates - 1) * 10
        return [
            TrackObservation(
                track_id=3,
                class_id=0,
                class_name="person",
                confidence=0.95,
                x1=center_x - 4,
                y1=12,
                x2=center_x + 4,
                y2=28,
            )
        ]


def make_clip(path, frames=8):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((48, 64, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_pipeline_smoke_writes_video_and_tracking_csv(tmp_path):
    source = tmp_path / "sample.mp4"
    make_clip(source)
    output_dir = tmp_path / "outputs"
    config = PipelineConfig(
        source=str(source),
        frame_skip=2,
        output_dir=str(output_dir),
        line_crossing=((10, 0), (10, 47)),
    )

    metrics = run_pipeline(config, detector=FakeDetector(), tracker=FakeTracker())

    assert metrics["frames"] == 8
    assert metrics["inference_frames"] == 4
    assert metrics["crossings"] == 1
    assert metrics["average_fps"] > 0
    assert metrics["inference_ms"] >= 0
    assert metrics["tracking_ms"] >= 0
    assert metrics["drawing_ms"] >= 0
    assert (output_dir / "sample_tracked.mp4").is_file()

    with (output_dir / "sample_tracks.csv").open(newline="", encoding="utf-8") as source_csv:
        rows = list(csv.reader(source_csv))
    assert rows[0] == ["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"]
    assert [row[0] for row in rows[1:]] == ["0", "2", "4", "6"]

    output_video = cv2.VideoCapture(str(output_dir / "sample_tracked.mp4"))
    output_frames = 0
    while True:
        ok, _frame = output_video.read()
        if not ok:
            break
        output_frames += 1
    output_video.release()
    assert output_frames == 8


def test_pipeline_finalizes_partial_outputs_after_keyboard_interrupt(tmp_path):
    source = tmp_path / "interrupt.mp4"
    make_clip(source, frames=4)

    class InterruptingDetector(FakeDetector):
        calls = 0

        def predict(self, frame, confidence, image_size, classes=None):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return frame

    output_dir = tmp_path / "partial"
    metrics = run_pipeline(
        PipelineConfig(source=str(source), output_dir=str(output_dir)),
        detector=InterruptingDetector(),
        tracker=FakeTracker(),
    )

    assert metrics["frames"] == 1
    assert (output_dir / "interrupt_tracked.mp4").is_file()
    with (output_dir / "interrupt_tracks.csv").open(newline="", encoding="utf-8") as source_csv:
        rows = list(csv.reader(source_csv))
    assert len(rows) == 2
