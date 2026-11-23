"""Streaming CSV output for tracking observations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, TextIO

from src.tracker import TrackObservation


CSV_COLUMNS = ("frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2")


class CsvTrackingWriter:
    """Write tracking rows incrementally without retaining video data in memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)

    def write_frame(self, frame_number: int, tracks: Iterable[TrackObservation]) -> None:
        """Write all tracked objects for one source frame and flush the rows."""
        for track in tracks:
            self._writer.writerow(
                (
                    frame_number,
                    track.track_id,
                    track.class_name,
                    f"{track.confidence:.6f}",
                    f"{track.x1:.3f}",
                    f"{track.y1:.3f}",
                    f"{track.x2:.3f}",
                    f"{track.y2:.3f}",
                )
            )
        self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CsvTrackingWriter":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
