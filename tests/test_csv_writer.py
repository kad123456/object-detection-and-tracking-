import csv

from src.csv_writer import CSV_COLUMNS, CsvTrackingWriter
from src.tracker import TrackObservation


def test_csv_writer_writes_expected_columns_and_rows(tmp_path) -> None:
    path = tmp_path / "nested" / "tracks.csv"
    track = TrackObservation(4, 2, "car", 0.87654, 1.25, 2.5, 30.75, 40.0)

    with CsvTrackingWriter(path) as writer:
        writer.write_frame(12, [track])

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))

    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[1] == ["12", "4", "car", "0.876540", "1.250", "2.500", "30.750", "40.000"]
