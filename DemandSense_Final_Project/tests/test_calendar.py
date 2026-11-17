import pandas as pd

from data_pipeline.festival_calendar import festival_name_for, festival_score, upcoming_festivals, wedding_season


def test_festival_score_peaks_on_festival_day():
    assert festival_score(["2024-11-01"])[0] == 1.0          # Diwali 2024


def test_festival_score_falls_linearly_and_stops_after_four_days():
    s = festival_score(pd.to_datetime(["2024-11-02", "2024-11-03", "2024-11-05", "2024-11-06"]))
    assert list(s) == [0.75, 0.5, 0.0, 0.0]


def test_wedding_season_months():
    flags = wedding_season(pd.to_datetime(["2024-01-10", "2024-03-10", "2024-05-10", "2024-08-10"]))
    assert list(flags) == [1, 0, 1, 0]


def test_upcoming_and_name():
    assert upcoming_festivals("2025-01-01", 1)[0][1] == "Holi"
    assert festival_name_for("2024-11-02") == "Diwali"
    assert festival_name_for("2024-06-15") is None
