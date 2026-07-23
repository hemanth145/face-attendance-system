from src.attendance import AttendanceLog


def test_mark_present_logs_a_new_entry():
    log = AttendanceLog(cooldown_seconds=60.0)
    logged = log.mark_present("Alice", now=1000.0)
    assert logged is True
    assert len(log) == 1


def test_mark_present_suppresses_duplicate_within_cooldown():
    log = AttendanceLog(cooldown_seconds=60.0)
    log.mark_present("Alice", now=1000.0)
    logged_again = log.mark_present("Alice", now=1010.0)
    assert logged_again is False
    assert len(log) == 1


def test_mark_present_allows_reentry_after_cooldown():
    log = AttendanceLog(cooldown_seconds=60.0)
    log.mark_present("Alice", now=1000.0)
    logged_again = log.mark_present("Alice", now=1065.0)
    assert logged_again is True
    assert len(log) == 2


def test_to_dataframe_has_expected_columns():
    log = AttendanceLog()
    log.mark_present("Bob", now=1000.0)
    df = log.to_dataframe()
    assert list(df.columns) == ["name", "timestamp"]
    assert len(df) == 1


def test_to_csv_on_empty_log_still_has_header():
    log = AttendanceLog()
    csv_text = log.to_csv()
    assert "name" in csv_text
    assert "timestamp" in csv_text
