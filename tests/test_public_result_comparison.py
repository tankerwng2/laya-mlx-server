from benchmarks.features_report import compare


def test_public_decimal_boundary_and_actual_overrun():
    errors, differences = compare({"p": 0.2}, {"p": 0.1999})
    assert errors == [0.0001] and not differences
    errors, differences = compare({"p": 0.2}, {"p": 0.1998})
    assert errors == [0.0002] and not differences
    assert compare({"p": 0.2}, {"p": float("nan")})[1]
    assert compare({"count": 0}, {"count": False})[1]
