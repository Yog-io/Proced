"""Radiometry: dB formulas, domain detection, uint16 round-trip (C.2)."""

import numpy as np
import pytest

from proced.config import DB_FORMULA_VERSION, NATIVE_FORMULA_VERSION
from proced.radiometry import (
    calibrate_window,
    db_to_uint16,
    detect_value_domain,
    to_db,
    uint16_to_db,
)


def test_spec_mapping_midpoint():
    """db=-15 → round(((−15+30)/30)*65535) = round(32767.5) = 32768."""
    out = db_to_uint16(np.array([-15.0]))
    assert out[0] == 32768


def test_round_trip_within_quantisation():
    dbs = np.array([-30.0, -20.0, -10.0, -0.5, 0.0])
    u = db_to_uint16(dbs)
    back = uint16_to_db(u)
    assert np.allclose(back, dbs, atol=30 / 65535 + 1e-4)


def test_clipping():
    u = db_to_uint16(np.array([-100.0, 50.0]))
    assert u[0] == 0
    assert u[1] == 65535


def test_domain_detection_negative_is_db():
    arr = np.array([-30.0, -5.0, 0.0], dtype=np.float32)
    assert detect_value_domain(arr) == "db"


def test_domain_detection_linear_power_not_misread_as_db():
    """THE archi bug: σ⁰ power ≤ 1 was treated as already-dB via max>1.0."""
    arr = np.array([0.001, 0.01, 0.1, 0.9], dtype=np.float32)
    assert detect_value_domain(arr) == "power"


def test_domain_detection_amplitude():
    arr = np.array([0.5, 5.0, 80.0, 300.0], dtype=np.float32)
    assert detect_value_domain(arr) == "amplitude"


def test_domain_native_for_uint8():
    assert detect_value_domain(np.zeros((4, 4), dtype=np.uint8)) == "native"


def test_power_uses_10log10_amplitude_uses_20log10():
    power = np.array([0.1], dtype=np.float32)
    amp = np.array([0.1], dtype=np.float32)
    assert to_db(power, "power")[0] == pytest.approx(-10.0, abs=1e-4)
    assert to_db(amp, "amplitude")[0] == pytest.approx(-20.0, abs=1e-4)


def test_calibrate_native_passthrough():
    win = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out, ver = calibrate_window(win, "native")
    assert ver == NATIVE_FORMULA_VERSION
    assert np.array_equal(out, win)


def test_calibrate_power_tags_v1():
    win = np.full((8, 8), 0.05, dtype=np.float32)
    out, ver = calibrate_window(win, "power")
    assert ver == DB_FORMULA_VERSION
    assert out.dtype == np.uint16
    # 0.05 → -13.01 dB → pixel ≈ ((-13.01+30)/30)*65535 ≈ 37180
    assert 36000 < out.mean() < 38500
