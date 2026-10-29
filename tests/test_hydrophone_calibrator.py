"""Calibration checks against generated truth, without capture hardware."""

import numpy as np
import pytest
import struct

from Scripts.check_calibration import inspect_capture
from Scripts.Hydrophone_Array import HydrophoneArray
from Scripts.Hydrophone_Calibrator import HydrophoneCalibrator
from Scripts.generate_calibration_data import make_capture, write_capture


def make_calibrator(**options):
    times, channels, truth = make_capture(**options)
    array = HydrophoneArray(sampling_freq=truth["sampling_freq_hz"])
    for hydro, channel in zip(array.hydrophones, channels):
        hydro.times = times.copy()
        hydro.voltages = channel.copy()
    return HydrophoneCalibrator(array), truth


def assert_corrections(cal, truth, *, amplitude_rtol=0.005, delay_atol=1):
    hydros = cal.hydroarray.hydrophones
    assert all(h.found_peak for h in hydros)
    np.testing.assert_allclose([h.freq_adjust for h in hydros], truth["expected_freq_adjust"], rtol=0.002)
    np.testing.assert_allclose([h.ps_adjust for h in hydros], truth["expected_ps_adjust_samples"], atol=delay_atol, rtol=0)
    np.testing.assert_allclose([h.ampl_adjust for h in hydros], truth["expected_ampl_adjust"], rtol=amplitude_rtol)


def test_identical_channels_have_identity_corrections():
    cal, truth = make_calibrator(amplitudes=(0.8,) * 4, delays=(0,) * 4)
    cal.calibrate_all()
    assert_corrections(cal, truth, amplitude_rtol=1e-12, delay_atol=0)


def test_known_gains_and_early_and_late_arrivals():
    cal, truth = make_calibrator()
    original = [h.voltages.copy() for h in cal.hydroarray.hydrophones]
    cal.calibrate_all()
    assert_corrections(cal, truth)
    for hydro, raw in zip(cal.hydroarray.hydrophones, original):
        np.testing.assert_array_equal(hydro.voltages, raw)


def test_different_carrier_frequencies():
    cal, truth = make_calibrator(frequencies=(31250.0, 30000.0, 35000.0, 37500.0))
    cal.calibrate_all()
    # The band's frequency response moves the operational envelope crossing
    # slightly when carriers differ; four samples is 5.12 us at this rate.
    assert_corrections(cal, truth, delay_atol=4)


def test_carriers_above_quarter_sample_rate_are_not_discarded():
    cal, truth = make_calibrator(sampling_freq=128000.0, num_samples=12800,
                               frequencies=(35000.0,) * 4)
    _, frequency, _ = cal.read_vals(cal.hydroarray.hydrophones[0])
    assert frequency == pytest.approx(35000.0, abs=10.0)
    cal.calibrate_all()
    assert_corrections(cal, truth)


@pytest.mark.parametrize("seed", [7, 42, 20261029])
def test_noise_and_dc_offsets(seed):
    cal, truth = make_calibrator(noise_std=0.01, dc_offsets=(0.1, -0.05, 0.2, -0.1), seed=seed)
    cal.calibrate_all()
    assert_corrections(cal, truth, amplitude_rtol=0.02, delay_atol=8)


def test_fake_capture_is_reproducible():
    first = make_capture(noise_std=0.01)
    second = make_capture(noise_std=0.01)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_binary_capture_round_trip(tmp_path):
    times, channels, truth = make_capture(noise_std=0.01)
    path = tmp_path / "synthetic.bin"
    write_capture(path, times, channels, truth)
    array = HydrophoneArray(sampling_freq=truth["sampling_freq_hz"])
    array.load_from_path(str(path))
    for hydro, original in zip(array.hydrophones, channels):
        np.testing.assert_allclose(hydro.times, times, atol=1e-15)
        np.testing.assert_array_equal(hydro.voltages, original.astype("<f4"))
    cal = HydrophoneCalibrator(array)
    cal.calibrate_all()
    assert_corrections(cal, truth, amplitude_rtol=0.02, delay_atol=8)


def test_calibrate_single_matches_all():
    cal, truth = make_calibrator()
    reference = cal.read_vals(cal.hydroarray.hydrophones[0])
    hydro = cal.hydroarray.hydrophones[1]
    cal.calibrate_single(hydro, *reference)
    assert hydro.freq_adjust == pytest.approx(truth["expected_freq_adjust"][1], rel=0.002)
    assert hydro.ampl_adjust == pytest.approx(truth["expected_ampl_adjust"][1], rel=0.005)
    assert hydro.ps_adjust == pytest.approx(truth["expected_ps_adjust_samples"][1], abs=1)


def test_repeated_calibration_does_not_accumulate_corrections():
    cal, truth = make_calibrator()
    cal.calibrate_all()
    cal.calibrate_all()
    assert_corrections(cal, truth)


@pytest.mark.parametrize("channel", [0, 2])
def test_silent_channel_rejected_without_overwriting_coefficients(channel):
    cal, _ = make_calibrator()
    for hydro in cal.hydroarray.hydrophones:
        hydro.freq_adjust, hydro.ps_adjust, hydro.ampl_adjust = 1.1, 12, 0.9
    cal.hydroarray.hydrophones[channel].voltages[:] = 0
    with pytest.raises(ValueError, match="No detectable calibration ping"):
        cal.calibrate_all()
    assert all((h.freq_adjust, h.ps_adjust, h.ampl_adjust) == (1.1, 12, 0.9)
               for h in cal.hydroarray.hydrophones)


@pytest.mark.parametrize("problem", ["length", "nan", "infinity", "sample_rate", "short", "missing"])
def test_invalid_captures_are_rejected(problem):
    cal, _ = make_calibrator()
    hydro = cal.hydroarray.hydrophones[1]
    if problem == "length":
        hydro.voltages = hydro.voltages[:-1]
    elif problem == "nan":
        hydro.voltages[100] = np.nan
    elif problem == "infinity":
        hydro.voltages[100] = np.inf
    elif problem == "sample_rate":
        hydro.times *= 2
    elif problem == "short":
        hydro.times, hydro.voltages = hydro.times[:32], hydro.voltages[:32]
    else:
        hydro.voltages = None
    with pytest.raises((ValueError, RuntimeError)):
        cal.calibrate_all()


@pytest.mark.parametrize("payload", [
    b"short",
    struct.pack("<QId", 100, 3, 1 / 781250),
    struct.pack("<QId", 100, 4, float("nan")),
    struct.pack("<QId", 100, 4, 1 / 781250) + b"truncated",
])
def test_capture_checker_rejects_malformed_binary(tmp_path, payload):
    path = tmp_path / "invalid.bin"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        inspect_capture(path)


def test_capture_checker_reports_expected_corrections(tmp_path):
    times, channels, truth = make_capture()
    path = tmp_path / "clean.bin"
    write_capture(path, times, channels, truth)
    report = inspect_capture(path)
    assert report["sampling_freq_hz"] == truth["sampling_freq_hz"]
    assert [h["ps_adjust_samples"] for h in report["channels"]] == truth["expected_ps_adjust_samples"]
    np.testing.assert_allclose([h["ampl_adjust"] for h in report["channels"]],
                               truth["expected_ampl_adjust"], rtol=0.005)
    with pytest.raises(ValueError, match="below Nyquist"):
        inspect_capture(path, high=800000)
