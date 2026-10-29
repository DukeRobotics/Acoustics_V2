"""Independent TOA truth: analytic Gaussian crossings and known sample delays."""

import numpy as np
import pytest

from Scripts.Hydrophone_Array import HydrophoneArray


def gaussian_capture(frequency=31250, amplitude=0.8, sigma=0.0004,
                     center=0.007, threshold=0.7):
    array = HydrophoneArray()
    array.threshold_factor = threshold
    hydro = array.hydrophones[0]
    hydro.times = np.arange(15625) / array.sampling_freq
    envelope = amplitude * np.exp(-0.5 * ((hydro.times - center) / sigma) ** 2)
    hydro.voltages = envelope * np.cos(2 * np.pi * frequency * hydro.times)
    expected = center - sigma * np.sqrt(-2 * np.log(threshold))
    return array, hydro, expected


@pytest.mark.parametrize("frequency", [28000, 31250, 37000])
@pytest.mark.parametrize("threshold", [0.3, 0.7, 0.9])
def test_arrival_matches_analytic_envelope_crossing(frequency, threshold):
    array, hydro, expected = gaussian_capture(frequency=frequency, threshold=threshold)
    array.estimate_toa_by_envelope(hydro)
    assert hydro.found_peak
    assert hydro.toa_time == pytest.approx(expected, abs=2 * array.dt)
    assert hydro.toa_time < hydro.toa_peak
    assert hydro.toa_peak == pytest.approx(0.007, abs=2 * array.dt)


@pytest.mark.parametrize("amplitude", [0.001, 0.5, 5.0])
@pytest.mark.parametrize("sigma", [0.0002, 0.0008])
def test_gain_and_burst_width_do_not_break_crossing(amplitude, sigma):
    array, hydro, expected = gaussian_capture(amplitude=amplitude, sigma=sigma)
    array.estimate_toa_by_envelope(hydro)
    assert hydro.toa_time == pytest.approx(expected, abs=2 * array.dt)


@pytest.mark.parametrize("offset", [0.0, 1.0, -10.0])
def test_silence_and_constant_dc_are_not_arrivals(offset):
    array, hydro, _ = gaussian_capture()
    hydro.voltages[:] = offset
    array.estimate_toa_by_envelope(hydro)
    assert not hydro.found_peak
    assert hydro.toa_idx is hydro.toa_time is hydro.toa_peak is None


@pytest.mark.parametrize("problem", ["missing", "length", "nan", "spacing", "threshold", "complex"])
def test_failed_analysis_clears_previous_detection(problem):
    array, hydro, _ = gaussian_capture()
    array.estimate_toa_by_envelope(hydro)
    assert hydro.found_peak
    if problem == "missing":
        hydro.voltages = None
    elif problem == "length":
        hydro.voltages = hydro.voltages[:-1]
    elif problem == "nan":
        hydro.voltages[100] = np.nan
    elif problem == "spacing":
        hydro.times *= 2
    elif problem == "threshold":
        array.threshold_factor = 1
    else:
        hydro.voltages = hydro.voltages.astype(complex) + 1j
    with pytest.raises((RuntimeError, ValueError)):
        array.estimate_toa_by_envelope(hydro)
    assert not hydro.found_peak
    assert hydro.toa_idx is hydro.toa_time is hydro.toa_peak is None
    assert hydro.envelope is hydro.filtered_signal is None


def test_relative_delays_and_selection():
    array, hydro, _ = gaussian_capture()
    delays = (0, 81, -63, 125)
    for h, delay in zip(array.hydrophones, delays):
        h.times = hydro.times.copy()
        h.voltages = np.roll(hydro.voltages.copy(), delay)
    array.estimate_selected_by_envelope([True, True, True, False])
    reference = array.hydrophones[0].toa_idx
    assert [h.toa_idx - reference for h in array.hydrophones[:3]] == list(delays[:3])
    assert array.hydrophones[3].toa_idx is None


@pytest.mark.parametrize("seed", [7, 42, 20261029])
def test_known_background_rejects_noise_but_accepts_ping(seed):
    array, hydro, expected = gaussian_capture()
    ping = hydro.voltages.copy()
    noise = np.random.default_rng(seed).normal(0, 0.01, len(ping))
    hydro.voltages = noise
    array.estimate_toa_by_envelope(hydro, noise_samples=1500)
    assert not hydro.found_peak
    hydro.voltages = ping + noise
    array.estimate_toa_by_envelope(hydro, noise_samples=1500)
    assert hydro.found_peak
    assert hydro.toa_time == pytest.approx(expected, abs=20 * array.dt)


def test_arrival_reports_start_of_sustained_run(monkeypatch):
    array = HydrophoneArray(sampling_freq=1000, search_band_min=100, search_band_max=400)
    hydro = array.hydrophones[0]
    hydro.times = np.arange(128) / 1000
    hydro.voltages = np.ones(128)
    envelope = np.zeros(128)
    envelope[10:15] = 1  # Less than the required ten samples.
    envelope[40:60] = 1
    monkeypatch.setattr(array, "apply_bandpass_filter", lambda *args, **kwargs: envelope)
    monkeypatch.setattr("Scripts.Hydrophone_Array.hilbert", lambda signal: signal)
    array.estimate_toa_by_envelope(hydro)
    assert hydro.toa_idx == 40


def test_capture_shorter_than_required_hold_has_no_arrival(monkeypatch):
    array = HydrophoneArray(sampling_freq=1000, search_band_min=1, search_band_max=100)
    hydro = array.hydrophones[0]
    hydro.times = np.arange(64) / 1000
    hydro.voltages = np.ones(64)
    monkeypatch.setattr(array, "apply_bandpass_filter", lambda *args, **kwargs: np.ones(64))
    monkeypatch.setattr("Scripts.Hydrophone_Array.hilbert", lambda signal: signal)
    array.estimate_toa_by_envelope(hydro)
    assert not hydro.found_peak


@pytest.mark.parametrize("noise_samples", [-1, 0, 63, 15625, 16000, 100.5])
def test_invalid_noise_window_is_rejected(noise_samples):
    array, hydro, _ = gaussian_capture()
    with pytest.raises(ValueError, match="noise_samples"):
        array.estimate_toa_by_envelope(hydro, noise_samples=noise_samples)


@pytest.mark.parametrize("minimum", [-1, np.nan, np.inf, 301])
def test_invalid_noise_ratio_is_rejected(minimum):
    array, hydro, _ = gaussian_capture()
    with pytest.raises(ValueError, match="min_snr_db"):
        array.estimate_toa_by_envelope(hydro, noise_samples=1500, min_snr_db=minimum)


def test_noise_gate_is_forwarded_for_selected_channels():
    array, hydro, _ = gaussian_capture()
    ping = hydro.voltages.copy()
    for i, h in enumerate(array.hydrophones):
        h.times = hydro.times.copy()
        h.voltages = np.random.default_rng(i).normal(0, 0.01, len(ping))
    array.hydrophones[1].voltages += ping
    array.estimate_selected_by_envelope([True, True, False, False], noise_samples=1500)
    assert not array.hydrophones[0].found_peak
    assert array.hydrophones[1].found_peak
    assert array.hydrophones[2].envelope is None


def test_dc_offset_does_not_move_arrival_or_modify_raw_data():
    array, hydro, expected = gaussian_capture()
    hydro.voltages += 100
    original = hydro.voltages.copy()
    array.estimate_toa_by_envelope(hydro)
    assert hydro.toa_time == pytest.approx(expected, abs=2 * array.dt)
    np.testing.assert_array_equal(hydro.voltages, original)
