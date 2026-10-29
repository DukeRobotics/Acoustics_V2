import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert

from Scripts.Hydrophone import Hydrophone
from Scripts.Hydrophone_Array import HydrophoneArray


class HydrophoneCalibrator:
    """Measure relative corrections against H0; raw samples are not modified.

    Frequency and amplitude corrections are multipliers. ps_adjust is a signed
    sample offset (negative means the channel arrived later than H0).
    """

    def __init__(self, arra: HydrophoneArray):
        self.hydroarray = arra

    def read_vals(self, hydro: Hydrophone):
        """Return detected arrival index, in-band FFT peak, and raw envelope."""
        # Also validates shapes, finite samples, and timestamp spacing.
        self.hydroarray.estimate_toa_by_envelope(hydro)
        if not hydro.found_peak:
            raise ValueError("No detectable calibration ping in channel.")

        centered = np.asarray(hydro.voltages, dtype=float) - np.mean(hydro.voltages)
        spectrum = np.abs(rfft(centered))
        freqs = rfftfreq(len(centered), self.hydroarray.dt)
        in_band = ((freqs >= self.hydroarray.search_band_min)
                   & (freqs <= self.hydroarray.search_band_max))
        if not np.any(in_band) or np.max(spectrum[in_band]) <= np.finfo(float).eps:
            raise ValueError("No calibration signal in the frequency search band.")
        frequency = freqs[in_band][np.argmax(spectrum[in_band])]
        envelope = np.abs(hilbert(centered))
        return hydro.toa_idx, float(frequency), envelope

    @staticmethod
    def _corrections(reference, measured):
        ref_toa, ref_freq, ref_ampl = reference
        hydtoa, hydfreq, hydampl = measured
        # Align at detected arrivals and compare only the common active burst.
        # Silent tails must not contribute near-zero envelope divisions.
        ref = ref_ampl[ref_toa:]
        hyd = hydampl[hydtoa:]
        length = min(len(ref), len(hyd))
        ref, hyd = ref[:length], hyd[:length]
        active = ((ref > 0.7 * np.max(ref)) & (hyd > 0.7 * np.max(hyd)))
        if not np.any(active):
            raise ValueError("Calibration channels have no overlapping active burst.")
        return ref_freq / hydfreq, ref_toa - hydtoa, float(np.median(ref[active] / hyd[active]))

    def calibrate_single(self, hydro: Hydrophone, ref_toa, ref_freq, ref_ampl):
        corrections = self._corrections(
            (ref_toa, ref_freq, ref_ampl), self.read_vals(hydro)
        )
        hydro.freq_adjust, hydro.ps_adjust, hydro.ampl_adjust = corrections

    def calibrate_all(self):
        # Measure every channel before replacing any stored coefficients.
        measurements = [self.read_vals(hydro) for hydro in self.hydroarray.hydrophones]
        corrections = [(1.0, 0, 1.0)] + [
            self._corrections(measurements[0], measured)
            for measured in measurements[1:]
        ]
        for hydro, values in zip(self.hydroarray.hydrophones, corrections):
            hydro.freq_adjust, hydro.ps_adjust, hydro.ampl_adjust = values
