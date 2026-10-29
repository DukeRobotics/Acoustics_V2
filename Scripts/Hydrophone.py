from typing import Optional
import numpy as np

class Hydrophone:
    # raw data
    times: Optional[np.ndarray] = None
    voltages: Optional[np.ndarray] = None

    # envelope-based ToA results
    toa_idx: Optional[int] = None
    toa_time: Optional[float] = None
    toa_peak: Optional[float] = None
    peak_freq: Optional[float] = None
    filtered_signal: Optional[np.ndarray] = None
    envelope: Optional[np.ndarray] = None
    found_peak: bool = False

    # bandpass filtered signal for GCC-PHAT
    filtered_signal_range: Optional[np.ndarray] = None

    # GCC-PHAT results (relative to a reference hydrophone)
    gcc_tdoa: Optional[float] = None
    gcc_cc: Optional[np.ndarray] = None  # cross-correlation array for plotting

    # calibration constants/coefficients
    freq_adjust: float = 1 # frequency adjust coefficient
    ps_adjust: float = 0 # phase shift constant
    ampl_adjust: float = 1 # amplitude adjust coefficient

    def reset(self):
        self.times = None
        self.voltages = None

        self.toa_idx = None
        self.toa_time = None
        self.toa_peak = None
        self.peak_freq = None
        self.filtered_signal = None
        self.envelope = None
        self.found_peak = False

        self.filtered_signal_range = None

        self.gcc_tdoa = None
        self.gcc_cc = None

        # probably should not reset calibrators