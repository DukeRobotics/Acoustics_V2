import Hydrophone
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import numpy as np
import HydrophoneArray

class HydrophoneCalibrator:
    """Calibrates signals of hydrophones in array to a reference hydrophone"""
    hydroarray: Optional[HydrophoneArray] = None

    def __init__(arra):
        self.hydroarray = arra

    def create_ref(self, hydro: Hydrophone.Hydrophone):
        #Calculates reference TOA, frequency and amplitude to use for calibrating other hydrophones.
        #literally tells you what the hydrophone thinks it read

        #Args:
        #    hydro: hydrophone to use as reference
        #    assumes the hydrophone has data loaded

        hydroarray.estimate_toa_by_envelope(hydro)
        ref_toa = hydro.toa_idx

        #if(hydro.found_peak):
        #    ref_freq = peak_freq
        #else:
        fft_output = fft(hydro.voltages)
        dt = hydro.times[1]-hydro.times[0] # or use the hydroarray samplefreq?
        freqs = fftfreq(len(hydro.voltages), dt)
        freqs = freqs[:n//2] #positives only
        fft_output = fft_output[:n//2]
            
        peak_freq_idx = np.argmax(np.abs(fft_output))
        ref_freq = freqs[peak_freq_idx]


        # ref_ampl = np.abs(hilbert(hydro.voltages))
        ref_ampl = np.max(hydro.voltages) - np.min(hydro.voltages)

        return ref_toa, ref_freq, ref_ampl
        #notes to self: TIMES is functionally the keyset for VOLTAGES?
        #should definitely check logic2 to see tf is going on

    def calibrate_single(self, hydro: Hydrophone.Hydrophone, ref_toa, ref_freq, ref_ampl):
        hydtoa, hydfreq, hydampl = create_ref(hydro)

        hydro.freq_adjust = ref_freq/hydfreq # calculate freq adj
        hydro.ps_adjust = ref_toa - hydtoa # calculate toa adj
        hydro.ampl_adjust = ref_ampl/hydampl # ref_ampl has alr been hilberted

        print(hydro.freq_adjust + " " + hydro.ps_adjust + " " + hydro.ampl_adjust)

    def calibrate_all(self):
        ref_toa, ref_freq, ref_ampl = create_ref(hydroarray[0])
        i = 1
        while i < len(hydroarray):
            calibrate_single(hydroarray[i])
