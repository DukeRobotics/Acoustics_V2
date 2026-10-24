import Scripts.Hydrophone as Hydrophone
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import numpy as np
from Scripts.Hydrophone_Array import HydrophoneArray

class HydrophoneCalibrator:
    """Calibrates signals of hydrophones in array to a reference hydrophone"""
    hydroarray: Optional[HydrophoneArray] = None

    def __init__(self, arra):
        self.hydroarray = arra

    def read_vals(self, hydro: Hydrophone.Hydrophone):
        #Calculates reference TOA, frequency and amplitude to use for calibrating other hydrophones.
        #literally tells you what the hydrophone thinks it read

        #Args:
        #    hydro: hydrophone to use as reference
        #    assumes the hydrophone has data loaded

        self.hydroarray.estimate_toa_by_envelope(hydro)
        ref_toa = hydro.toa_idx

        #if(hydro.found_peak):
        #    ref_freq = peak_freq
        #else:
        fft_output = fft(hydro.voltages)
        dt = hydro.times[1]-hydro.times[0] # or use the hydroarray samplefreq?
        freqs = fftfreq(len(hydro.voltages), dt)
        freqs = freqs[:len(freqs)//2] #positives only
        fft_output = fft_output[:len(freqs)//2]
            
        peak_freq_idx = np.argmax(np.abs(fft_output))
        ref_freq = freqs[peak_freq_idx]


        ref_ampl = np.abs(hilbert(hydro.voltages))
        # ref_ampl = np.max(hydro.voltages) - np.min(hydro.voltages)

        return ref_toa, ref_freq, ref_ampl
        #notes to self: TIMES is functionally the keyset for VOLTAGES?
        #should definitely check logic2 to see tf is going on

    def calibrate_single(self, hydro: Hydrophone.Hydrophone, ref_toa, ref_freq, ref_ampl):
        hydtoa, hydfreq, hydampl = self.read_vals(hydro)

        hydro.freq_adjust = ref_freq/hydfreq # calculate freq adj
        hydro.ps_adjust = ref_toa - hydtoa # calculate toa adj
        ref_ampl = ref_ampl[ref_toa:]
        hydampl = hydampl[hydtoa:]
        hydro.ampl_adjust = np.mean(ref_ampl/hydampl)

        print(str(hydro.freq_adjust) + " " + str(hydro.ps_adjust) + " " + str(hydro.ampl_adjust))

    def calibrate_all(self):
        ref_toa, ref_freq, ref_ampl = self.read_vals(self.hydroarray.hydrophones[0])
        print(ref_freq)
        i = 1
        while i < len(self.hydroarray.hydrophones):
            self.calibrate_single(self.hydroarray.hydrophones[i], ref_toa, ref_freq, ref_ampl)
            i += 1


