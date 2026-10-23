import Scripts.Hydrophone
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import numpy as np
from Scripts.Hydrophone_Array import HydrophoneArray
from Scripts.Hydrophone_Calibrator import HydrophoneCalibrator
class TestCalibrator:
    def test_script_opposites(self):
        cal = HydrophoneCalibrator(HydrophoneArray())
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
            i += 1
        cal.hydroarray.hydrophones[0].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        cal.hydroarray.hydrophones[1].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0])
        # expect opposite ps of 2
        cal.calibrate_all()
        assert cal.hydroarray.hydrophones[3].ps_adjust == -1*cal.hydroarray.hydrophones[2].ps_adjust

    def test_ampl_ad_simple(self):
        cal = HydrophoneCalibrator(HydrophoneArray())
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
            i += 1
        cal.hydroarray.hydrophones[0].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        cal.hydroarray.hydrophones[1].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0])
        # expect opposite ps of 2
        cal.calibrate_all()
        assert cal.hydroarray.hydrophones[1].ampl_adjust == 1
    
    def test_freq_adj_simple(self):
        cal = HydrophoneCalibrator(HydrophoneArray())
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
            i += 1
        cal.hydroarray.hydrophones[0].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        cal.hydroarray.hydrophones[1].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0])
        # expect opposite ps of 2
        cal.calibrate_all()
        assert cal.hydroarray.hydrophones[1].freq_adjust == 1
    
    def test_ampl_adj_threefive_shifted(self):
        cal = HydrophoneCalibrator(HydrophoneArray())
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
            i += 1
        cal.hydroarray.hydrophones[0].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        cal.hydroarray.hydrophones[1].voltages = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 7.0, 10.5, 7.0, 3.5, 0.0, -3.5, -7.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 7.0])
        # expect opposite ps of 2
        cal.calibrate_all()
        assert cal.hydroarray.hydrophones[1].ampl_adjust == 1
    
    def test_freq_adj_half(self):
        cal = HydrophoneCalibrator(HydrophoneArray())
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
            i += 1
        cal.hydroarray.hydrophones[0].voltages = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        cal.hydroarray.hydrophones[1].voltages = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -2.5, -2.0])
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 7.0, 10.5, 7.0, 3.5, 0.0, -3.5, -7.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 7.0])
        # expect opposite ps of 2
        cal.calibrate_all()
        assert cal.hydroarray.hydrophones[1].freq_adjust == 2.0
    
    def test_ampl_adj_half_freq(self):
        cal = HydrophoneCalibrator(HydrophoneArray())
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0])
            i += 1
        cal.hydroarray.hydrophones[0].voltages = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        cal.hydroarray.hydrophones[1].voltages = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -2.5, -2.0])
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 7.0, 10.5, 7.0, 3.5, 0.0, -3.5, -7.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 7.0])
        # expect opposite ps of 2
        cal.calibrate_all()
        assert cal.hydroarray.hydrophones[1].ampl_adjust == 1