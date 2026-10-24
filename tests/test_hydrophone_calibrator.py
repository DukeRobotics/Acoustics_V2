import Scripts.Hydrophone
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import numpy as np
from Scripts.Hydrophone_Array import HydrophoneArray
from Scripts.Hydrophone_Calibrator import HydrophoneCalibrator
class TestCalibrator:
    # def test_script_opposites(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[3].ps_adjust == -1*cal.hydroarray.hydrophones[2].ps_adjust

    # def test_ampl_ad_simple(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[1].ampl_adjust == 1
    
    # def test_freq_adj_simple(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[1].freq_adjust == 1
    
    # def test_ampl_adj_threefive_shifted(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 10.5 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[1].ampl_adjust == 1
    
    # def test_freq_adj_half(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (0.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 10.5 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[1].freq_adjust == 2.0
    
    # def test_ampl_adj_half_freq(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (0.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(np.pi * (np.arange(1100, dtype=float) - (3.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages = np.concatenate(([0.0]*100, cal.hydroarray.hydrophones[1].voltages))
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 10.5 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[1].ampl_adjust == 1
    
    # def test_freq_adj_half_ps(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 10.5 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[1].freq_adjust == 2.0
    
    def test_ps(self):
        cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
        cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
        cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(2* np.pi * (np.arange(1100, dtype=float) - (3.0)) / 12) )
        cal.hydroarray.hydrophones[1].voltages = np.concatenate(([0.0]*100, cal.hydroarray.hydrophones[1].voltages))
        # 1: expect ampl 1, ps 0, freqadj 1
        cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(np.pi/10 * (np.arange(100, dtype=float) - (2.0)) / 12) )
        # 2: expect ampl 1, ps 1, freqadj 1
        cal.hydroarray.hydrophones[3].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
        # expect opposite ps of 2
        i = 0
        while i < 4:
            cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
            cal.hydroarray.hydrophones[i].voltages *= 120
            i += 1
        cal.calibrate_all()
        print(cal.hydroarray.hydrophones[2].voltages)
        assert cal.hydroarray.hydrophones[2].toa_idx == -1
    
    # def test_ps_again(self):
    #     cal = HydrophoneCalibrator(HydrophoneArray(search_band_min=1, search_band_max=400))
    #     cal.hydroarray.hydrophones[0].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     cal.hydroarray.hydrophones[1].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (3.0)) / 12) )
    #     # 1: expect ampl 1, ps 0, freqadj 1
    #     cal.hydroarray.hydrophones[2].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (2.0)) / 12) )
    #     # 2: expect ampl 1, ps 1, freqadj 1
    #     cal.hydroarray.hydrophones[3].voltages =  ( 3.0 * np.sin(2 * np.pi * (np.arange(1200, dtype=float) - (4.0)) / 12) )
    #     # expect opposite ps of 2
    #     i = 0
    #     while i < 4:
    #         cal.hydroarray.hydrophones[i].times = np.arange(1, 1201, dtype=float) / 400.0
    #         cal.hydroarray.hydrophones[i].voltages *= 120
    #         i += 1
    #     cal.calibrate_all()
    #     assert cal.hydroarray.hydrophones[3].toa_idx == -1