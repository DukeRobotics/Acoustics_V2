# Hydrophone calibrator testing

Run commands from the repository root in the project's Python environment.
These procedures measure corrections relative to channel H0. They do not
establish absolute hydrophone sensitivity or apply corrections to raw data.

## Setup and automated checks

The calibration tools need NumPy, SciPy, pandas, and matplotlib; the tests also
need pytest. They do not require Saleae hardware or the neural network packages.

```powershell
python -m pip install numpy scipy pandas matplotlib pytest
python -B -m pytest -p no:cacheprovider tests/test_hydrophone_calibrator.py -v
```

The suite checks identical channels, known gains, arrivals before and after H0,
different carriers, frequencies above one quarter of the sample rate, three
noise seeds, DC offsets, deterministic generation, binary loading, repeated
calibration, the single-channel API, silence, and malformed input. Invalid
captures must raise an error without replacing existing calibration coefficients.
Raw voltage samples must remain unchanged.

## Fake data

The generator creates four channels with a common time base: 781,250 samples/s,
15,625 samples (20 ms), and an 8 ms tone burst starting around 4 ms. Cosine ramps
of 0.2 ms give finite leading/trailing edges; silence surrounds the burst.
The default seed is 20261029. Noise is independent between channels.
The `.json` alongside each `.bin` records generation parameters and expected
corrections calculated directly from those parameters, independently of the
calibrator.

| Channel | Peak volts | Carrier, clean/noisy | Delay from H0, samples | Expected amplitude multiplier | Expected sample correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 0.80 | 31,250 Hz | 0 | 1 | 0 |
| H1 | 0.40 | 31,250 Hz | +80 | 2 | -80 |
| H2 | 1.20 | 31,250 Hz | -60 | 0.666667 | +60 |
| H3 | 0.64 | 31,250 Hz | +125 | 1.25 | -125 |

Generate and check all three profiles:

```powershell
python -B -m Scripts.generate_calibration_data --profile clean --out artifacts/calibration/clean.bin
python -B -m Scripts.check_calibration artifacts/calibration/clean.bin --truth artifacts/calibration/clean.json --out artifacts/calibration/clean_report.json

python -B -m Scripts.generate_calibration_data --profile noisy --out artifacts/calibration/noisy.bin
python -B -m Scripts.check_calibration artifacts/calibration/noisy.bin --truth artifacts/calibration/noisy.json --out artifacts/calibration/noisy_report.json

python -B -m Scripts.generate_calibration_data --profile frequency --out artifacts/calibration/frequency.bin
python -B -m Scripts.check_calibration artifacts/calibration/frequency.bin --truth artifacts/calibration/frequency.json --out artifacts/calibration/frequency_report.json
```

The noisy profile adds 0.01 V RMS Gaussian noise and DC offsets of
`[0.1, -0.05, 0.2, -0.1]` V. The frequency profile uses
`[31250, 30000, 35000, 37500]` Hz, with expected frequency multipliers
`[1, 1.0416667, 0.8928571, 0.8333333]`. Artificial channel frequency differences
exercise software behavior; they do not model normal propagation of one tone.

Success means exit code zero and `"synthetic_truth_check": "passed"` in the
report. A truth mismatch raises an assertion and returns a nonzero exit code.
The command checks frequency within 0.2%, amplitude within 2%, and timing within
8 samples (10.24 microseconds). Unit tests use tighter clean-data limits:
0.5% amplitude and 1 sample for matching carriers, or 4 samples for different
carriers because filtering changes the envelope crossing slightly.

To customize fixtures, import `make_capture` and `write_capture` from
`Scripts.generate_calibration_data`. Amplitudes are positive peak volts; delays
are integer samples. Keep the full burst inside the capture and keep carriers
inside the configured search band. Do not edit measured reports into truth files.

## Real data

### Prepare and capture

1. Record sensor/channel identities, H0 reference, acquisition settings, sample
   rate, voltage units, source frequency, geometry, and any known injected gain
   or timing changes. Keep gain settings fixed throughout a comparison.
2. For electronics checks, supply the same known waveform to all acquisition
   paths. For acoustic checks, use a fixed source and controlled geometry with
   equal source-to-sensor paths, or separately account for known path delays.
   Otherwise the measured timing includes propagation as well as channel delay.
3. Capture all four channels simultaneously, including background before the
   ping, the complete burst, and background afterward. Inspect for flat-topped
   clipping, missing channels, dropouts, interference, and truncated bursts.
4. Collect at least 10 repeated captures with the same setup. Also collect a
   source-off background capture and captures with independently known changes
   in gain or delay, if the setup permits. Keep originals unchanged.

### Required file format

`Scripts.check_calibration` reads the repository's combined binary format:

| Field | Encoding |
| --- | --- |
| Samples per channel | Little-endian uint64 |
| Number of channels | Little-endian uint32; must be 4 |
| Sample period in seconds | Little-endian float64 |
| Payload | Little-endian float32, all H0 samples, then H1, H2, H3 |

The header is 20 bytes, followed by exactly `4 * samples * 4` bytes of payload.
The checker derives the processing sample rate from the header. The array loader
alone does not update its configured sample rate when loading a binary file.
Raw vendor exports need conversion if they do not match this format.

The existing CSV loader populates **only H0**, so it cannot directly run a
four-channel calibration. For a CSV containing `time_seconds,H0,H1,H2,H3`, load
and assign all channels explicitly, for example in a Python session launched
from the repository root:

```python
import numpy as np
import pandas as pd
from Scripts.Hydrophone_Array import HydrophoneArray
from Scripts.Hydrophone_Calibrator import HydrophoneCalibrator

data = pd.read_csv("path/to/capture.csv")  # One header row, five numeric columns.
times = data.iloc[:, 0].to_numpy(dtype=float)
sample_rate = 1.0 / np.median(np.diff(times))
array = HydrophoneArray(sampling_freq=sample_rate,
                        search_band_min=25000, search_band_max=40000)
for i, hydro in enumerate(array.hydrophones):
    hydro.times = times.copy()
    hydro.voltages = data.iloc[:, i + 1].to_numpy(dtype=float)
HydrophoneCalibrator(array).calibrate_all()
for i, hydro in enumerate(array.hydrophones):
    print(i, hydro.freq_adjust, hydro.ps_adjust, hydro.ampl_adjust)
array.plot_selected_envelope()
```

### Run and evaluate

Replace the example paths with actual captures; no real recording is assumed
to be a known calibration reference merely because it is already in the repo.

```powershell
python -B -m Scripts.check_calibration path/to/real_capture.bin --out artifacts/calibration/real_report.json
```

Use `--low` and `--high` to change the search band when needed. The band must
contain the source tone and stay below half the sample rate. Without `--truth`,
successful execution means a report was produced; it does not certify the setup.

Interpret the reported coefficients as follows:

- `freq_adjust = frequency_H0 / frequency_channel`. One common source should
  give values near 1; FFT resolution is approximately sample rate / sample count.
- `ps_adjust_samples = arrival_H0 - arrival_channel`. A later channel has a
  negative correction. Multiply by the sample period to obtain seconds.
- `ampl_adjust` is the median ratio of DC-removed Hilbert envelopes over their
  aligned active burst. H0 always receives identity coefficients `(1, 0, 1)`.

Check the envelope plots as well as the coefficients. The arrival is the
beginning of a sustained crossing above 70% of the filtered envelope peak,
not necessarily the physical onset of the acoustic wave.

Choose hardware acceptance limits before evaluating captures, based on the
system's timing and gain requirements. For an initial bench check, the fake
data's 2% gain and 8-sample timing limits can serve as provisional targets;
they are not measured hardware specifications. Compare known injected changes
against their independently measured values, then summarize the median and
spread across repeated captures. A twofold voltage increase should halve the
amplitude correction; an extra D-sample delay should reduce the sample
correction by D. Reuse coefficients from one capture on a separate capture
to check that they generalize before saving them as operational settings.

### Failure cases and records

Silence, nonfinite values, mismatched array lengths, too few samples, and
timestamp spacing inconsistent with the configured rate must be rejected.
The checker also rejects malformed binary headers/payloads. Do not treat a
source-off detection as a valid calibration: the envelope detector uses a
relative threshold and can detect noise or interference by default. Clipping
and multipath also require capture inspection.

Save raw capture paths, setup details, source/gain/delay truth, software revision,
report JSON, acceptance limits, repeatability results, and pass/fail reasoning.
Synthetic checks do not establish performance under multipath, severe noise,
clock drift, polarity inversion, clipping, or absolute sensitivity. Real hardware
validation remains a separate step.

## Direct TOA analyzer verification

```powershell
python -B -m pytest -p no:cacheprovider tests/test_hydrophone_toa.py -v
```

These tests use analytic Gaussian-envelope tone bursts independently of the
calibrator's fixture generator. The expected crossing is
`center - sigma * sqrt(-2 * log(threshold_factor))`. At the default sample rate,
the tested clean crossings agree within 2 samples (2.56 microseconds), across
three carriers, three thresholds, different amplitudes, and different burst
widths. Tests also cover relative sample delays, channel selection, DC-only
captures, stale detections after invalid input, sustained crossings, and the
optional background gate.

For captures with a **known signal-free prefix**, enable the optional gate:

```python
array.estimate_selected_by_envelope(
    noise_samples=1500,
    min_snr_db=12.0,
)
```

Here, the first 1500 samples must be background only on every selected channel.
The peak and accepted crossing must exceed the RMS background envelope by
12 dB (approximately four times its amplitude), and searching begins after
the prefix. The gate rejects the tested stationary Gaussian source-off noise
captures. It is a peak-envelope/background ratio rather than a calibrated
acoustic SNR measurement or a guarantee against changing interference.
Filtering is performed on the full capture, so leave a margin between the
background window and the burst to reduce filter leakage into the background.
The calibrator and command-line checker retain the default relative threshold;
the optional gate is available through the array's direct TOA APIs.

The 70% crossing measures a repeatable point on the burst's rise, rather than
its physical first arrival. A weak direct path followed by a strong reflection
can cause the later reflection to determine the threshold. Zero-phase filtering,
capture edges, narrow filtering, and whitening can also smear a leading edge.
Clean synthetic crossing accuracy does not establish physical onset accuracy
in multipath recordings.

The implementation follows SciPy's documented
[Hilbert-envelope calculation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html)
and [forward-backward SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
