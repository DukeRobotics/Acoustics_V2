"""Deterministic four-channel calibration captures, with independent truth.

Run from the repository root: python -m Scripts.generate_calibration_data --help
"""

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def make_capture(*, amplitudes=(0.8, 0.4, 1.2, 0.64),
                 frequencies=(31250.0,) * 4, delays=(0, 80, -60, 125),
                 noise_std=0.0, dc_offsets=(0.0,) * 4, seed=20261029,
                 sampling_freq=781250.0, num_samples=15625):
    """Return times, channel-major voltages, and expected relative corrections.

    Each channel contains a finite 8 ms tone burst with 0.2 ms cosine edges,
    starting around 4 ms, plus optional independent Gaussian noise and DC.
    Delays are integer samples relative to H0. Amplitudes are peak volts.
    """
    if any(len(values) != 4 for values in (amplitudes, frequencies, delays, dc_offsets)):
        raise ValueError("Exactly four channel values are required.")
    if sampling_freq <= 0 or noise_std < 0:
        raise ValueError("Sampling frequency must be positive and noise nonnegative.")
    if any(a <= 0 for a in amplitudes) or any(not 0 < f < sampling_freq / 2 for f in frequencies):
        raise ValueError("Amplitudes must be positive and tones below Nyquist.")
    if any(int(delay) != delay for delay in delays):
        raise ValueError("Delays must be integer sample offsets.")

    times = np.arange(num_samples, dtype=float) / sampling_freq
    burst_samples = round(0.008 * sampling_freq)
    edge_samples = round(0.0002 * sampling_freq)
    envelope = np.ones(burst_samples)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge_samples))
    envelope[:edge_samples], envelope[-edge_samples:] = ramp, ramp[::-1]
    rng = np.random.default_rng(seed)
    channels = []
    for amplitude, frequency, delay, offset in zip(amplitudes, frequencies, delays, dc_offsets):
        start = round(0.004 * sampling_freq) + int(delay)
        stop = start + burst_samples
        if start < 0 or stop >= num_samples:
            raise ValueError("Capture must contain the entire burst with trailing silence.")
        voltage = rng.normal(0, noise_std, num_samples) + offset
        local_times = np.arange(burst_samples) / sampling_freq
        voltage[start:stop] += amplitude * envelope * np.sin(2 * np.pi * frequency * local_times)
        channels.append(voltage)

    truth = {
        "synthetic": True, "seed": seed, "sampling_freq_hz": sampling_freq,
        "num_samples": num_samples, "noise_std_volts": noise_std,
        "amplitudes_volts": list(amplitudes), "frequencies_hz": list(frequencies),
        "delays_samples": list(delays), "dc_offsets_volts": list(dc_offsets),
        "expected_freq_adjust": [frequencies[0] / f for f in frequencies],
        "expected_ps_adjust_samples": [delays[0] - d for d in delays],
        "expected_ampl_adjust": [amplitudes[0] / a for a in amplitudes],
    }
    return times, np.asarray(channels), truth


def write_capture(path, times, channels, truth):
    """Write the existing array loader's <QId header and channel-major float32."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<QId", channels.shape[1], channels.shape[0], times[1] - times[0]))
        stream.write(np.asarray(channels, dtype="<f4").tobytes())
    path.with_suffix(".json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/calibration/synthetic.bin"))
    parser.add_argument("--profile", choices=("clean", "noisy", "frequency"), default="clean")
    args = parser.parse_args()
    options = {}
    if args.profile == "noisy":
        options = {"noise_std": 0.01, "dc_offsets": (0.1, -0.05, 0.2, -0.1)}
    elif args.profile == "frequency":
        options = {"frequencies": (31250.0, 30000.0, 35000.0, 37500.0)}
    times, channels, truth = make_capture(**options)
    truth["profile"] = args.profile
    write_capture(args.out, times, channels, truth)
    print(f"Wrote synthetic capture {args.out} and truth {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
