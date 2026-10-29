"""Report calibration from a four-channel capture; optionally verify fake truth."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from Scripts.Hydrophone_Array import HydrophoneArray
from Scripts.Hydrophone_Calibrator import HydrophoneCalibrator


def inspect_capture(path, low=25000.0, high=40000.0):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20:
        raise ValueError("Capture needs a 20-byte <QId header.")
    samples, channels, period = struct.unpack("<QId", header)
    if channels != 4 or samples < 64 or not np.isfinite(period) or period <= 0:
        raise ValueError("Expected four channels, >=64 samples, and a positive finite sample period.")
    if path.stat().st_size != 20 + samples * channels * 4:
        raise ValueError("Capture payload length does not match its header.")
    if not 0 < low < high < 0.5 / period:
        raise ValueError("Search band must be positive, ordered, and below Nyquist.")

    array = HydrophoneArray(sampling_freq=1 / period, search_band_min=low, search_band_max=high)
    array.load_from_path(str(path))
    HydrophoneCalibrator(array).calibrate_all()
    return {
        "capture": str(path), "sampling_freq_hz": 1 / period,
        "num_samples": samples,
        "channels": [
            {"channel": i, "toa_idx": h.toa_idx, "toa_time_seconds": float(h.toa_time),
             "freq_adjust": h.freq_adjust, "ps_adjust_samples": h.ps_adjust,
             "ps_adjust_seconds": h.ps_adjust * period, "ampl_adjust": h.ampl_adjust}
            for i, h in enumerate(array.hydrophones)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--truth", type=Path, help="Generated synthetic JSON; enables acceptance checks")
    parser.add_argument("--out", type=Path, help="Save the measured report as JSON")
    parser.add_argument("--low", type=float, default=25000.0)
    parser.add_argument("--high", type=float, default=40000.0)
    args = parser.parse_args()
    report = inspect_capture(args.capture, args.low, args.high)
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        np.testing.assert_allclose(report["sampling_freq_hz"], truth["sampling_freq_hz"], rtol=1e-10)
        if report["num_samples"] != truth["num_samples"]:
            raise ValueError("Capture sample count differs from truth")
        for actual, expected, rtol, atol in [
            ("freq_adjust", "expected_freq_adjust", 0.002, 0),
            ("ps_adjust_samples", "expected_ps_adjust_samples", 0, 8),
            ("ampl_adjust", "expected_ampl_adjust", 0.02, 0),
        ]:
            np.testing.assert_allclose([h[actual] for h in report["channels"]], truth[expected],
                                       rtol=rtol, atol=atol, err_msg=actual)
        report["synthetic_truth_check"] = "passed"
    formatted = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(formatted, encoding="utf-8")
    print(formatted, end="")


if __name__ == "__main__":
    main()
