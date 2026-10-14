#!/usr/bin/env python3
"""snr_sweep.py CAP.cs16 --sf N [--bw 125e3] [--fs 1e6] [--offset-khz -200] [--secs S] [--snr -4,-6,...]

Sensitivity by noise injection: take a clean recording, measure each frame's in-channel signal
power, add complex white Gaussian noise so the in-channel SNR equals the target, decode, and count
CRC-valid frames. Reports the decode rate per target SNR (compare: Semtech SX127x/SX126x datasheet
demodulator floors SF7 -7.5, SF8 -10, SF9 -12.5, SF10 -15, SF11 -17.5, SF12 -20 dB).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lorasdr as L

SPEC = {7: -7.5, 8: -10, 9: -12.5, 10: -15, 11: -17.5, 12: -20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap"); ap.add_argument("--sf", type=int, required=True)
    ap.add_argument("--bw", type=float, default=125e3); ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--offset-khz", type=float, default=-200.0); ap.add_argument("--secs", type=float, default=None)
    ap.add_argument("--snr", default=None, help="comma list of target SNRs in dB")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    x = L.load_cs16(a.cap)
    if a.secs: x = x[:int(a.secs * a.fs)]
    off = a.offset_khz * 1e3
    clean = list(L.frames(x, a.fs, off, a.bw, a.sf))
    good = [f for f in clean if f["result"] and f["result"]["crc_ok"]]
    if not good:
        print("no clean frames to start from"); return
    # in-channel signal power (noise already present is small: these recordings are +30..+50 dB)
    y = L.to_baseband(x, a.fs, off, a.bw); N = 1 << a.sf
    ps = float(np.median([np.mean(np.abs(y[int(f["pay"]):int(f["pay"]) + len(f["syms"]) * N]) ** 2) for f in good]))
    spec = SPEC.get(a.sf)
    targets = [float(v) for v in a.snr.split(",")] if a.snr else [spec + d for d in (6, 3, 1, 0, -1, -2, -3)]
    print(f"{a.cap}: SF{a.sf} BW{a.bw / 1e3:g}k, {len(good)}/{len(clean)} clean frames CRC-ok, "
          f"clean SNR ~{np.nanmedian([f['snr'] for f in good]):+.1f} dB, datasheet floor {spec} dB")
    rng = np.random.default_rng(a.seed)
    for t in targets:
        pn_chan = ps / 10 ** (t / 10)                 # wanted in-channel noise power
        sigma2 = pn_chan * a.fs / a.bw                # white noise over the full capture bandwidth
        nz = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))).astype(np.complex64) * np.float32(np.sqrt(sigma2 / 2))
        res = list(L.frames(x + nz, a.fs, off, a.bw, a.sf))
        ok = sum(1 for f in res if f["result"] and f["result"]["crc_ok"])
        print(f"  SNR {t:+6.1f} dB ({t - spec:+.1f} vs datasheet): found {len(res):2d}  CRC ok {ok:2d}/{len(good)}", flush=True)


if __name__ == "__main__":
    main()
