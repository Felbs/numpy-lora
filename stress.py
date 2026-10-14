#!/usr/bin/env python3
"""Stress test: for each case, record with the RSPdx while the Heltec transmits n known frames,
then decode and count frames found / CRC OK / exact text matches.  Results -> stress_results.tsv"""
import math, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/Documents/lora-sdr/stress")
os.makedirs(OUT, exist_ok=True)
MP = os.path.expanduser("~/.local/bin/mpremote")

def airtime(sf, bw, size, cr=1):
    tsym = (1 << sf) / bw
    de = 1 if tsym > 0.016 else 0
    n = 8 + max(math.ceil((8 * size - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (8 + 4.25) * tsym + n * tsym

def run(case, freq_mhz, sf, bw_khz, sync=0x34, dbm=10, n=10, size=0, extra=None, dec=(), tx_offset_khz=0.0,
        also_radiobox=None):
    """extra: dict for the Heltec driver (coding_rate, preamble_len, invert_iq_tx, crc_en, implicit_header).
    dec: extra decoder args. tx_offset_khz: transmit off-frequency but decode as if on freq_mhz.
    also_radiobox: shell snippet run on the gateway at the same time (collision tests)."""
    size_eff = size or len(f"{case} #10")
    cr = (extra or {}).get("coding_rate", 5) - 4
    at = airtime(sf, bw_khz * 1e3, size_eff, cr)
    gap_ms = int(max(300, at * 1000 * 0.5))
    secs = 6 + n * (at + gap_ms / 1000) + 3
    fs = 2e6 if bw_khz >= 500 else 1e6
    cap = f"{OUT}/{case}.cs16"
    capp = subprocess.Popen([sys.executable, os.path.expanduser("~/Projects/capture_cs16.py"), cap, "--mhz", f"{freq_mhz + 0.2:.4f}",
                             "--fs", f"{fs:.0f}", "--secs", f"{secs:.1f}", "--antenna", "Antenna B", "--ifgr", "40", "--rfgr", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(5)
    rb = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "radiobox", also_radiobox], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if also_radiobox else None
    bw = {125: "125", 250: "250", 500: "500", 62: "62.5"}[bw_khz]
    tx_khz = int(round(freq_mhz * 1000 + tx_offset_khz))
    tx = subprocess.run([MP, "connect", "/dev/ttyUSB0", "exec",
                         f"import stress_tx; stress_tx.burst({tx_khz}, {sf}, '{bw}', {sync}, {dbm}, {n}, {gap_ms}, b'{case}', {size}, {extra!r})"],
                        capture_output=True, text=True, timeout=900)
    if rb: rb.wait(timeout=600)
    capout = capp.communicate(timeout=900)[0]
    sent = "DONE" in tx.stdout
    bw_hz = 62500 if bw_khz == 62 else bw_khz * 1e3
    decout = subprocess.run([sys.executable, f"{HERE}/lorasdr.py", "decode", cap, "--fs", f"{fs:.0f}", "--offset-khz", "-200",
                             "--sf", str(sf), "--bw", f"{bw_hz:.0f}", "--sync", hex(sync), "--max-frames", str(n + 5), *dec],
                            capture_output=True, text=True, timeout=7200).stdout
    found = len(re.findall(r"^frame ", decout, re.M))
    crc = len(re.findall(r"crc_ok=True", decout))
    # a frame counts as correct if its text starts with "<case> #<n>" and (when a CRC was sent) the CRC passed
    good = set()
    for blk in decout.split("\nframe ")[1:]:
        m = re.search(r"text '(" + re.escape(case) + r" #\d+)", blk)
        if m and "crc_ok=False" not in blk: good.add(m.group(1))
    rep = len(re.findall(r"repaired", decout))
    integ = re.search(r"integrity: .*= ([+-]\d+) ppm", capout)
    row = (f"{case}\t{freq_mhz}{'%+gk' % tx_offset_khz if tx_offset_khz else ''}\tSF{sf}\t{bw_khz}k\t{hex(sync)}\t{dbm}dBm\tsize{size_eff}\t{extra or ''}"
           f"\tsent {n if sent else 0}\tfound {found}\tcrc {crc}\tcorrect {len(good)}\trepaired {rep}\tdrop {integ.group(1) if integ else '?'}ppm")
    print(row, flush=True)
    with open(f"{OUT}/stress_results.tsv", "a") as f: f.write(time.strftime("%H:%M:%S") + "\t" + row + "\n")
    with open(f"{OUT}/{case}.decode.txt", "w") as f: f.write(decout)
    if not sent: print("   TX failed:", tx.stdout[-300:], tx.stderr[-300:])


CASES = {
 "freq": [("F9023", 902.3, 7, 125), ("F9099", 909.9, 7, 125), ("F9150", 915.0, 7, 125), ("F9211", 921.1, 7, 125), ("F9277", 927.7, 7, 125)],
 "sf":   [("S8", 915.0, 8, 125), ("S9", 915.0, 9, 125), ("S10", 915.0, 10, 125), ("S11", 915.0, 11, 125), ("S12", 915.0, 12, 125)],
 "bw":   [("B250s7", 915.0, 7, 250), ("B500s7", 915.0, 7, 500), ("B250s11", 915.0, 11, 250)],
 "sync": [("Y12", 915.0, 7, 125, 0x12), ("Y2B", 915.0, 7, 125, 0x2B)],
 "size": [("Z1", 915.0, 7, 125, 0x34, 10, 10, 1), ("Z60", 915.0, 7, 125, 0x34, 10, 10, 60), ("Z255", 915.0, 7, 125, 0x34, 10, 10, 255)],
 # ---- round 2
 "iq":    [dict(case="IQ7", freq_mhz=915.0, sf=7, bw_khz=125, extra={"invert_iq_tx": True}, dec=("--invert-iq",)),
           dict(case="IQ10", freq_mhz=915.0, sf=10, bw_khz=125, extra={"invert_iq_tx": True}, dec=("--invert-iq",))],
 "cr":    [dict(case="CR6", freq_mhz=915.0, sf=7, bw_khz=125, extra={"coding_rate": 6}),
           dict(case="CR7", freq_mhz=915.0, sf=7, bw_khz=125, extra={"coding_rate": 7}),
           dict(case="CR8s9", freq_mhz=915.0, sf=9, bw_khz=125, extra={"coding_rate": 8})],
 "narrow":[dict(case="MESHCORE", freq_mhz=910.525, sf=7, bw_khz=62, sync=0x12),
           dict(case="B62s9", freq_mhz=915.0, sf=9, bw_khz=62)],
 "pre":   [dict(case="PRE16", freq_mhz=915.0, sf=7, bw_khz=125, extra={"preamble_len": 16}),
           dict(case="MESHTASTIC", freq_mhz=906.875, sf=11, bw_khz=250, sync=0x2B, extra={"preamble_len": 16})],
 "cfo":   [dict(case=f"CFO{k:+d}k", freq_mhz=915.0, sf=7, bw_khz=125, tx_offset_khz=k) for k in (5, -10, 20, 30)],
 "hdr":   [dict(case="NOCRC", freq_mhz=915.0, sf=7, bw_khz=125, extra={"crc_en": False}),
           dict(case="IMPL", freq_mhz=915.0, sf=7, bw_khz=125, size=12, extra={"implicit_header": True}, dec=("--implicit", "12,1,1")),
           dict(case="IMPL9", freq_mhz=915.0, sf=9, bw_khz=125, size=12, extra={"implicit_header": True}, dec=("--implicit", "12,1,1"))],
 "collide":[dict(case="COLL7", freq_mhz=923.3, sf=7, bw_khz=125, n=10,
                 also_radiobox="python3 ~/lora/tx_test.py 10 0.35 7 'RADIOBOX'"),
            dict(case="COLL7v10", freq_mhz=923.3, sf=7, bw_khz=125, n=10,
                 also_radiobox="python3 ~/lora/tx_test.py 6 0.3 10 'RBSF10'")],
}
if __name__ == "__main__":
    groups = sys.argv[1:] or list(CASES)
    for g in groups:
        for c in CASES[g]:
            run(**c) if isinstance(c, dict) else run(*c)
