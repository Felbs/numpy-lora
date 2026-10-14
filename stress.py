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

def run(case, freq_mhz, sf, bw_khz, sync=0x34, dbm=10, n=10, size=0):
    tag = case.encode().hex()[:0] or case
    size_eff = size or len(f"{case} #10")
    at = airtime(sf, bw_khz * 1e3, size_eff)
    gap_ms = int(max(300, at * 1000 * 0.5))
    secs = 6 + n * (at + gap_ms / 1000) + 3
    fs = 2e6 if bw_khz >= 500 else 1e6
    cap = f"{OUT}/{case}.cs16"
    capp = subprocess.Popen([sys.executable, os.path.expanduser("~/Projects/capture_cs16.py"), cap, "--mhz", f"{freq_mhz + 0.2:.4f}",
                             "--fs", f"{fs:.0f}", "--secs", f"{secs:.1f}", "--antenna", "Antenna B", "--ifgr", "40", "--rfgr", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(5)
    bw = {125: "125", 250: "250", 500: "500", 62: "62.5"}[bw_khz]
    tx = subprocess.run([MP, "connect", "/dev/ttyUSB0", "exec",
                         f"import stress_tx; stress_tx.burst({int(freq_mhz * 1000)}, {sf}, '{bw}', {sync}, {dbm}, {n}, {gap_ms}, b'{case}', {size})"],
                        capture_output=True, text=True, timeout=600)
    capout = capp.communicate(timeout=900)[0]
    sent = "DONE" in tx.stdout
    dec = subprocess.run([sys.executable, f"{HERE}/lorasdr.py", "decode", cap, "--fs", f"{fs:.0f}", "--offset-khz", "-200",
                          "--sf", str(sf), "--bw", f"{bw_khz * 1e3:.0f}", "--sync", hex(sync), "--max-frames", str(n + 5)],
                         capture_output=True, text=True, timeout=7200).stdout
    found = len(re.findall(r"^frame ", dec, re.M))
    crc = len(re.findall(r"crc_ok=True", dec))
    texts = set(re.findall(r"text '(" + re.escape(case) + r" #\d+)", dec))
    good = len([t for t in texts])
    over = re.search(r"overflows (\d+)", capout)
    row = f"{case}\t{freq_mhz}\tSF{sf}\t{bw_khz}k\t{hex(sync)}\t{dbm}dBm\tsize{size_eff}\tsent {n if sent else 0}\tfound {found}\tcrc {crc}\tmatch {good}\tovf {over.group(1) if over else '?'}"
    print(row, flush=True)
    with open(f"{OUT}/stress_results.tsv", "a") as f: f.write(time.strftime("%H:%M:%S") + "\t" + row + "\n")
    with open(f"{OUT}/{case}.decode.txt", "w") as f: f.write(dec)
    os.remove(cap) if os.path.getsize(cap) > 300e6 else None

CASES = {
 "freq": [("F9023", 902.3, 7, 125), ("F9099", 909.9, 7, 125), ("F9150", 915.0, 7, 125), ("F9211", 921.1, 7, 125), ("F9277", 927.7, 7, 125)],
 "sf":   [("S8", 915.0, 8, 125), ("S9", 915.0, 9, 125), ("S10", 915.0, 10, 125), ("S11", 915.0, 11, 125), ("S12", 915.0, 12, 125)],
 "bw":   [("B250s7", 915.0, 7, 250), ("B500s7", 915.0, 7, 500), ("B250s11", 915.0, 11, 250)],
 "sync": [("Y12", 915.0, 7, 125, 0x12), ("Y2B", 915.0, 7, 125, 0x2B)],
 "size": [("Z1", 915.0, 7, 125, 0x34, 10, 10, 1), ("Z60", 915.0, 7, 125, 0x34, 10, 10, 60), ("Z255", 915.0, 7, 125, 0x34, 10, 10, 255)],
}
if __name__ == "__main__":
    groups = sys.argv[1:] or list(CASES)
    for g in groups:
        for c in CASES[g]:
            run(*c)
