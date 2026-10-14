#!/usr/bin/env python3
"""lorasdr — a from-scratch LoRa decoder for SDR IQ captures (numpy only).

  lorasdr.py decode CAP.cs16 --fs 1e6 --offset-khz -200 --sf 7 --bw 125e3 [--sync 0x34] [--expect "TEXT"]
  lorasdr.py whiten CAP.cs16 ...        (same, prints recovered whitening bytes from an all-zero payload)

Pipeline: shift the signal to 0 Hz -> decimate to 1 sample/chip -> find preamble up-chirps ->
use the sync/down-chirps to estimate frequency + timing offsets -> FFT-demodulate symbols ->
gray decode -> de-interleave -> Hamming decode -> explicit header -> de-whiten -> CRC16.
Validated against known transmissions from radiobox's SX1302 (see README).
"""
import argparse, math, sys
import numpy as np

# ------------------------------------------------------------------ signal front end
def load_cs16(path, max_samples=None):
    raw = np.fromfile(path, dtype=np.int16, count=-1 if max_samples is None else 2 * max_samples)
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0


def to_baseband(x, fs, offset_hz, bw):
    """Shift the LoRa channel (at offset_hz from capture centre) to 0 Hz and decimate to fs=bw."""
    n = np.arange(len(x), dtype=np.float64)
    x = x * np.exp(-2j * np.pi * offset_hz * n / fs).astype(np.complex64)
    dec = int(round(fs / bw))
    if abs(fs / bw - dec) > 1e-6:
        raise SystemExit(f"fs/bw must be an integer (got {fs / bw})")
    from scipy.signal import resample_poly
    return resample_poly(x, 1, dec).astype(np.complex64)


# ------------------------------------------------------------------ chirps
def chirps(sf):
    N = 1 << sf
    k = np.arange(N)
    up = np.exp(1j * np.pi * (k * k / N - k)).astype(np.complex64)     # frequency sweeps -bw/2 .. +bw/2
    return N, up, np.conj(up)


def dechirp_fft(seg, ref, zoom=1):
    """|FFT(seg * ref)| with optional zero-padding (zoom) for finer peaks."""
    N = len(ref)
    return np.abs(np.fft.fft(seg * ref, N * zoom))


def find_preamble(y, sf, min_up=6):
    """Return the sample index where the first preamble up-chirp starts, plus its FFT bin."""
    N, up, dn = chirps(sf)
    nsym = len(y) // N
    # coarse: for each whole-symbol window, bin of the strongest up-chirp
    bins = np.empty(nsym, int); mags = np.empty(nsym)
    for i in range(nsym):
        f = dechirp_fft(y[i * N:(i + 1) * N], dn)
        bins[i] = f.argmax(); mags[i] = f.max() / (f.mean() + 1e-12)
    # a preamble = min_up consecutive windows with the same bin and strong peaks
    for i in range(nsym - min_up):
        if mags[i] > 8 and np.all(bins[i:i + min_up] == bins[i]):
            # fine: the bin tells how far into the window the chirp starts (mod N); step back to align
            start = i * N - ((N - bins[i]) % N)
            if start < 0: start += N
            return start, bins[i]
    return None, None


def sync_estimate(y, start, sf, n_up_max=12):
    """Walk the preamble: count up-chirps, read the 2 sync symbols, then the 2.25 down-chirps.
    Returns (payload_start, cfo_bins, sync_syms)."""
    N, up, dn = chirps(sf)
    i = start; n_up = 0; ups = []
    while n_up < n_up_max:
        f = dechirp_fft(y[i:i + N], dn)
        b = f.argmax()
        if f.max() / (f.mean() + 1e-12) < 6 or (ups and (b - ups[-1]) % N not in (0, 1, N - 1)):
            break
        ups.append(b); n_up += 1; i += N
    # two sync symbols (modulated up-chirps)
    sync = [int(dechirp_fft(y[i + k * N:i + (k + 1) * N], dn).argmax()) for k in range(2)]
    i += 2 * N
    # 2.25 down-chirps: dechirp with the up-chirp
    down = [int(dechirp_fft(y[i + k * N:i + (k + 1) * N], up).argmax()) for k in range(2)]
    # up-chirp bin = cfo + sto, down-chirp bin = cfo - sto  (all mod N, signed)
    def signed(b): return b - N if b > N // 2 else b
    fu, fd = signed(int(np.median(ups))), signed(int(np.median(down)))
    cfo = (fu + fd) / 2.0
    sto = (fu - fd) / 2.0
    payload_start = i + int(round(2.25 * N)) - int(round(sto))
    return payload_start, cfo, sync, n_up, sto


def demod_symbols(y, start, sf, nsym, cfo_bins=0.0):
    N, up, dn = chirps(sf)
    # remove residual CFO (in bins -> cycles per symbol)
    k = np.arange(N)
    rot = np.exp(-2j * np.pi * cfo_bins * k / N).astype(np.complex64)
    syms = []; conf = []
    for s in range(nsym):
        seg = y[start + s * N: start + (s + 1) * N]
        if len(seg) < N: break
        f = dechirp_fft(seg * rot, dn)
        syms.append(int(f.argmax())); conf.append(float(f.max() / (f.mean() + 1e-12)))
    return np.array(syms), np.array(conf)


# ------------------------------------------------------------------ bit level
def gray_decode(v):            # LoRa transmits gray-coded symbol values; undo on the way in
    g = v ^ (v >> 1)
    return g


def deinterleave(symbols, sf_app, cw_len):
    """LoRa diagonal de-interleaver. symbols: cw_len symbol values of sf_app bits each.
    Returns sf_app codewords of cw_len bits (as ints)."""
    inter = [[(symbols[i] >> (sf_app - 1 - j)) & 1 for j in range(sf_app)] for i in range(cw_len)]
    de = [[0] * cw_len for _ in range(sf_app)]
    for i in range(cw_len):
        for j in range(sf_app):
            de[(i - j - 1) % sf_app][i] = inter[i][j]
    return [int("".join(map(str, row)), 2) for row in de]


def hamming_decode(cw, cw_len):
    """Return (nibble, corrected) for one LoRa codeword. Data = 4 MSBs, parity bits after.
    cw_len 8 -> Hamming(8,4) corrects 1 bit; 7 -> (7,4) corrects 1 bit; 6/5 -> detect only."""
    bits = [(cw >> (cw_len - 1 - i)) & 1 for i in range(cw_len)]
    d = bits[:4]
    if cw_len >= 7:
        p = bits[4:]
        s0 = d[0] ^ d[1] ^ d[2] ^ p[0]
        s1 = d[1] ^ d[2] ^ d[3] ^ p[1]
        s2 = d[0] ^ d[1] ^ d[3] ^ p[2]
        synd = (s0 << 2) | (s1 << 1) | s2
        # syndrome -> bit position (data bits 0..3, parity 4..6)
        table = {0b111: 1, 0b110: 2, 0b101: 0, 0b011: 3, 0b100: 4, 0b010: 5, 0b001: 6}
        if synd:
            pos = table.get(synd)
            if pos is not None and pos < 4:
                d[pos] ^= 1
            return (d[0] << 3) | (d[1] << 2) | (d[2] << 1) | d[3], True
    return (d[0] << 3) | (d[1] << 2) | (d[2] << 1) | d[3], False


def crc16(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def lora_crc(payload):
    """LoRa payload CRC: CRC16 over all but the last two bytes, then xor the last two in."""
    if len(payload) < 2: return crc16(payload)
    c = crc16(payload[:-2])
    return c ^ payload[-1] ^ (payload[-2] << 8)


# ------------------------------------------------------------------ frame decode
def decode_frame(syms, sf, whitening=None, ldro=False, verbose=True):
    """syms: demodulated symbol values starting at the first header symbol."""
    N = 1 << sf
    # --- header block: 8 symbols at CR4/8, sf-2 bits per symbol ("reduced rate")
    sf_app = sf - 2
    hdr_syms = [gray_decode(((s - 1) % N) >> 2) for s in syms[:8]]
    cws = deinterleave(hdr_syms, sf_app, 8)
    nibbles = [hamming_decode(c, 8)[0] for c in cws]
    length = (nibbles[0] << 4) | nibbles[1]
    cr = nibbles[2] >> 1
    has_crc = nibbles[2] & 1
    hdr_chk = ((nibbles[3] & 1) << 4) | nibbles[4]
    n0, n1, n2 = nibbles[0], nibbles[1], nibbles[2]
    # header checksum (Semtech): 5 bits over the first 3 nibbles
    c4 = (n0 >> 3 & 1) ^ (n0 >> 2 & 1) ^ (n0 >> 1 & 1) ^ (n0 & 1)
    c3 = (n0 >> 3 & 1) ^ (n1 >> 3 & 1) ^ (n1 >> 2 & 1) ^ (n1 >> 1 & 1) ^ (n2 & 1)
    c2 = (n0 >> 2 & 1) ^ (n1 >> 3 & 1) ^ (n1 & 1) ^ (n2 >> 3 & 1) ^ (n2 >> 1 & 1)
    c1 = (n0 >> 1 & 1) ^ (n1 >> 2 & 1) ^ (n1 & 1) ^ (n2 >> 2 & 1) ^ (n2 >> 1 & 1) ^ (n2 & 1)
    c0 = (n0 & 1) ^ (n1 >> 1 & 1) ^ (n2 >> 3 & 1) ^ (n2 >> 2 & 1) ^ (n2 >> 1 & 1) ^ (n2 & 1)
    chk = (c4 << 4) | (c3 << 3) | (c2 << 2) | (c1 << 1) | c0
    hdr_ok = chk == hdr_chk
    if verbose:
        print(f"header: len={length} cr=4/{4 + cr} crc={'yes' if has_crc else 'no'} "
              f"chk={hdr_chk:#04x} calc={chk:#04x} {'OK' if hdr_ok else 'MISMATCH'}")
    if not hdr_ok or cr < 1 or cr > 4:
        return None
    payload_nibbles = list(nibbles[5:])           # leftover nibbles of the header block (SF>7)
    need = 2 * length + (4 if has_crc else 0)
    # --- payload blocks
    cw_len = 4 + cr
    sf_pl = sf - 2 if ldro else sf
    pos = 8; corrected = 0
    while len(payload_nibbles) < need:
        blk = syms[pos:pos + cw_len]
        if len(blk) < cw_len:
            if verbose: print(f"ran out of symbols: have {len(payload_nibbles)}/{need} nibbles")
            return None
        vals = [gray_decode((s - 1) % N) for s in blk]
        if ldro: vals = [v >> 2 for v in vals]
        for cw in deinterleave(vals, sf_pl, cw_len):
            nib, corr = hamming_decode(cw, cw_len); corrected += corr
            payload_nibbles.append(nib)
        pos += cw_len
    data = bytes((payload_nibbles[2 * i] << 4) | payload_nibbles[2 * i + 1] for i in range(length + (2 if has_crc else 0)))
    payload, crc_rx = data[:length], data[length:length + 2]
    if whitening is not None:
        payload = bytes(b ^ w for b, w in zip(payload, whitening))
    crc_ok = None
    if has_crc and len(crc_rx) == 2:
        crc_ok = lora_crc(payload) == ((crc_rx[1] << 8) | crc_rx[0]) or lora_crc(payload) == ((crc_rx[0] << 8) | crc_rx[1])
    return dict(length=length, cr=cr, has_crc=has_crc, payload=payload, crc_rx=crc_rx.hex(),
                crc_ok=crc_ok, symbols_used=pos, corrected=corrected)


# ------------------------------------------------------------------ whitening table (recovered from radiobox)
def load_whitening(path):
    try:
        return bytes.fromhex(open(path).read().split())
    except Exception:
        return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["decode", "whiten", "scan"])
    ap.add_argument("cap")
    ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--offset-khz", type=float, default=0.0, help="LoRa channel minus capture centre")
    ap.add_argument("--sf", type=int, default=7)
    ap.add_argument("--bw", type=float, default=125e3)
    ap.add_argument("--expect", default=None)
    ap.add_argument("--whitening", default=None, help="hex file with the whitening sequence")
    ap.add_argument("--max-frames", type=int, default=50)
    a = ap.parse_args()

    x = load_cs16(a.cap)
    y = to_baseband(x, a.fs, a.offset_khz * 1e3, a.bw)
    N = 1 << a.sf
    print(f"{len(x)} samples @ {a.fs:.0f} -> {len(y)} @ {a.bw:.0f} ({len(y) / a.bw:.1f} s), N={N}")
    wh = load_whitening(a.whitening) if a.whitening else None
    pos = 0; found = 0; ok = 0
    while found < a.max_frames:
        start, b = find_preamble(y[pos:], a.sf)
        if start is None: break
        start += pos
        try:
            ps, cfo, sync, n_up, sto = sync_estimate(y, start, a.sf)
        except Exception as e:
            print("sync failed:", e); pos = start + N; continue
        found += 1
        syms, conf = demod_symbols(y, ps, a.sf, 8 + 64, cfo)
        print(f"\nframe {found} @ {start / a.bw:.3f} s: {n_up} up-chirps, sync {sync}, "
              f"cfo {cfo:+.2f} bins ({cfo * a.bw / N:+.0f} Hz), sto {sto:+.1f}, conf {conf[:8].mean():.1f}")
        if a.mode == "scan":
            print("  symbols:", syms[:24].tolist())
        else:
            r = decode_frame(syms, a.sf, whitening=wh)
            if r:
                p = r["payload"]
                txt = p.decode("ascii", "replace")
                print(f"  payload[{r['length']}] cr=4/{4 + r['cr']} crc_rx={r['crc_rx']} crc_ok={r['crc_ok']} "
                      f"corrected={r['corrected']}\n  hex {p.hex()}\n  text {txt!r}")
                if a.expect and a.expect.encode() in p: ok += 1; print("  MATCHES expected text")
                if a.mode == "whiten":
                    print("  whitening (payload xor zeros):", p.hex())
        pos = ps + (8 + 10) * N
    print(f"\n{found} frames found" + (f", {ok} matched" if a.expect else ""))


if __name__ == "__main__":
    main()
