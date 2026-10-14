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
    """Coarse search: index of the first whole-symbol window that begins a run of min_up
    up-chirps with the same FFT bin. Alignment is fixed later by sync_estimate."""
    N, up, dn = chirps(sf)
    nsym = len(y) // N
    bins = np.empty(nsym, int); mags = np.empty(nsym)
    for i in range(nsym):
        f = dechirp_fft(y[i * N:(i + 1) * N], dn)
        bins[i] = f.argmax(); mags[i] = f.max() / (f.mean() + 1e-12)
    for i in range(nsym - min_up):
        if mags[i] > 8 and np.all(bins[i:i + min_up] == bins[i]):
            return i * N, int(bins[i])
    return None, None


def _peak(y, i, N, ref):
    f = dechirp_fft(y[i:i + N], ref)
    return int(f.argmax()), float(f.max() / (f.mean() + 1e-12))


def sync_estimate(y, start, sf, n_up_max=16):
    """From a coarse preamble window: align to the chirp boundary using the up-chirps and
    the down-chirps (bin_up = cfo + sto, bin_down = cfo - sto), read the two sync symbols,
    and return (payload_start, cfo_bins, sync_syms, n_up, sto)."""
    N, up, dn = chirps(sf)
    def signed(b): return b - N if b > N // 2 else b
    # walk forward while the up-chirp bin is steady (preamble), then expect sync, then down-chirps
    i = start; ups = []
    while len(ups) < n_up_max:
        b, m = _peak(y, i, N, dn)
        if m < 6 or (ups and (b - ups[-1]) % N not in (0, 1, N - 1)): break
        ups.append(b); i += N
    bu = int(np.median(ups))
    # the two windows after the sync pair should be down-chirps: find them by DOWN > UP strength
    j = i
    for k in range(6):
        bd, md = _peak(y, j, N, up); bu2, mu2 = _peak(y, j, N, dn)
        if md > mu2 and md > 6: break
        j += N
    else:
        raise RuntimeError("no down-chirps after preamble")
    bd, _ = _peak(y, j, N, up)
    bd2, _ = _peak(y, j + N, N, up)
    bd = int(np.round(np.mean([signed(bd), signed(bd2)])))
    fu, fd = signed(bu), bd
    cfo = (fu + fd) / 2.0
    sto = (fu - fd) / 2.0                  # chips the windows start AFTER the chirp boundary
    # realign: shift windows by sto so bins read 0 (+cfo); try both signs, keep the one that works
    best = None
    for sgn in (+1, -1):
        s2 = start + int(round(sgn * sto))
        if s2 < 0: continue
        b2, m2 = _peak(y, s2 + N, N, dn)
        d = min(abs(signed(b2) - cfo), abs(signed(b2) - cfo - 1), abs(signed(b2) - cfo + 1))
        if best is None or d < best[0]: best = (d, s2, sgn)
    _, start, sgn = best
    # re-walk from the aligned start to find sync + down-chirps precisely
    i = start; ups = []
    while len(ups) < n_up_max:
        b, m = _peak(y, i, N, dn)
        if m < 6 or (ups and (b - ups[-1]) % N not in (0, 1, N - 1)): break
        ups.append(b); i += N
    base = int(np.round(np.median(ups)))
    sync = [(_peak(y, i + k * N, N, dn)[0] - base) % N for k in range(2)]
    i += 2 * N
    payload_start = i + int(round(2.25 * N))
    return payload_start, cfo, sync, len(ups), sgn * sto, start


def demod_symbols(y, start, sf, nsym, base_f=0.0):
    """Symbol value = FFT bin of the de-chirped window after rotating away the preamble's
    fine bin offset (cfo + timing), so every peak lands on an integer bin."""
    N, up, dn = chirps(sf)
    k = np.arange(N)
    rot = np.exp(-2j * np.pi * base_f * k / N).astype(np.complex64)
    syms = []; conf = []
    for s in range(nsym):
        seg = y[start + s * N: start + (s + 1) * N]
        if len(seg) < N: break
        f = dechirp_fft(seg * rot, dn)
        syms.append(int(f.argmax()) % N); conf.append(float(f.max() / (f.mean() + 1e-12)))
    return np.array(syms), np.array(conf)


def preamble_base(y, start, sf, n_up, cfo_bins=0.0, zoom=16):
    """Fine (fractional) FFT bin of the preamble up-chirps, by zero-padded FFT. This bin
    (= cfo + timing offset) is what every payload symbol is offset by; demod removes it."""
    N, up, dn = chirps(sf)
    acc = np.zeros(N * zoom)
    for i in range(1, max(2, n_up - 1)):
        acc += dechirp_fft(y[start + i * N:start + (i + 1) * N], dn, zoom)
    k = int(acc.argmax())
    # parabolic interpolation around the peak
    l, c, r = acc[(k - 1) % len(acc)], acc[k], acc[(k + 1) % len(acc)]
    frac = 0.5 * (l - r) / (l - 2 * c + r) if (l - 2 * c + r) != 0 else 0.0
    bin_f = (k + frac) / zoom
    return bin_f if bin_f <= N / 2 else bin_f - N


def refine_and_demod(x, fs, offset_hz, bw, sf, astart, n_up, nsym, margin=4, syncword=0x34, extra=0):
    """Fine synchronisation on the 1 MS/s stream: fractional CFO from the phase advance between
    preamble up-chirps, fractional timing from the fine up-chirp bin, then re-cut the frame with
    the timing corrected (1/dec chip resolution), rotate the CFO away and demodulate.
    Returns (symbols, conf, info)."""
    from scipy.signal import resample_poly
    N, up, dn = chirps(sf)
    dec = int(round(fs / bw))
    def cut(shift_samples):
        k0 = astart * dec + shift_samples + extra - margin * N * dec
        k1 = k0 + (n_up + 4 + nsym + 2 * margin) * N * dec
        seg = x[max(k0, 0):k1]
        n = np.arange(len(seg), dtype=np.float64) + max(k0, 0)
        seg = seg * np.exp(-2j * np.pi * offset_hz * n / fs).astype(np.complex64)
        return resample_poly(seg, 1, dec).astype(np.complex64)
    y = cut(0)
    p0 = margin * N                                   # aligned preamble start inside y
    # fractional CFO: phase advance of the dechirped preamble tone between consecutive symbols
    k = np.arange(N)
    pk = []; vals = []
    for i in range(1, n_up - 1):
        F = np.fft.fft(y[p0 + i * N:p0 + (i + 1) * N] * dn)
        b = int(np.abs(F).argmax()); pk.append(b); vals.append(F[b])
    dphi = np.angle(np.sum(np.array(vals[1:]) * np.conj(np.array(vals[:-1]))))
    cfo_frac = dphi / (2 * np.pi)                      # bins, in (-0.5, 0.5)
    fu = preamble_base(y, p0, sf, n_up)                # fine up-chirp bin = cfo + sto
    # integer part of the CFO: the up-chirp bin minus the timing offset must be consistent with
    # cfo_frac; we take the integer cfo that makes sto smallest in magnitude (coarse sync is good to <1 chip)
    cands = [(abs(fu - (m + cfo_frac)), m + cfo_frac) for m in range(-3, 4)]
    cfo = min(cands)[1]
    sto = fu - cfo                                     # chips (fractional)
    best = None
    for sgn in (+1, -1):
        shift = int(round(sgn * sto * dec))
        y2 = cut(shift)
        rot = np.exp(-2j * np.pi * cfo * k / N).astype(np.complex64)
        # quality: how close the two sync symbols sit to integer bins (zoomed)
        ps0 = p0 + n_up * N
        q = 0.0; sync = []
        for j in range(2):
            f = dechirp_fft(y2[ps0 + j * N:ps0 + (j + 1) * N] * rot, dn, 16)
            fb = int(f.argmax()) / 16
            sync.append(fb); q += abs(fb - round(fb))
        if best is None or q < best[0]:
            best = (q, y2, sync, sgn)
    q, y2, sync, sgn = best
    rot = np.exp(-2j * np.pi * cfo * k / N).astype(np.complex64)
    # the sync word is known: use it as the absolute bin reference for the payload symbols
    exp_sync = [((syncword >> 4) & 0xF) * 8, (syncword & 0xF) * 8]
    def wrap(d): return (d + N / 2) % N - N / 2
    ref = float(np.mean([wrap(sync[j] - exp_sync[j]) for j in range(2)]))
    ps = p0 + n_up * N + 2 * N + int(round(2.25 * N))
    syms = []; conf = []; fine = []
    for sidx in range(nsym):
        seg = y2[ps + sidx * N:ps + (sidx + 1) * N]
        if len(seg) < N: break
        f = dechirp_fft(seg * rot, dn, 4)
        fb = int(f.argmax()) / 4
        fine.append(fb); syms.append(int(round(fb - ref)) % N); conf.append(float(f.max() / (f.mean() + 1e-12)))
    info = dict(cfo=cfo, sto=sto * sgn, sync=[round(v, 2) for v in sync], sync_q=round(q, 2), fine=fine, ref=round(ref, 2))
    return np.array(syms), np.array(conf), info


def best_demod(x, fs, offset_hz, bw, sf, astart, n_up, nsym, syncword=0x34, span=None):
    """Decision-directed timing: try timing nudges of whole input samples (1/dec chip) and keep
    the one whose symbols land closest to integer bins (sum of distances over the first symbols)."""
    dec = int(round(fs / bw))
    span = dec if span is None else span
    best = None
    for extra in range(-span, span + 1):
        syms, conf, info = refine_and_demod(x, fs, offset_hz, bw, sf, astart, n_up, nsym, syncword=syncword, extra=extra)
        fine = np.array(info["fine"][:min(len(info["fine"]), 48)]) - info["ref"]
        score = float(np.mean(np.abs(fine - np.round(fine))))
        if best is None or score < best[0]:
            best = (score, syms, conf, dict(info, extra=extra, frac_err=round(score, 3)))
    return best[1], best[2], best[3]


def _fine_peak(seg, ref, N, zoom=16):
    """Fractional dechirp peak in signed bins (-N/2, N/2], parabolic interpolation."""
    f = dechirp_fft(seg, ref, zoom)
    k = int(f.argmax()); M = len(f)
    l, c, r = f[(k - 1) % M], f[k], f[(k + 1) % M]
    den = l - 2 * c + r
    b = (k + (0.5 * (l - r) / den if den != 0 else 0.0)) / zoom
    return (b + N / 2) % N - N / 2, float(c / (f.mean() + 1e-12))


def synchronize(x, fs, offset_hz, bw, sf, coarse, syncword=0x34, max_walk=16, cfo_hint=None):
    """Measure-don't-guess frame sync.
    coarse: index (at fs=bw) of a window inside the preamble.
    1. walk windows until the two known sync symbols appear (relative to the preamble bin)
    2. fine up-chirp bin fu (preamble) and down-chirp bin fd (2 windows after sync): with the
       window starting tau chips early, fu = cfo - tau, fd = cfo + tau
    3. re-cut the 1 MS/s stream shifted by tau, de-rotate cfo; verify the sync symbols land on
       integers; return the aligned stream and where the payload starts in it."""
    from scipy.signal import resample_poly
    N, up, dn = chirps(sf)
    dec = int(round(fs / bw))
    s0, s1 = ((syncword >> 4) & 0xF) * 8, (syncword & 0xF) * 8
    def cut(i0_chips, nsym, shift_samples=0.0):
        k0 = int(round(i0_chips * dec + shift_samples))
        k1 = k0 + nsym * N * dec
        seg = x[max(k0, 0):k1]
        n = np.arange(len(seg), dtype=np.float64) + max(k0, 0)
        seg = seg * np.exp(-2j * np.pi * offset_hz * n / fs).astype(np.complex64)
        return resample_poly(seg, 1, dec).astype(np.complex64)
    # --- 1. structure: preamble bin and sync position
    back = 2 * N
    y = cut(coarse - back, max_walk + 8)
    pb, _ = _fine_peak(y[back:back + N], dn, N)
    i = back; ksync = None
    cands = []
    q = N // 8                                   # eighth-symbol steps: some step is within N/8 of alignment
    for w in range(4 * max_walk):
        o = i + w * q
        if o + 2 * N > len(y): break
        a, ma = _fine_peak(y[o:o + N], dn, N)
        b, mb = _fine_peak(y[o + N:o + 2 * N], dn, N)
        # moving the window by d chips moves every dechirped bin by -d: correct the preamble reference
        ref = pb - (o - back)
        ra, rb = (a - ref) % N, (b - ref) % N
        if min(abs(ra - s0), N - abs(ra - s0)) < 2.5 and min(abs(rb - s1), N - abs(rb - s1)) < 2.5:
            cands.append((ma + mb, o))
        elif cands:
            break                                # past the sync word
    if cands:
        ksync = max(cands)[1]                    # strongest = best aligned
    if ksync is None:
        raise RuntimeError("sync word not found")
    # --- 2. fine bins
    fu = float(np.median([_fine_peak(y[ksync - k * N:ksync - (k - 1) * N], dn, N)[0] for k in (1, 2, 3, 4)]))
    fd = float(np.mean([_fine_peak(y[ksync + k * N:ksync + (k + 1) * N], up, N)[0] for k in (2, 3)]))
    def wrap(v): return (v + N / 2) % N - N / 2
    _dbg = dict(fu=round(fu, 2), fd=round(fd, 2), ksync=ksync - back)
    best = None
    max_cfo = min(N / 8, 20e3 / (bw / N))          # real crystals: well under 20 kHz
    for c_off in (0.0, N / 2):
        cfo = wrap((fu + fd) / 2 + c_off); tau = wrap((fd - fu) / 2 + c_off)
        if abs(cfo) > max_cfo: continue
        for sgn in (+1, -1):
            start_chip = coarse - back + ksync - 6 * N          # 6 preamble symbols of margin
            y2 = cut(start_chip, 6 + 4 + 2 + 1, shift_samples=sgn * tau * dec)
            k = np.arange(len(y2)); y2 = y2 * np.exp(-2j * np.pi * cfo * k / N).astype(np.complex64)
            ks = 6 * N
            e = 0.0; vals = []
            for j, want in ((-2, 0), (-1, 0), (0, s0), (1, s1)):
                v, _ = _fine_peak(y2[ks + j * N:ks + (j + 1) * N], dn, N)
                vals.append(round(v % N, 2)); e += abs(wrap(v - want))
            d, _ = _fine_peak(y2[ks + 2 * N:ks + 3 * N], up, N); e += abs(wrap(d))
            if best is None or e < best[0]:
                best = (e, cfo, sgn * tau, start_chip, vals)
    if best is None and cfo_hint is not None:
        # implausible own measurement: trust the previous frame's cfo, timing from the preamble alone
        cfo = cfo_hint; t0 = wrap(cfo - fu)
        for sgn in (+1, -1):
            start_chip = coarse - back + ksync - 6 * N
            y2 = cut(start_chip, 6 + 4 + 2 + 1, shift_samples=sgn * t0 * dec)
            k = np.arange(len(y2)); y2 = y2 * np.exp(-2j * np.pi * cfo * k / N).astype(np.complex64)
            e = sum(abs(wrap(_fine_peak(y2[6 * N + j * N:6 * N + (j + 1) * N], dn, N)[0] - w))
                    for j, w in ((-2, 0), (-1, 0), (0, s0), (1, s1)))
            if best is None or e < best[0]:
                best = (e, cfo, sgn * t0, start_chip, [])
        _dbg["fallback"] = "cfo_hint"
    if best is None:
        raise RuntimeError(f"no plausible cfo/timing: {_dbg}")
    e, cfo, tau, start_chip, vals = best
    # --- 4. refine on the aligned stream: residual up bin ru = dcfo - dtau, down bin rd = dcfo + dtau
    ks = 6 * N
    def measure(cfo, tau):
        y2 = cut(start_chip, 6 + 4 + 2 + 1, shift_samples=tau * dec)
        k = np.arange(len(y2)); y2 = y2 * np.exp(-2j * np.pi * cfo * k / N).astype(np.complex64)
        ru = float(np.mean([_fine_peak(y2[ks + j * N:ks + (j + 1) * N], dn, N)[0] for j in (-4, -3, -2, -1)]))
        rd = float(np.mean([_fine_peak(y2[ks + j * N:ks + (j + 1) * N], up, N)[0] for j in (2, 3)]))
        sy = [_fine_peak(y2[ks + j * N:ks + (j + 1) * N], dn, N)[0] % N for j in (0, 1)]
        return ru, rd, sy, y2
    for _ in range(3):
        ru, rd, sy, _y = measure(cfo, tau)
        dcfo, dtau = (ru + rd) / 2, (rd - ru) / 2
        cand = [(cfo + dcfo, tau + dtau), (cfo + dcfo, tau - dtau)]
        scored = []
        for c, t in cand:
            ru2, rd2, _, _ = measure(c, t)
            scored.append((abs(ru2) + abs(rd2), c, t))
        scored.sort()
        if scored[0][0] >= abs(ru) + abs(rd):
            break
        _, cfo, tau = scored[0]
    ru, rd, sy, _y = measure(cfo, tau)
    e = abs(ru) + abs(rd)
    info = dict(cfo=cfo, sto=tau, sync=sy, err=round(e, 3), pre=[ru, rd])
    # payload starts 2 sync + 2.25 down-chirps after the first sync window
    pay_chip = start_chip + 6 * N + 2 * N + 2.25 * N
    return cfo, tau, pay_chip, info


def demod_from(x, fs, offset_hz, bw, sf, cfo, tau, pay_chip, nsym):
    from scipy.signal import resample_poly
    N, up, dn = chirps(sf)
    dec = int(round(fs / bw))
    k0 = int(round(pay_chip * dec + tau * dec))  # same convention as synchronize.cut
    seg = x[k0:k0 + nsym * N * dec]
    n = np.arange(len(seg), dtype=np.float64) + k0
    seg = seg * np.exp(-2j * np.pi * offset_hz * n / fs).astype(np.complex64)
    y = resample_poly(seg, 1, dec).astype(np.complex64)
    k = np.arange(len(y)); y = y * np.exp(-2j * np.pi * cfo * k / N).astype(np.complex64)
    syms = []; conf = []
    for i in range(nsym):
        w = y[i * N:(i + 1) * N]
        if len(w) < N: break
        v, m = _fine_peak(w, dn, N, 4)
        syms.append(int(round(v)) % N); conf.append(m)
    return np.array(syms), np.array(conf)


# ------------------------------------------------------------------ bit level
def gray_decode(v):            # received symbol value -> interleaved bits
    return v ^ (v >> 1)


def deinterleave(symbols, sf_app, cw_len):
    """LoRa diagonal de-interleaver (gr-lora_sdr convention): returns sf_app codewords of
    cw_len bits, with the bit order already reversed so that the low 4 bits are the data nibble
    and the bits above are the parity bits in the order p0 p1 p2 p3 (p0 = lowest)."""
    inter = [[(symbols[i] >> (sf_app - 1 - j)) & 1 for j in range(sf_app)] for i in range(cw_len)]
    de = [[0] * cw_len for _ in range(sf_app)]
    for i in range(cw_len):
        for j in range(sf_app):
            de[(i - j - 1) % sf_app][i] = inter[i][j]
    # row bits were read MSB-first; the codeword is that integer bit-reversed
    out = []
    for row in de:
        v = 0
        for i, bit in enumerate(row):
            v |= bit << i
        out.append(v)
    return out


def hamming_decode(cw, cw_len):
    """cw: data nibble in bits 0-3 (bit3 = d0 = nibble MSB), parity above. Returns (nibble, corrected).
    Parity bits (verified on air): q0 = d1^d2^d3, q1 = d0^d1^d2, q2 = d0^d2^d3, q3 = d0^d1^d3."""
    n = cw & 0xF
    d0, d1, d2, d3 = (n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1
    if cw_len < 8:
        return n, False
    q = [(cw >> 4) & 1, (cw >> 5) & 1, (cw >> 6) & 1, (cw >> 7) & 1]
    calc = [d1 ^ d2 ^ d3, d0 ^ d1 ^ d2, d0 ^ d2 ^ d3, d0 ^ d1 ^ d3]
    bad = frozenset(i for i in range(4) if q[i] != calc[i])
    if not bad:
        return n, False
    member = {0: frozenset({1, 2, 3}), 1: frozenset({0, 1, 3}), 2: frozenset({0, 1, 2}), 3: frozenset({0, 2, 3})}
    for bit, sets in member.items():
        if bad == sets:
            n ^= 1 << (3 - bit)
            return n, True
    return n, False            # parity-bit error or uncorrectable: leave data alone


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
def decode_frame(syms, sf, dewhiten=True, ldro=False, verbose=True, header_only=False):
    """syms: demodulated symbol values starting at the first header symbol."""
    N = 1 << sf
    # --- header block: 8 symbols at CR4/8, sf-2 bits per symbol ("reduced rate")
    sf_app = sf - 2
    hdr_syms = [gray_decode(((s - 1) % N) >> 2) for s in syms[:8]]     # TX adds 1 to every symbol
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
    if header_only:
        return dict(length=length, cr=cr, has_crc=has_crc)
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
        vals = [gray_decode(((s - 1) % N) >> 2) if ldro else gray_decode((s - 1) % N) for s in blk]
        for cw in deinterleave(vals, sf_pl, cw_len):
            nib, corr = hamming_decode(cw, cw_len); corrected += corr
            payload_nibbles.append(nib)
        pos += cw_len
    # payload bytes are sent low nibble first (verified with single-bit test frames, 10/13)
    data = bytes(payload_nibbles[2 * i] | (payload_nibbles[2 * i + 1] << 4) for i in range(length + (2 if has_crc else 0)))
    payload, crc_rx = data[:length], data[length:length + 2]
    if dewhiten:
        payload = bytes(b ^ w for b, w in zip(payload, whitening(len(payload))))
    crc_ok = None
    if has_crc and len(crc_rx) == 2:
        crc_ok = lora_crc(payload) == ((crc_rx[1] << 8) | crc_rx[0])      # sent low byte first
    return dict(length=length, cr=cr, has_crc=has_crc, payload=payload, crc_rx=crc_rx.hex(),
                crc_ok=crc_ok, symbols_used=pos, corrected=corrected)


# ------------------------------------------------------------------ whitening
def whitening(n):
    """LoRa payload whitening sequence: 8-bit LFSR, state starts at 0xFF, new bit = b7^b5^b4^b3.
    Recovered on air from all-zero payloads sent by radiobox (10/13) and matches the known table."""
    out = bytearray(); st = 0xFF
    for _ in range(n):
        out.append(st)
        nb = ((st >> 7) ^ (st >> 5) ^ (st >> 4) ^ (st >> 3)) & 1
        st = ((st << 1) | nb) & 0xFF
    return bytes(out)


# ------------------------------------------------------------------ encoder (reference symbols)
def _gray_inv(v):
    g = 0
    while v:
        g ^= v; v >>= 1
    return g


def _hdr_checksum(n0, n1, n2):
    c4 = (n0 >> 3 & 1) ^ (n0 >> 2 & 1) ^ (n0 >> 1 & 1) ^ (n0 & 1)
    c3 = (n0 >> 3 & 1) ^ (n1 >> 3 & 1) ^ (n1 >> 2 & 1) ^ (n1 >> 1 & 1) ^ (n2 & 1)
    c2 = (n0 >> 2 & 1) ^ (n1 >> 3 & 1) ^ (n1 & 1) ^ (n2 >> 3 & 1) ^ (n2 >> 1 & 1)
    c1 = (n0 >> 1 & 1) ^ (n1 >> 2 & 1) ^ (n1 & 1) ^ (n2 >> 2 & 1) ^ (n2 >> 1 & 1) ^ (n2 & 1)
    c0 = (n0 & 1) ^ (n1 >> 1 & 1) ^ (n2 >> 3 & 1) ^ (n2 >> 2 & 1) ^ (n2 >> 1 & 1) ^ (n2 & 1)
    return (c4 << 4) | (c3 << 3) | (c2 << 2) | (c1 << 1) | c0


def _hamming_encode(n, cw_len):
    d0, d1, d2, d3 = (n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1
    if cw_len == 5:
        q = [d0 ^ d1 ^ d2 ^ d3]
    else:
        q = [d1 ^ d2 ^ d3, d0 ^ d1 ^ d2, d0 ^ d2 ^ d3, d0 ^ d1 ^ d3][:cw_len - 4]
    cw = n
    for i, b in enumerate(q):
        cw |= b << (4 + i)
    return cw


def _interleave(cws, sf_app, cw_len):
    """inverse of deinterleave(): sf_app codewords -> cw_len symbol values of sf_app bits"""
    out = []
    for i in range(cw_len):
        v = 0
        for j in range(sf_app):
            v |= ((cws[(i - j - 1) % sf_app] >> i) & 1) << (sf_app - 1 - j)
        out.append(v)
    return out


def encode(payload, sf, cr=1, has_crc=True, ldro=False):
    """payload bytes -> list of transmitted symbol values (explicit header), matching decode_frame."""
    N = 1 << sf
    wp = bytes(b ^ w for b, w in zip(payload, whitening(len(payload))))
    data = bytearray(wp)
    if has_crc:
        c = lora_crc(payload)
        data += bytes([c & 0xFF, c >> 8])
    nib = []
    for b in data:
        nib += [b & 0xF, b >> 4]
    L_ = len(payload)
    n0, n1, n2 = L_ >> 4, L_ & 0xF, (cr << 1) | (1 if has_crc else 0)
    chk = _hdr_checksum(n0, n1, n2)
    hdr = [n0, n1, n2, chk >> 4, chk & 0xF]
    sf_app = sf - 2
    first = hdr + nib[:sf_app - 5]
    rest = nib[sf_app - 5:]
    first += [0] * (sf_app - len(first))
    syms = [((_gray_inv(v) << 2) + 1) % N for v in _interleave([_hamming_encode(n, 8) for n in first], sf_app, 8)]
    cw_len = 4 + cr
    sf_pl = sf - 2 if ldro else sf
    for k in range(0, len(rest), sf_pl):
        blk = rest[k:k + sf_pl]
        blk += [0] * (sf_pl - len(blk))
        for v in _interleave([_hamming_encode(n, cw_len) for n in blk], sf_pl, cw_len):
            syms.append(((_gray_inv(v) << (2 if ldro else 0)) + 1) % N)
    return syms


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
    ap.add_argument("--raw", action="store_true", help="do not de-whiten the payload")
    ap.add_argument("--sync", type=lambda v: int(v, 0), default=0x34)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ldro", type=int, default=None, help="force low-data-rate optimisation 0/1 (default: auto)")
    ap.add_argument("--max-frames", type=int, default=50)
    a = ap.parse_args()

    x = load_cs16(a.cap)
    y = to_baseband(x, a.fs, a.offset_khz * 1e3, a.bw)
    N = 1 << a.sf
    ldro = a.ldro if a.ldro is not None else (N / a.bw > 0.016)
    print(f"{len(x)} samples @ {a.fs:.0f} -> {len(y)} @ {a.bw:.0f} ({len(y) / a.bw:.1f} s), N={N}")
    pos = 0; found = 0; ok = 0; last_cfo = None
    while found < a.max_frames:
        start, b = find_preamble(y[pos:], a.sf)
        if start is None: break
        start += pos
        try:
            cfo, sto, pay, info = synchronize(x, a.fs, a.offset_khz * 1e3, a.bw, a.sf, start + 2 * N, syncword=a.sync, cfo_hint=last_cfo)
        except Exception as e:
            if a.verbose: print(f"  [no sync @ {start / a.bw:.3f} s: {e}]")
            pos = start + 4 * N; continue
        found += 1
        if info["err"] < 0.5: last_cfo = cfo
        syms, conf = demod_from(x, a.fs, a.offset_khz * 1e3, a.bw, a.sf, cfo, sto, pay, 8)
        need = 8 + 4 + 8 * ((2 * (255 + 2)) // (a.sf - (2 if ldro else 0)) + 1)
        hdr = decode_frame(syms, a.sf, ldro=ldro, verbose=False, header_only=True)
        if hdr:
            nib = 2 * hdr["length"] + (4 if hdr["has_crc"] else 0)
            need = 8 + (4 + hdr["cr"]) * (nib // (a.sf - (2 if ldro else 0)) + 2)
        if not hdr:
            need = 8
        syms, conf = demod_from(x, a.fs, a.offset_khz * 1e3, a.bw, a.sf, cfo, sto, pay, need)
        ps = int(pay) + len(syms) * N
        print(f"\nframe {found} @ {start / a.bw:.3f} s: pre {[round(float(v), 2) for v in info['pre']]} sync {[round(float(v), 2) for v in info['sync']]} "
              f"cfo {cfo:+.2f} bins ({cfo * a.bw / N:+.0f} Hz), sto {sto:+.2f} chips, err {info['err']}, conf {conf[:8].mean():.1f}")
        if a.mode == "scan":
            print("  symbols:", syms[:24].tolist())
        else:
            r = decode_frame(syms, a.sf, dewhiten=not a.raw, ldro=ldro)
            if r:
                p = r["payload"]
                txt = p.decode("ascii", "replace")
                print(f"  payload[{r['length']}] cr=4/{4 + r['cr']} crc_rx={r['crc_rx']} crc_ok={r['crc_ok']} "
                      f"corrected={r['corrected']}\n  hex {p.hex()}\n  text {txt!r}")
                if a.expect and a.expect.encode() in p: ok += 1; print("  MATCHES expected text")
                if a.mode == "whiten":
                    print("  payload xor whitening table (all-zero TX should give zeros):", p.hex())
        pos = max(ps, start + 12 * N)
    print(f"\n{found} frames found" + (f", {ok} matched" if a.expect else ""))


if __name__ == "__main__":
    main()
