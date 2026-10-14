# numpy-lora

A LoRa receiver in pure NumPy: raw SDR IQ in, LoRa frames out (payload bytes, coding rate,
CRC check). No GNU Radio, no vendor code: about 600 lines of numpy/scipy.

Every PHY convention in it was **measured on air**, not copied: a LoRa gateway (SX1302) and a
Heltec V3 (SX1262) were made to send known payloads (all zeros, single set bits, known text),
and the decoder was adjusted until it reproduced them, CRC and all. It also contains a
reference **encoder**, so a capture can be checked symbol by symbol.

```
lorasdr.py decode CAP.cs16 --fs 1e6 --offset-khz -200 --sf 7 [--bw 125e3] [--sync 0x34] [--invert-iq]
                  [--implicit LEN,CR,CRC] [--ldro 0|1] [--verbose]
lorasdr.py scan   CAP.cs16 ...      # list frames and raw symbols only
```
`CAP.cs16` is interleaved int16 I/Q; `--offset-khz` is the LoRa channel minus the capture centre
(tune a little off-channel to keep the SDR's DC spike out of the signal).

## Pipeline
1. shift the channel to 0 Hz, decimate to one sample per chip
2. coarse detection: a run of identical de-chirped up-chirp bins (the preamble)
3. sync: find the two known sync-word symbols (searched in 1/8-symbol steps), then measure
   the frequency and timing offsets from the up-chirp and down-chirp bins
   (up = cfo − τ, down = cfo + τ), re-cut the full-rate stream, refine twice
4. FFT-demodulate each symbol; value = (bin − 1) mod 2^SF (the transmitter adds 1)
5. gray code (v ^ v>>1), diagonal de-interleave, Hamming decode
6. explicit header (length, coding rate, CRC flag, 5-bit checksum), reduced-rate header block
7. payload nibbles **low nibble first**, LFSR de-whitening (state 0xFF, taps 7,5,4,3),
   CRC16 (poly 0x1021, init 0, last two bytes xored in, sent low byte first)
8. low-data-rate optimisation (SF11/12 at 125 kHz) drops the two LSBs of each payload symbol

## Verified on air
Transmitters: a Heltec V3 (SX1262) and a LoRa gateway (SX1302), 10 known frames per case,
recorded with an RSPdx at 1 MS/s (2 MS/s for 500 kHz). "Correct" = right text and CRC OK.

| Case | Correct |
|---|---|
| frequencies 902.3 / 909.9 / 915.0 / 921.1 / 927.7 MHz, SF7 | 49/50 |
| SF8 / SF9 / SF10 / SF11 / SF12 (BW125; SF11/12 with low-data-rate optimisation) | 9 / 9 / 9 / 9 / 8 |
| BW 62.5 / 250 / 500 kHz | 7 (SF9) / 9 / 10 |
| coding rate 4/5, 4/6, 4/7, 4/8 | all ≥ 9/10 |
| payload 1 / 60 / 255 bytes | 10 / 10 / 9 |
| sync word 0x12 (private), 0x2B (Meshtastic) | 10 / 10 |
| **Meshtastic LongFast** (SF11, BW250, sync 0x2B, 16-symbol preamble) | 9/10 |
| **MeshCore US** (910.525 MHz, SF7, BW62.5, sync 0x12) | 10/10 |
| **inverted IQ** (LoRaWAN downlinks), SF7 / SF10 | 10 / 9 |
| transmitter detuned +5 / −10 / +20 / +30 kHz | 10 / 10 / 9 / 9 |
| implicit header SF7 / SF9, implicit + no CRC, no CRC | 9 / 10 / 9 / 10 |
| collision, two SF7 frames overlapping on one channel | 14 of 20 recovered |

### Sensitivity (noise injected into real recordings)
| | datasheet floor (SX127x/SX126x) | numpy-lora |
|---|---|---|
| SF7 | −7.5 dB | 10/10 at −7 dB, 7/10 at −7.5 to −9 dB |
| SF10 | −15 dB | 9/9 at −16 dB, 7/9 at −17 dB |
| SF12 | −20 dB | 5/5 at −22 dB |

`snr_sweep.py` reproduces this on any clean recording.

### Things measured on air that the documentation doesn't make obvious
- every symbol is transmitted as value + 1; payload bytes go low nibble first
- whitening: 8-bit LFSR, state 0xFF, taps 7,5,4,3; CRC16 0x1021 init 0 with the last two
  payload bytes xored in; a 1-byte payload's CRC is the byte itself
- **sync-word nibbles are signed**: 0x2B sends symbols 16 and −40 (not 88); at SF7 the two
  coincide, at SF11 they don't
- the SDR (RSPdx, SDRplay API 3.15 via SoapySDR) **silently drops samples**, ~200–700 ppm, no
  overflow flag. Long frames straddle a gap: the decoder tracks the slip mid-frame and lets the
  CRC referee small ambiguities ("repaired: slip +1 after symbol 219")

## Tools
- `stress.py` — records with an RSPdx (SoapySDR) while a Heltec V3 running MicroPython sends
  known frames (`stress_tx.py` on the board), then decodes and scores every case
- `snr_sweep.py` — sensitivity by noise injection
- `lorasdr.encode()` — reference encoder (symbols a real chip sends), for symbol-by-symbol checks
- capture helper expected at `~/Projects/capture_cs16.py` (any cs16 recorder works)

## Known gaps
- recordings only (no live/streaming mode yet)
- one spreading factor per run (a mixed-SF collision needs one pass per SF)
- SF12 frames with two or more sample drops can still fail (fix the SDR drops, not the decoder)

MIT licence.
