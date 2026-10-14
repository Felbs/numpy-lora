# numpy-lora

A LoRa receiver in pure NumPy: raw SDR IQ in, LoRa frames out (payload bytes, coding rate,
CRC check). No GNU Radio, no vendor code: about 600 lines of numpy/scipy.

Every PHY convention in it was **measured on air**, not copied: a LoRa gateway (SX1302) and a
Heltec V3 (SX1262) were made to send known payloads (all zeros, single set bits, known text),
and the decoder was adjusted until it reproduced them, CRC and all. It also contains a
reference **encoder**, so a capture can be checked symbol by symbol.

```
lorasdr.py decode CAP.cs16 --fs 1e6 --offset-khz -200 --sf 7 [--bw 125e3] [--sync 0x34] [--verbose]
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
- SF7 and SF10, coding rates 4/5 and 4/8, payloads from 8 to 230 bytes, CRC OK
- whitening sequence recovered from all-zero payloads, matched byte for byte over 230 bytes
- every symbol of clean frames matches the built-in encoder
- stress test across frequency, SF, bandwidth, sync word and payload size: see `stress.py`
  (results are added as they come in)

## Tools
- `stress.py` — records with an RSPdx (SoapySDR) while a Heltec V3 running MicroPython sends
  known frames (`stress_tx.py` on the board), then decodes and scores every case
- capture helper expected at `~/Projects/capture_cs16.py` (any cs16 recorder works)

## Known gaps
- recordings only (no live/streaming mode yet)
- implicit-header mode untested
- a frame overlapped by another transmission is lost (no collision recovery)

MIT licence.
