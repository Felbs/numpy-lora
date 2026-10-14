# lora-sdr — a LoRa decoder written from scratch, checked against a real transmitter

`lorasdr.py` turns an IQ recording from an SDR (RSPdx, 1 MS/s, int16 interleaved) into LoRa
frames: payload bytes, coding rate, CRC check. Plain numpy/scipy, ~400 lines.

Every convention in it was **measured on air**, not copied: radiobox's SX1302 card was told to
send known payloads (all zeros, single set bits, "HELLO SDR…"), and the decoder was adjusted
until it reproduced them. First CRC-verified decode: 2026-10-13, 923.3 MHz, SF7 BW125.

```
lorasdr.py decode CAP.cs16 --fs 1e6 --offset-khz -200 --sf 7            # channel is 200 kHz below the capture centre
lorasdr.py scan   CAP.cs16 ...                                           # just list frames + raw symbols
```

## Pipeline
1. shift the channel to 0 Hz, decimate to 1 sample per chip
2. find a run of identical up-chirps (preamble), align with the up/down-chirp pair
3. fine sync on the 1 MS/s stream: fractional CFO from the phase advance between preamble
   chirps, fractional timing from the fine preamble bin; re-cut, rotate
4. read the two sync symbols (0x34 → 24, 32) and use them as the absolute bin reference
5. FFT-demodulate every symbol; value = (bin − 1) mod N (the transmitter adds 1)
6. gray-code (v ^ v>>1), diagonal de-interleave, Hamming decode, explicit header (+5-bit
   checksum), payload nibbles **low nibble first**, LFSR de-whitening, CRC16 (0x1021, init 0,
   last two bytes xored in, sent low byte first)

## Verified
- SF7 and SF10, CR 4/5 and 4/8, 23-byte and 230-byte payloads, CRC OK
- whitening sequence recovered from zero payloads = LFSR 0xFF, new bit b7^b5^b4^b3

## Known gaps
- occasional single-symbol errors on strong signals (32 of 36 frames CRC-clean tonight); the symbol decision
  still has a quarter-bin bias for odd symbol values — needs decision-directed timing refinement
- implicit header, LDRO (SF11/12) and sync words other than 0x34 untested
- no streaming / live mode yet; it decodes recordings
