# Runs on the Heltec: send n numbered frames with a given LoRa configuration.
import time
import heltec

def burst(freq_khz, sf, bw, sync, dbm, n, gap_ms, tag, size=0, extra=None):
    m = heltec.radio(freq_khz=freq_khz, sf=sf, bw=bw, syncword=sync, dbm=dbm)
    if extra:
        extra = dict(extra)
        # the micropython-lib driver has no config keys for these two; it reads the attributes at send time
        if "crc_en" in extra: m._crc_en = bool(extra.pop("crc_en"))
        if "implicit_header" in extra: m._implicit_header = bool(extra.pop("implicit_header"))
        if extra:
            m.standby(); m.configure(extra)  # e.g. coding_rate, preamble_len, invert_iq_tx
    for i in range(1, n + 1):
        msg = b"%s #%d" % (tag, i)
        if size:
            msg = (msg + b"." * size)[:size]
        heltec.LED.on(); m.send(msg); heltec.LED.off()
        time.sleep_ms(gap_ms)
    m.sleep()
    print("DONE", n)
