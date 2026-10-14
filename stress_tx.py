# Runs on the Heltec: send n numbered frames with a given LoRa configuration.
import time
import heltec

def burst(freq_khz, sf, bw, sync, dbm, n, gap_ms, tag, size=0):
    m = heltec.radio(freq_khz=freq_khz, sf=sf, bw=bw, syncword=sync, dbm=dbm)
    for i in range(1, n + 1):
        msg = b"%s #%d" % (tag, i)
        if size:
            msg = (msg + b"." * size)[:size]
        heltec.LED.on(); m.send(msg); heltec.LED.off()
        time.sleep_ms(gap_ms)
    m.sleep()
    print("DONE", n)
