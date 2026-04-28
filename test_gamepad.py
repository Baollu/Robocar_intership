# -*- coding: utf-8 -*-
"""Lance ce script sur le Jetson pour voir les valeurs brutes des axes."""
import struct
import glob
import sys

JS_EVENT_FORMAT = 'IhBB'
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS   = 0x02
JS_EVENT_INIT   = 0x80

devices = sorted(glob.glob('/dev/input/js*'))
if not devices:
    print("Aucun /dev/input/js* trouve. Branche la manette et relance.")
    sys.exit(1)

print("Manette trouvee :", devices[0])
print("Appuie sur les axes/boutons pour voir les valeurs (Ctrl+C pour quitter)\n")

with open(devices[0], 'rb') as fd:
    while True:
        raw = fd.read(JS_EVENT_SIZE)
        if not raw:
            break
        ts, value, ev_type, number = struct.unpack(JS_EVENT_FORMAT, raw)
        is_init = bool(ev_type & JS_EVENT_INIT)
        ev_type &= ~JS_EVENT_INIT
        if ev_type == JS_EVENT_AXIS:
            norm = value / 32767.0
            tag = "[INIT]" if is_init else "      "
            print(f"{tag} AXIS {number:2d}  raw={value:7d}  norm={norm:+.4f}")
        elif ev_type == JS_EVENT_BUTTON:
            tag = "[INIT]" if is_init else "      "
            print(f"{tag} BTN  {number:2d}  val={value}")
