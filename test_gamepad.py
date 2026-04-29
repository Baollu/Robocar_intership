# -*- coding: utf-8 -*-
"""Simule la boucle de control_moteur.py sans VESC — affiche duty et steering."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.Gamepad.Gamepad import Gamepad, available
import time

MAX_DUTY_CYCLE = 0.2
MAX_STEERING   = 0.3
diff_steering  = 0.05
duty_smoothing = 10
duty_colapse   = 0.0001

if not available():
    print('Branche la manette puis relance.')
    sys.exit(1)

gamepad = Gamepad()
gamepad.startBackgroundUpdates()

duty    = 0.0
steering = 0.0

print("ax0=turn_max  ax2=brake  ax5=throttle  ax6=steering  |  duty  steering_norm")
print("-" * 75)

try:
    while True:
        ax0 = gamepad.axis(0)
        ax2 = gamepad.axis(2)
        ax5 = gamepad.axis(5)
        ax6 = gamepad.axis(6)

        if ax5 != -1:
            duty += ((ax5 + 1) / 2) / duty_smoothing
        if ax2 != -1:
            duty -= ((ax2 + 1) / 2) / duty_smoothing

        if duty > MAX_DUTY_CYCLE:
            duty = MAX_DUTY_CYCLE
        if duty < -MAX_DUTY_CYCLE:
            duty = -MAX_DUTY_CYCLE

        if duty > 0:
            duty = max(0, duty - duty_colapse)
        if duty < 0:
            duty = min(0, duty + duty_colapse)

        if abs(ax0) > 0.1:
            steering = ax0 * MAX_STEERING
        elif abs(ax6) > 0.1:
            target = ax6 * MAX_STEERING
            if abs(steering - target) <= diff_steering:
                steering = target
            elif steering < target:
                steering += diff_steering
            elif steering > target:
                steering -= diff_steering
        else:
            if abs(steering) < diff_steering:
                steering = 0
            elif steering > 0:
                steering -= diff_steering
            elif steering < 0:
                steering += diff_steering

        snorm = (steering / MAX_STEERING + 1) / 2
        snorm = max(0, min(1, snorm))

        print(f"ax0={ax0:+.3f}  ax2={ax2:+.3f}  ax5={ax5:+.3f}  ax6={ax6:+.3f}"
              f"  |  duty={duty:+.4f}  steer={snorm:.3f}")

        time.sleep(0.05)

except KeyboardInterrupt:
    pass
finally:
    gamepad.disconnect()
