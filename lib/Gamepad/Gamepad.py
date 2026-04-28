"""
Gamepad library for Linux joystick input via /dev/input/jsX.
Compatible interface with PiBorg Gamepad for Jetson Nano / Raspberry Pi.
"""

import os
import struct
import threading
import time
import glob

# Linux joystick event format: timestamp(4B) + value(2B) + type(1B) + number(1B)
_JS_EVENT_FORMAT = 'IhBB'
_JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FORMAT)

_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS   = 0x02
_JS_EVENT_INIT   = 0x80

_JS_AXIS_MAX = 32767.0
_JS_DEVICES_GLOB = '/dev/input/js*'


def available(joystickNumber=0):
    """Return True if at least one joystick device file exists."""
    devices = sorted(glob.glob(_JS_DEVICES_GLOB))
    return len(devices) > joystickNumber


class Gamepad:
    def __init__(self, joystickNumber=0, deadzone=0.05):
        devices = sorted(glob.glob(_JS_DEVICES_GLOB))
        if not devices:
            raise IOError('Aucun gamepad trouvé dans /dev/input/js*')
        if joystickNumber >= len(devices):
            raise IOError(f'Gamepad {joystickNumber} introuvable (seulement {len(devices)} détecté(s))')

        self._device_path = devices[joystickNumber]
        self._deadzone = deadzone
        self._axes = {}
        self._buttons = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._fd = None

    def startBackgroundUpdates(self, waitForReady=True):
        """Start background thread reading joystick events."""
        self._fd = open(self._device_path, 'rb')
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        if waitForReady:
            # Wait briefly for init events to populate axes/buttons
            time.sleep(0.2)

    def _read_loop(self):
        while self._running:
            try:
                raw = self._fd.read(_JS_EVENT_SIZE)
                if not raw or len(raw) < _JS_EVENT_SIZE:
                    break
                _, value, ev_type, number = struct.unpack(_JS_EVENT_FORMAT, raw)
                is_init = bool(ev_type & _JS_EVENT_INIT)
                ev_type &= ~_JS_EVENT_INIT
                with self._lock:
                    if ev_type == _JS_EVENT_AXIS:
                        # Skip init events: triggers (and other axes) get their
                        # real value on the first genuine user input.
                        # Default of -1.0 in axis() handles the resting trigger state.
                        if is_init:
                            continue
                        norm = value / _JS_AXIS_MAX
                        if abs(norm) < self._deadzone:
                            norm = 0.0
                        self._axes[number] = norm
                    elif ev_type == _JS_EVENT_BUTTON:
                        self._buttons[number] = bool(value)
            except Exception:
                break

    def axis(self, axisIndex):
        """Return normalised axis value [-1.0, 1.0]. Returns -1.0 if not yet read."""
        with self._lock:
            # Triggers (axis 2 & 5) start at -1.0 when unpressed on most drivers
            return self._axes.get(axisIndex, -1.0)

    def button(self, buttonIndex):
        """Return True if button is pressed."""
        with self._lock:
            return self._buttons.get(buttonIndex, False)

    def isConnected(self):
        return self._running and self._fd is not None

    def disconnect(self):
        """Stop background updates and close device."""
        self._running = False
        if self._fd:
            try:
                self._fd.close()
            except Exception:
                pass
            self._fd = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
