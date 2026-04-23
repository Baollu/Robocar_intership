from pynput import keyboard

__all__ = ['keys', 'is_key_pressed']

keys = set()

def on_press(key):
    try:
        keys.add(key.char)
    except AttributeError:
        if key == keyboard.Key.esc:
            keys.add("esc")
        return

def on_release(key):
    try:
        if key.char in keys:
            keys.remove(key.char)
    except AttributeError:
        if key == keyboard.Key.esc:
            keys.remove("esc")
        return

def is_key_pressed(key):
    return key in keys

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
