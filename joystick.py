import array
import os
import struct
import threading
import time
from fcntl import ioctl

# These constants were borrowed from linux/input.h
axis_names = {
    0x00: "x",
    0x01: "y",
    0x02: "z",
    0x03: "rx",
    0x04: "ry",
    0x05: "rz",
    0x06: "throttle",
    0x07: "rudder",
    0x08: "wheel",
    0x09: "gas",
    0x0A: "brake",
    0x10: "hat0x",
    0x11: "hat0y",
    0x12: "hat1x",
    0x13: "hat1y",
    0x14: "hat2x",
    0x15: "hat2y",
    0x16: "hat3x",
    0x17: "hat3y",
    0x18: "pressure",
    0x19: "distance",
    0x1A: "tilt_x",
    0x1B: "tilt_y",
    0x1C: "tool_width",
    0x20: "volume",
    0x28: "misc",
}

button_names = {
    0x120: "trigger",
    0x121: "thumb",
    0x122: "thumb2",
    0x123: "top",
    0x124: "top2",
    0x125: "pinkie",
    0x126: "base",
    0x127: "base2",
    0x128: "base3",
    0x129: "base4",
    0x12A: "base5",
    0x12B: "base6",
    0x12F: "dead",
    0x130: "a",
    0x131: "b",
    0x132: "c",
    0x133: "x",
    0x134: "y",
    0x135: "z",
    0x136: "tl",
    0x137: "tr",
    0x138: "tl2",
    0x139: "tr2",
    0x13A: "select",
    0x13B: "start",
    0x13C: "mode",
    0x13D: "thumbl",
    0x13E: "thumbr",
    0x220: "dpad_up",
    0x221: "dpad_down",
    0x222: "dpad_left",
    0x223: "dpad_right",
    # XBox 360 controller uses these codes.
    0x2C0: "dpad_left",
    0x2C1: "dpad_right",
    0x2C2: "dpad_up",
    0x2C3: "dpad_down",
}


class Joystick:
    def __init__(self):
        # We'll store the states here.
        self.axis_states = {}
        self.button_states = {}

        self.closed = False

        self.connect()

    def connect(self):
        # Iterate over the joystick devices.
        print("Available devices:")

        for fn in os.listdir("/dev/input"):
            if fn.startswith("js"):
                print("  /dev/input/%s" % (fn))

        self.axis_map = []
        self.button_map = []

        # Open the joystick device.
        fn = "/dev/input/js0"
        print("Opening %s..." % fn)
        jsdev = self.jsdev = open(fn, "rb")

        # Get the device name.
        # buf = bytearray(63)
        buf = array.array("B", [0] * 64)
        ioctl(jsdev, 0x80006A13 + (0x10000 * len(buf)), buf)  # JSIOCGNAME(len)
        js_name = buf.tobytes().rstrip(b"\x00").decode("utf-8")
        print("Device name: %s" % js_name)

        # Get number of axes and buttons.
        buf = array.array("B", [0])
        ioctl(jsdev, 0x80016A11, buf)  # JSIOCGAXES
        num_axes = buf[0]

        buf = array.array("B", [0])
        ioctl(jsdev, 0x80016A12, buf)  # JSIOCGBUTTONS
        num_buttons = buf[0]

        # Get the axis map.
        buf = array.array("B", [0] * 0x40)
        ioctl(jsdev, 0x80406A32, buf)  # JSIOCGAXMAP

        for axis in buf[:num_axes]:
            axis_name = axis_names.get(axis, "unknown(0x%02x)" % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array("H", [0] * 200)
        ioctl(jsdev, 0x80406A34, buf)  # JSIOCGBTNMAP

        for btn in buf[:num_buttons]:
            btn_name = button_names.get(btn, "unknown(0x%03x)" % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0

        print("%d axes found: %s" % (num_axes, ", ".join(self.axis_map)))
        print("%d buttons found: %s" % (num_buttons, ", ".join(self.button_map)))

    def _run(self, callback=None):
        # Main event loop
        while not self.closed:
            try:
                evbuf = self.jsdev.read(8)
            except Exception as e:
                print("jsdev read error")
                print(e)
                try:
                    self.connect()
                except Exception as e_c:
                    print("connect error")
                    print(e_c)
                time.sleep(1)
                continue

            if evbuf:
                t, value, type, number = struct.unpack("IhBB", evbuf)

                if type & 0x01:
                    button = self.button_map[number]
                    if button:
                        self.button_states[button] = value

                    if callback is not None:
                        callback(button, value)

                if type & 0x02:
                    axis = self.axis_map[number]
                    if axis:
                        fvalue = value / 32767.0
                        self.axis_states[axis] = fvalue

                        if callback is not None:
                            callback(axis, fvalue)

        self.jsdev.close()

    def run(self, callback=None):
        t = threading.Thread(target=self._run, args=(callback,))
        t.start()

    def close(self):
        self.closed = True


if __name__ == "__main__":

    def callback(n, v):
        if n in button_names.values():
            print(n, "Pressed" if v else "Released")

        if n in axis_names.values():
            print(n, v)

    js = Joystick()
    js.run(callback)

    time.sleep(100)
