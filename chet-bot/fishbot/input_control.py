"""Mouse input abstraction for low-latency control.

Primary Windows implementation uses SendInput via ctypes.
Falls back to pynput if SendInput not available.
Dry-run mode suppresses real OS events.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

_is_windows = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")

try:
    if _is_windows:
        import ctypes
        from ctypes import wintypes
except Exception:  # pragma: no cover
    _is_windows = False

try:  # fallback
    from pynput.mouse import Controller, Button
except Exception:  # pragma: no cover
    Controller = None  # type: ignore


class MouseController:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._state_down = False
        self._last_toggle_time = 0.0
        if _is_windows:
            self._init_sendinput()
        elif Controller:
            self._pynput = Controller()
        else:
            self._pynput = None

    def _init_sendinput(self):  # Windows only
        # Define necessary structures for SendInput (mouse events)
        self.SendInput = ctypes.windll.user32.SendInput
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)))
        class INPUT(ctypes.Structure):
            _fields_ = (("type", wintypes.DWORD), ("mi", MOUSEINPUT))
        self.MOUSEINPUT = MOUSEINPUT
        self.INPUT = INPUT

    def _send_windows(self, down: bool):
        if self.dry_run:
            return
        flags = self.MOUSEEVENTF_LEFTDOWN if down else self.MOUSEEVENTF_LEFTUP
        extra = ctypes.pointer(wintypes.ULONG(0))
        mi = self.MOUSEINPUT(0, 0, 0, flags, 0, extra)
        inp = self.INPUT(0, mi)
        self.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_fallback(self, down: bool):
        if self.dry_run or not Controller:
            return
        if down and not self._state_down:
            self._pynput.press(Button.left)  # type: ignore
        elif not down and self._state_down:
            self._pynput.release(Button.left)  # type: ignore

    def set_down(self, down: bool):
        if down == self._state_down:
            return
        self._state_down = down
        try:
            import logging
            log = logging.getLogger("fishbot")
            if self.dry_run:
                log.debug(f"[DRY] Mouse {'DOWN' if down else 'UP'}")
        except Exception:
            pass
        if _is_windows:
            self._send_windows(down)
        else:
            self._send_fallback(down)
        self._last_toggle_time = time.perf_counter()

    def is_down(self) -> bool:
        return self._state_down

    def last_toggle_age(self) -> float:
        return time.perf_counter() - self._last_toggle_time
