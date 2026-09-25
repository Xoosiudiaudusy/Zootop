"""Windows power throttling (EcoQoS) opt-out and a busy-core meter for long runs.

Found 2026-09-24 by the C++ agent: during a 16-thread training started from the desktop app,
13 of 16 threads were runnable but not running while 75% of the CPU was idle (about 4 busy
cores); opting the process out of power throttling restored 15+ busy cores.  Windows may treat
such processes as background work and schedule them on efficiency cores at low speed.

``disable_power_throttling()`` sets PROCESS_POWER_THROTTLING_EXECUTION_SPEED off for the current
process (SetProcessInformation / ProcessPowerThrottling); threads started later, including the
C++ trainer's std::threads, belong to the process.  It changes scheduling only, never a result.
``CoreMeter`` reports how many cores the process kept busy between two calls (CPU time of all
threads / wall time), so a throttled run shows up in the log as ~4 instead of ~15.
"""
from __future__ import annotations

import os
import time

_PROCESS_POWER_THROTTLING = 4             # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1


def disable_power_throttling() -> bool:
    """Opt the current process out of Windows power throttling.  True if applied; False on
    other systems or if the call is not available (then nothing changes)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        # ControlMask selects the policy we manage; StateMask 0 = that throttling is OFF
        state = _State(_PROCESS_POWER_THROTTLING_CURRENT_VERSION, _PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
        return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                                   ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


class CoreMeter:
    """Busy cores of this process since the previous ``lap()`` (CPU seconds of all threads / wall seconds)."""

    def __init__(self) -> None:
        self._cpu = time.process_time()
        self._wall = time.perf_counter()

    def lap(self) -> float:
        cpu, wall = time.process_time(), time.perf_counter()
        busy = (cpu - self._cpu) / max(wall - self._wall, 1e-9)
        self._cpu, self._wall = cpu, wall
        return busy
