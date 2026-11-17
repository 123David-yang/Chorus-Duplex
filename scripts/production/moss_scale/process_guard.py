"""Bind coordinator-owned workers to its lifetime on Windows and Linux."""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import uuid


class OwnedProcesses:
    def __enter__(self):
        self.kernel = None
        self.jobs = {}
        self.events = []
        if os.name == "nt":
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class IoCounters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
            kernel.SetInformationJobObject.restype = wintypes.BOOL
            kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            kernel.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel.TerminateJobObject.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle.restype = wintypes.BOOL
            kernel.CreateEventW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
            kernel.CreateEventW.restype = wintypes.HANDLE
            kernel.SetEvent.argtypes = (wintypes.HANDLE,)
            kernel.SetEvent.restype = wintypes.BOOL
            self.kernel, self.Extended = kernel, Extended
        elif not sys.platform.startswith("linux"):
            raise RuntimeError("Automatic worker cleanup is supported on Windows and Linux")
        return self

    def popen(self, args, **kwargs):
        if sys.platform.startswith("linux"):
            # Configure PR_SET_PDEATHSIG in a fresh interpreter before exec,
            # avoiding Python's unsafe preexec_fn in a potentially threaded parent.
            args = [sys.executable, "-m", "scripts.production.moss_scale.process_guard",
                    "--child", str(os.getpid()), *args]
        if self.kernel is None:
            return subprocess.Popen(args, **kwargs)
        job = self.kernel.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = self.Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.kernel.CloseHandle(job)
            raise ctypes.WinError(error)
        event_name = "Local\\ChorusGuard-" + uuid.uuid4().hex
        event = self.kernel.CreateEventW(None, True, False, event_name)
        if not event:
            error = ctypes.get_last_error()
            self.kernel.CloseHandle(job)
            raise ctypes.WinError(error)
        self.events.append(event)
        guarded = [sys.executable, "-m", "scripts.production.moss_scale.process_guard",
                   "--child-windows", str(os.getpid()), event_name, *args]
        try:
            process = subprocess.Popen(guarded, **kwargs)
        except BaseException:
            self.kernel.CloseHandle(job)
            raise
        if (not self.kernel.AssignProcessToJobObject(job, int(process._handle))
                or not self.kernel.SetEvent(event)):
            error = ctypes.get_last_error()
            process.terminate()
            process.wait()
            self.kernel.CloseHandle(job)
            raise ctypes.WinError(error)
        self.jobs[process.pid] = job
        return process

    def kill(self, process):
        if self.kernel is None:
            process.kill()
        elif process.pid in self.jobs:
            if not self.kernel.TerminateJobObject(self.jobs[process.pid], 1):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            raise RuntimeError("Process does not belong to this coordinator")

    def __exit__(self, *args):
        if self.kernel is not None:
            for job in self.jobs.values():
                self.kernel.CloseHandle(job)
            for event in self.events:
                self.kernel.CloseHandle(event)


def _linux_child() -> None:
    if not sys.platform.startswith("linux") or len(sys.argv) < 4 or sys.argv[1] != "--child":
        raise SystemExit("This module is a worker-lifetime helper")
    parent = int(sys.argv[2])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise SystemExit(125)
    os.execvpe(sys.argv[3], sys.argv[3:], os.environ)


def _windows_child() -> None:
    from ctypes import wintypes
    if os.name != "nt" or len(sys.argv) < 5:
        raise SystemExit(125)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenEventW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.OpenEventW.restype = wintypes.HANDLE
    kernel.WaitForMultipleObjects.argtypes = (wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD)
    kernel.WaitForMultipleObjects.restype = wintypes.DWORD
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    parent = kernel.OpenProcess(0x00100000, False, int(sys.argv[2]))
    event = kernel.OpenEventW(0x00100000, False, sys.argv[3])
    if not parent or not event:
        raise SystemExit(125)
    try:
        handles = (wintypes.HANDLE * 2)(parent, event)
        if kernel.WaitForMultipleObjects(2, handles, False, 30000) != 1 or kernel.WaitForSingleObject(parent, 0) != 0x102:
            raise SystemExit(125)
    finally:
        kernel.CloseHandle(parent)
        kernel.CloseHandle(event)
    # Windows execvpe has different quoting/lifetime semantics. Keep the wrapper
    # in the job and wait for its job-inheriting child instead.
    raise SystemExit(subprocess.call(sys.argv[4:]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child-windows":
        _windows_child()
    else:
        _linux_child()
