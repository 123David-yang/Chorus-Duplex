import ctypes
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.production.moss_scale.produce_and_upload import _clear_owned_stops, _remaining, ProducerProgress
from scripts.production.moss_scale.storage import atomic_json


class CoordinatorTests(unittest.TestCase):
    def test_producer_watchdog_uses_delivered_items_not_status_heartbeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            progress = ProducerProgress(root, 100)
            self.assertFalse(progress.stalled(root, 150, 60))
            atomic_json(root / "status.json", {"dialogues": 1})
            self.assertFalse(progress.stalled(root, 155, 60))
            atomic_json(root / "status.json", {"dialogues": 1})
            self.assertTrue(progress.stalled(root, 216, 60))
            atomic_json(root / "status.json", {"dialogues": 2})
            self.assertFalse(progress.stalled(root, 217, 60))

    def test_total_deadline_limits_each_blocking_operation(self):
        with patch("scripts.production.moss_scale.produce_and_upload.time.monotonic", return_value=100):
            self.assertEqual(_remaining(130, 120), 30)
            with self.assertRaisesRegex(RuntimeError, "total time limit"):
                _remaining(100, 120)

    def test_two_producers_and_uploader_coordinate_with_stub_processes(self):
        from scripts.production.moss_scale import produce_and_upload as coordinator
        task_dir = Path("D:/DuplexVoice_production") if os.name == "nt" and Path("D:/DuplexVoice_production").is_dir() else None
        with tempfile.TemporaryDirectory(dir=task_dir) as folder:
            base = Path(folder)
            source = base / "fake source"
            module_dir = source / "scripts" / "production" / "moss_scale"
            module_dir.mkdir(parents=True)
            for directory in (source / "scripts", source / "scripts" / "production", module_dir):
                (directory / "__init__.py").touch()
            guard = Path(coordinator.__file__).with_name("process_guard.py")
            shutil.copyfile(guard, module_dir / "process_guard.py")
            (source / "scripts" / "production" / "scale_moss.py").write_text("""
import json,sys
from pathlib import Path
if '--check' in sys.argv: raise SystemExit(0)
root=Path(sys.argv[sys.argv.index('--root')+1]); root.mkdir(parents=True,exist_ok=True)
tag=sys.argv[sys.argv.index('--run-tag')+1]
(root/'status.json').write_text(json.dumps({'credited_hours':.1}))
(root/'test-tag.txt').write_text(tag)
""", encoding="utf-8")
            (module_dir / "remote_verify.py").write_text("""
import json,sys
mode=sys.argv[sys.argv.index('--mode')+1]
data={'status':'ok'}
if mode=='final': data.update(hours={'zh':.1,'en':.1},revision='fake-head')
print(json.dumps(data))
""", encoding="utf-8")
            (module_dir / "stream_upload.py").write_text("""
import sys,time
from pathlib import Path
done=Path(sys.argv[sys.argv.index('--done-file')+1])
deadline=time.monotonic()+20
while not done.exists() and time.monotonic()<deadline: time.sleep(.05)
raise SystemExit(0 if done.exists() else 1)
""", encoding="utf-8")
            root = base / "output data"
            argv = ["produce_and_upload", "--root", str(root), "--repo", "example/dataset",
                    "--hours-zh", ".01", "--hours-en", ".01", "--workers", "2", "--execute"]
            with patch.object(coordinator, "ROOT", source), patch.object(coordinator, "load_env"), \
                 patch.dict(os.environ, {"PYTHONPATH": str(source)}), patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                coordinator.main()
            result = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertEqual(result["status"], "uploaded_and_verified")
            self.assertEqual((root / "zh" / "test-tag.txt").read_text(), result["run_tag"])
            self.assertEqual((root / "en" / "test-tag.txt").read_text(), result["run_tag"])

    def test_guard_runs_commands_with_spaces_and_preserves_exit_status(self):
        from scripts.production.moss_scale.process_guard import OwnedProcesses
        with OwnedProcesses() as owner:
            process = owner.popen([sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(7)",
                                   "argument with spaces"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 7)
            self.assertEqual(stdout.strip(), "argument with spaces")
            self.assertEqual(stderr, "")

    def test_coordinator_stop_is_recovered_without_removing_user_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            control = base / "control"
            roots = {"zh": base / "zh", "en": base / "en"}
            control.mkdir()
            for root in roots.values():
                root.mkdir()
            stop = roots["zh"] / ".chorus-coordinator-stop"
            stop.write_text("owned", encoding="ascii")
            (roots["en"] / "STOP").write_text("user", encoding="ascii")
            _clear_owned_stops(roots)
            self.assertFalse(stop.exists())
            self.assertTrue((roots["en"] / "STOP").exists())

    @unittest.skipUnless(os.name == "nt", "Windows job-object process lifetime test")
    def test_windows_workers_exit_if_their_coordinator_dies(self):
        from ctypes import wintypes
        code = """
from pathlib import Path
import sys,time
from scripts.production.moss_scale.process_guard import OwnedProcesses
with OwnedProcesses() as owner:
    child=owner.popen([sys.executable,'-c','import time; time.sleep(60)'])
    Path(sys.argv[1]).write_text(str(child.pid),encoding='ascii')
    time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "child-pid"
            parent = subprocess.Popen([sys.executable, "-c", code, str(marker)], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            handle = None
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            try:
                deadline = time.monotonic() + 10
                pid = None
                while pid is None and parent.poll() is None and time.monotonic() < deadline:
                    try:
                        pid = int(marker.read_text(encoding="ascii"))
                    except (FileNotFoundError, ValueError):
                        pass
                    time.sleep(.05)
                self.assertIsNotNone(pid, "test parent did not start its child")
                handle = kernel.OpenProcess(0x00100001, False, pid)
                self.assertTrue(handle)
                parent.terminate()
                parent.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
            finally:
                if parent.poll() is None:
                    parent.terminate()
                    parent.wait(timeout=5)
                if handle:
                    if kernel.WaitForSingleObject(handle, 0) != 0:
                        kernel.TerminateProcess(handle, 1)  # Only the test-owned child handle.
                    kernel.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Windows worker startup race test")
    def test_windows_worker_waits_for_job_assignment(self):
        from ctypes import wintypes
        code = """
from pathlib import Path
import ctypes,sys,time
from ctypes import wintypes
from scripts.production.moss_scale.process_guard import OwnedProcesses
with OwnedProcesses() as owner:
    original=owner.kernel
    class Delayed:
        def __getattr__(self, name): return getattr(original, name)
        def AssignProcessToJobObject(self, job, handle):
            original.GetProcessId.argtypes=(wintypes.HANDLE,)
            original.GetProcessId.restype=wintypes.DWORD
            Path(sys.argv[1]).write_text(str(original.GetProcessId(handle)),encoding='ascii')
            time.sleep(60)
            return original.AssignProcessToJobObject(job,handle)
    owner.kernel=Delayed()
    owner.popen([sys.executable,'-c','import time; time.sleep(60)'])
"""
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "child-pid"
            parent = subprocess.Popen([sys.executable, "-c", code, str(marker)], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = None
            try:
                deadline = time.monotonic() + 10
                pid = None
                while pid is None and parent.poll() is None and time.monotonic() < deadline:
                    try:
                        pid = int(marker.read_text(encoding="ascii"))
                    except (FileNotFoundError, ValueError):
                        pass
                    time.sleep(.05)
                self.assertIsNotNone(pid, "test parent did not reach assignment")
                handle = kernel.OpenProcess(0x00100001, False, pid)
                self.assertTrue(handle)
                parent.terminate()
                parent.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
            finally:
                if parent.poll() is None:
                    parent.terminate()
                    parent.wait(timeout=5)
                if handle:
                    if kernel.WaitForSingleObject(handle, 0) != 0:
                        kernel.TerminateProcess(handle, 1)
                    kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
