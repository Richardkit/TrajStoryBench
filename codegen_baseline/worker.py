"""Credential-free execution worker with file/network/process guardrails.

This is a Python audit guard, not an OS-level security sandbox. Run only trusted
benchmark-generated programs; adversarial code needs an external sandbox.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def main():
    program = Path(sys.argv[1]).resolve()
    area = program.parent
    inputs = area / "input"
    artifacts = area / "artifacts"
    # Read-only runtime roots permit normal imports without exposing the repository.
    runtime = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    if os.name == "nt" and os.environ.get("SystemRoot"):
        runtime.append((Path(os.environ["SystemRoot"]) / "Fonts").resolve())
    read_roots = [area, *runtime]
    write_roots = [artifacts]
    os.chdir(area)
    sys.dont_write_bytecode = True

    def check(value, write=False):
        if value is None or isinstance(value, int):
            return
        path = Path(os.fsdecode(value)).resolve()
        if not inside(path, write_roots if write else read_roots):
            raise PermissionError("CodeGen file access outside allowed area")
        if write and inside(path, [inputs]):
            raise PermissionError("CodeGen inputs are read-only")

    def guard(event, args):
        if event == "open":
            name, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+"))
            writing = writing or (isinstance(flags, int) and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            check(name, writing)
        elif event in {"os.listdir", "os.scandir"}:
            check(args[0])
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime"}:
            check(args[0], True)
        elif event in {"os.rename", "os.link", "os.symlink"}:
            check(args[0], True)
            check(args[1], True)
        elif event == "os.chdir":
            check(args[0], True)
        elif event == "ctypes.dlopen":
            # Scientific dependencies can load their packaged BLAS/system DLLs.
            name = args[0]
            if not name:
                raise PermissionError("Loading the current process as a native library is disabled")
            native = Path(os.fsdecode(name))
            if native.is_absolute():
                check(native)
            elif os.name == "nt":
                dll_name = str(native) if native.suffix else str(native) + ".dll"
                library_dirs = [Path(sys.base_prefix) / "Library/bin",
                                Path(os.environ["SystemRoot"]) / "System32"]
                if native.name != str(native) or not any((d / dll_name).is_file() for d in library_dirs):
                    raise PermissionError("Native library is outside the Python/system runtime")
            else:
                raise PermissionError("Native library requires an allowed absolute path")
        elif (event.startswith("socket.") or event.startswith("subprocess.")
              or event in {"os.system", "os.fork", "os.posix_spawn", "os.spawn",
                           "os.exec", "os.startfile", "os.startfile/2"}):
            raise PermissionError("CodeGen network/process access disabled")

    sys.addaudithook(guard)
    # One invocation of the submitted program, including all its own data handling.
    runpy.run_path(str(program), run_name="__main__")


if __name__ == "__main__":
    main()
