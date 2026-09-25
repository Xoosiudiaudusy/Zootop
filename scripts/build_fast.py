"""Build the C++ core (``negpluribus._fastcore``) in place with CMake + pybind11.

    python -m pip install pybind11 cmake ninja      # once (cmake/ninja optional if already installed)
    python scripts/build_fast.py                    # -> negpluribus/_fastcore.cp3XX-win_amd64.pyd
    python scripts/build_fast.py --clean            # wipe build/fast first
    python scripts/build_fast.py --generator Ninja  # needs cl.exe on PATH (run from a VS dev prompt)

On Windows the default is the Visual Studio generator, which finds MSBuild/MSVC by itself, so
no developer prompt is needed.  On Linux/macOS the default generator (Unix Makefiles / Ninja)
and the system compiler are used.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSRC = os.path.join(ROOT, "csrc")
BUILD = os.path.join(ROOT, "build", "fast")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--generator", default=None, help="CMake generator (default: CMake's choice; VS on Windows)")
    ap.add_argument("--config", default="Release")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    try:
        import pybind11
    except ImportError:
        print("pybind11 missing: python -m pip install pybind11", file=sys.stderr)
        return 2
    cmake = shutil.which("cmake")
    if cmake is None:
        print("cmake not found on PATH: python -m pip install cmake (or install CMake)", file=sys.stderr)
        return 2

    if args.clean and os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD, exist_ok=True)

    configure = [
        cmake, "-S", CSRC, "-B", BUILD,
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        f"-DCMAKE_BUILD_TYPE={args.config}",
    ]
    if args.generator:
        configure += ["-G", args.generator]
    elif os.name == "nt":
        configure += ["-A", "x64"]
    print("+", " ".join(configure), flush=True)
    subprocess.check_call(configure)

    build = [cmake, "--build", BUILD, "--config", args.config]
    if args.jobs:
        build += ["--parallel", str(args.jobs)]
    else:
        build += ["--parallel"]
    print("+", " ".join(build), flush=True)
    subprocess.check_call(build)

    # smoke test in a fresh interpreter (the module may already be imported in this one)
    code = "import negpluribus._fastcore as f, sys; print('built', f.__file__); print('evaluate([0,5,10,15,20,25,30]) =', f.evaluate([0,5,10,15,20,25,30]))"
    subprocess.check_call([sys.executable, "-c", code], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
