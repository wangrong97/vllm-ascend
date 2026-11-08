# SPDX-License-Identifier: Apache-2.0
"""Compile and run the production Host offset helper without CANN or Torch."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def host_compiler(temp_dir):
    for candidate in ("c++", "g++", "clang++", "zig", "cl"):
        compiler = shutil.which(candidate)
        if compiler:
            return compiler, os.environ.copy()
    if os.name != "nt":
        return None, None
    # VS Build Tools need not be on PATH in the development shell. Configure a
    # child environment only; never alter the user's shell or global settings.
    program_files = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
    vswhere = program_files / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        return None, None
    installations = subprocess.run(
        [str(vswhere), "-all", "-products", "*", "-property", "installationPath"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for installation in installations:
        root = Path(installation)
        setup = root / "VC/Auxiliary/Build/vcvarsall.bat"
        candidates = sorted((root / "VC/Tools/MSVC").glob("*/bin/Host*/x64/cl.exe"), reverse=True)
        if not setup.is_file() or not candidates:
            continue
        compiler = candidates[0]
        architecture = "x86_amd64" if compiler.parent.parent.name == "Hostx86" else "amd64"
        script = temp_dir / "compiler-environment.bat"
        script.write_text(f'@echo off\ncall "{setup}" {architecture} >nul\nif errorlevel 1 exit /b 1\nset\n')
        configured = subprocess.run(["cmd.exe", "/d", "/c", str(script)], capture_output=True, text=True, check=True)
        environment = os.environ.copy()
        for line in configured.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key:
                environment[key] = value
        # Finding cl.exe alone is insufficient: VS may have been installed
        # without the UCRT SDK. Do not misreport that as a source build error.
        if not any((Path(path) / "crtdbg.h").is_file() for path in environment.get("INCLUDE", "").split(";")):
            continue
        return str(compiler), environment
    return None, None


class WeightOffsetTests(unittest.TestCase):
    def test_compiled_production_helper_matches_sequential_scan(self):
        with tempfile.TemporaryDirectory(prefix="vq2a8-weight-offsets-") as directory:
            temporary = Path(directory)
            compiler, environment = host_compiler(temporary)
            if compiler is None:
                self.skipTest("A usable host C++ toolchain is required for the production offset arithmetic test")
            executable = temporary / ("offset-check.exe" if os.name == "nt" else "offset-check")
            source = ROOT / "tests/weight_offsets_host_test.cpp"
            if Path(compiler).name.lower() in ("cl", "cl.exe"):
                command = [
                    compiler,
                    "/nologo",
                    "/std:c++17",
                    "/utf-8",
                    "/EHsc",
                    "/W4",
                    f"/I{ROOT}",
                    str(source),
                    f"/Fe{executable}",
                ]
            else:
                command = [compiler]
                if Path(compiler).stem.lower() == "zig":
                    command.append("c++")
                command.extend(["-std=c++17", "-Wall", "-Wextra", "-I", str(ROOT), str(source), "-o", str(executable)])
            build = subprocess.run(
                command,
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(executable)], cwd=temporary, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("704 random/boundary banks and 11 invalid cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
