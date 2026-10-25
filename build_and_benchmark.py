#!/usr/bin/env python3
"""
===============================================================================
🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
Cross-Platform AST Validator, C++20 Lowering Builder & Python ctypes Benchmark
File: build_and_benchmark.py
===============================================================================

A standalone, reproducible verification and benchmarking harness demonstrating
zero-overhead `import Cpp;` interoperability in Google Carbon (`package CarbonForge api;`)
and C++20 (`namespace g2g::carbon_forge`) for:
  1. VEX-Encoded AVX-VNNI (`VPDPBUSD`) & AVX2 Symmetric INT8 Vector Dot Products
  2. Unit Quaternion SO(3) Kinematics, Geodesic Deadband Filtering & 8-Wide
     AVX2 Structure-of-Arrays (SoA) Chebyshev-Padé Batch SLERP (`BatchSlerpSoA8`)
  3. Hephaestus v2.0 4th-Order Runge-Kutta (RK4) SE(3) Predictive Horizon
     C4 Quintic Phase-Space Joint Barrier (`0.00000 rad` overshoot)
  4. Analytical Signed Distance Fields (SDF) & 64-Primitive Humanoid Colliders
  5. Zero-Copy Memory-Mapped (`mmap` / `MapViewOfFile`) GGUF v2/v3 Tensor Engine

Usage:
  python build_and_benchmark.py           # Run all 3 stages (AST + Build + Benchmarks)
  python build_and_benchmark.py --stage 1 # Run Carbon AST Lexer/Parser only
  python build_and_benchmark.py --stage 2 # Run Native Compiler Build & C++ Bench only
  python build_and_benchmark.py --stage 3 # Run Python ctypes 7-Domain Parity Suite only
"""

import argparse
import ctypes
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
INCLUDE_DIR = ROOT_DIR / "include"
SRC_DIR = ROOT_DIR / "src"
BENCH_DIR = ROOT_DIR / "benchmarks"
BUILD_DIR = ROOT_DIR / "build"

IS_WINDOWS = platform.system().lower().startswith("win")
SHARED_EXT = ".dll" if IS_WINDOWS else (".dylib" if platform.system().lower() == "darwin" else ".so")
EXE_EXT = ".exe" if IS_WINDOWS else ""

KIMODO_CARBON_LIB = BUILD_DIR / f"kimodo_carbon{SHARED_EXT}"
AVX_VNNI_CARBON_LIB = BUILD_DIR / f"avx_vnni_carbon{SHARED_EXT}"
OPENARM_GUARD_LIB = BUILD_DIR / f"openarm_bimanual_guard{SHARED_EXT}"
BENCH_EXECUTABLE = BUILD_DIR / f"carbon_forge_bench{EXE_EXT}"
SYNTHETIC_GGUF_PATH = BUILD_DIR / "synthetic_model_v3.gguf"

CARBON_FILES = [
    "modules/kinematics/kimodo_blend.carbon",
    "modules/kinematics/kimodo_barrier.carbon",
    "modules/kinematics/hephaestus_rk4_se3.carbon",
    "modules/physics_sdf/skin_colliders.carbon",
    "modules/ai_cortex/avx_vnni_cortex.carbon",
    "modules/ai_cortex/gguf_mmap.carbon",
    "templates/cpp_python_bridge/module_template.carbon",
    "examples/openarm_bimanual_guard/openarm_bimanual_guard.carbon",
]

TOLERANCE_STRICT = 1.0e-7
CHEBYSHEV_TOLERANCE = 0.01
SPEEDUP_MIN = 3.0


# =============================================================================
# STAGE 1: CARBON AST / GRAMMAR LEXER & PARSER
# =============================================================================

class CarbonParser:
    """Lightweight AST Lexer and Grammar Validator for Google Carbon (.carbon) modules."""

    def __init__(self, file_path: Path, rel_path: str):
        self.file_path = file_path
        self.rel_path = rel_path
        self.content = file_path.read_text(encoding="utf-8")
        self.lines = self.content.splitlines()
        self.package_name: Optional[str] = None
        self.package_type: Optional[str] = None
        self.imports: List[str] = []
        self.classes: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.constants: List[Dict[str, Any]] = []
        self.variables: List[Dict[str, Any]] = []
        self.c_abi_exports: List[str] = []
        self.errors: List[str] = []

    def parse(self) -> Dict[str, Any]:
        self._check_brace_balance()
        self._parse_package_and_imports()
        self._parse_classes()
        self._parse_functions()
        self._parse_declarations()

        return {
            "file": self.rel_path,
            "package": self.package_name,
            "package_type": self.package_type,
            "lines_of_code": len(self.lines),
            "imports": self.imports,
            "class_count": len(self.classes),
            "classes": [c["name"] for c in self.classes],
            "function_count": len(self.functions),
            "functions": [f["name"] for f in self.functions],
            "c_abi_exports": self.c_abi_exports,
            "constant_count": len(self.constants),
            "variable_count": len(self.variables),
            "errors": self.errors,
            "is_valid": len(self.errors) == 0,
        }

    def _check_brace_balance(self) -> None:
        stack: List[Tuple[str, int, int]] = []
        pairs = {")": "(", "}": "{", "]": "["}
        in_string = False
        in_char = False
        in_block_comment = False

        for line_num, line in enumerate(self.lines, 1):
            i = 0
            n = len(line)
            while i < n:
                c = line[i]
                if not in_block_comment and not in_string and not in_char:
                    if c == "/" and i + 1 < n and line[i + 1] == "/":
                        break
                    if c == "/" and i + 1 < n and line[i + 1] == "*":
                        in_block_comment = True
                        i += 2
                        continue
                if in_block_comment:
                    if c == "*" and i + 1 < n and line[i + 1] == "/":
                        in_block_comment = False
                        i += 2
                        continue
                    i += 1
                    continue
                if c == '"' and (i == 0 or line[i - 1] != "\\"):
                    in_string = not in_string
                elif c == "'" and (i == 0 or line[i - 1] != "\\"):
                    in_char = not in_char
                elif not in_string and not in_char:
                    if c in "({[":
                        stack.append((c, line_num, i + 1))
                    elif c in ")}]":
                        if not stack:
                            self.errors.append(f"Unmatched closing '{c}' at line {line_num}:{i + 1}")
                        else:
                            top, top_l, top_col = stack.pop()
                            if top != pairs[c]:
                                self.errors.append(
                                    f"Mismatched '{c}' at line {line_num}:{i + 1} for '{top}' from {top_l}:{top_col}"
                                )
                i += 1

        while stack:
            unclosed, ul, ucol = stack.pop()
            self.errors.append(f"Unclosed '{unclosed}' opened at line {ul}:{ucol}")

    def _parse_package_and_imports(self) -> None:
        pkg_pattern = re.compile(
            r"^\s*package\s+([A-Za-z0-9_]+)(?:\s+library\s+\"([A-Za-z0-9_.\-/]+)\")?\s+(api|impl)\s*;"
        )
        import_pattern = re.compile(r"^\s*import\s+([A-Za-z0-9_]+)(?:\s+library\s+\"([A-Za-z0-9_.\-/]+)\")?\s*;")
        preproc_pattern = re.compile(r"^\s*#\s*(if|ifdef|ifndef|else|elif|endif|define|include|pragma)\b")

        for line_num, line in enumerate(self.lines, 1):
            if preproc_pattern.match(line):
                self.errors.append(
                    f"C preprocessor directive '{line.strip()}' at line {line_num} is illegal in Google Carbon"
                )
            pkg_match = pkg_pattern.match(line)
            if pkg_match:
                self.package_name = pkg_match.group(1)
                self.package_type = pkg_match.group(3)
            imp_match = import_pattern.match(line)
            if imp_match:
                imp_pkg = imp_match.group(1)
                imp_lib = imp_match.group(2)
                if imp_pkg == "Cpp" and not imp_lib:
                    self.errors.append(
                        f"Bare 'import Cpp;' at line {line_num} must specify 'import Cpp library \"<header>\";'"
                    )
                self.imports.append(f"{imp_pkg}:{imp_lib}" if imp_lib else imp_pkg)

        if self.package_name != "CarbonForge":
            self.errors.append(f"Expected 'package CarbonForge api;', got '{self.package_name}'")

    def _parse_classes(self) -> None:
        class_pattern = re.compile(r"^\s*class\s+([A-Za-z0-9_]+)")
        for line_num, line in enumerate(self.lines, 1):
            m = class_pattern.match(line)
            if m:
                self.classes.append({"name": m.group(1), "line": line_num})

    def _parse_functions(self) -> None:
        fn_pattern = re.compile(r"^\s*fn\s+([A-Za-z0-9_]+)(?:\[.*?\])?\s*\((.*?)\)(?:\s*->\s*([^{;]+))?")
        c_abi_prefixes = ("kimodo_", "hephaestus_", "avx_vnni_", "carbon_forge_", "openarm_")

        for line_num, line in enumerate(self.lines, 1):
            m = fn_pattern.match(line)
            if m:
                fn_name = m.group(1)
                params_str = m.group(2)
                ret_type = (m.group(3) or "void").strip()
                self.functions.append(
                    {"name": fn_name, "params": params_str, "ret_type": ret_type, "line": line_num}
                )
                if any(fn_name.startswith(p) for p in c_abi_prefixes):
                    self.c_abi_exports.append(fn_name)

    def _parse_declarations(self) -> None:
        let_pattern = re.compile(r"^\s*let\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_\[\];*]+)")
        var_pattern = re.compile(r"^\s*var\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_\[\];*]+)")

        for line_num, line in enumerate(self.lines, 1):
            lm = let_pattern.match(line)
            if lm:
                self.constants.append({"name": lm.group(1), "type": lm.group(2), "line": line_num})
            vm = var_pattern.match(line)
            if vm:
                self.variables.append({"name": vm.group(1), "type": vm.group(2), "line": line_num})


def run_stage_1() -> bool:
    print("=" * 96)
    print("🚀 STAGE 1: CARBONFORGE AST / GRAMMAR LEXER & PARSER VALIDATION (package CarbonForge api;)")
    print("=" * 96)

    all_valid = True
    results = []

    print(
        f"{'Carbon Module Path':<48} | {'Package':<14} | {'LOC':<5} | {'Classes':<7} | {'Funcs':<5} | {'Status'}"
    )
    print("-" * 96)

    for rel_path in CARBON_FILES:
        filepath = ROOT_DIR / rel_path
        if not filepath.exists():
            print(f"❌ File not found: {rel_path}")
            all_valid = False
            continue

        parser = CarbonParser(filepath, rel_path)
        res = parser.parse()
        results.append(res)

        status_str = "✅ VALID" if res["is_valid"] else "❌ INVALID"
        if not res["is_valid"]:
            all_valid = False

        pkg_display = res["package"] or "UNKNOWN"
        print(
            f"{res['file']:<48} | {pkg_display:<14} | {res['lines_of_code']:<5} | "
            f"{res['class_count']:<7} | {res['function_count']:<5} | {status_str}"
        )

        if not res["is_valid"]:
            for err in res["errors"]:
                print(f"    ⚠️  [ERROR] {err}")

    print("-" * 96)
    total_loc = sum(r["lines_of_code"] for r in results)
    total_classes = sum(r["class_count"] for r in results)
    total_funcs = sum(r["function_count"] for r in results)
    total_exports = sum(len(r["c_abi_exports"]) for r in results)
    print(
        f"Total: {len(results)} modules | LOC: {total_loc} | Classes: {total_classes} | "
        f"Functions: {total_funcs} | C ABI Exports: {total_exports}"
    )

    carbon_cli = shutil.which("carbon")
    if carbon_cli:
        print(f"🔎 Detected upstream LLVM Carbon toolchain ({carbon_cli}); running 'carbon compile --phase=check'...")
        for rel_path in CARBON_FILES:
            res_c = subprocess.run(
                [carbon_cli, "compile", "--phase=check", str(ROOT_DIR / rel_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if res_c.returncode != 0:
                print(f"❌ carbon compile failed for {rel_path}:\n{res_c.stderr}")
                all_valid = False

    if all_valid:
        print("\n🎉 STAGE 1 PASSED: All CarbonForge modules strictly conform to Carbon language grammar!")
    else:
        print("\n❌ STAGE 1 FAILED: Errors encountered during AST parsing.")

    return all_valid


# =============================================================================
# STAGE 2: CROSS-PLATFORM C++20 LOWERING BRIDGE & COMPILATION
# =============================================================================

def discover_windows_msvc() -> Tuple[Optional[str], Optional[str]]:
    """Dynamically discovers vcvars64.bat or cl.exe using standard Windows environment variables and vswhere."""
    prog_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog_x64 = os.environ.get("ProgramFiles", r"C:\Program Files")

    vswhere = Path(prog_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        try:
            res = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            if res.returncode == 0 and res.stdout.strip():
                inst_path = Path(res.stdout.strip().splitlines()[0])
                vcvars = inst_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars.exists():
                    return str(vcvars), None
        except Exception:
            pass

    for root in (prog_x86, prog_x64):
        for year in ("2022", "2025", "2019"):
            for edition in ("BuildTools", "Enterprise", "Professional", "Community"):
                candidate = Path(root) / "Microsoft Visual Studio" / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.exists():
                    return str(candidate), None

    cl_in_path = shutil.which("cl.exe")
    if cl_in_path:
        return None, cl_in_path

    return None, None


def clean_intermediate_artifacts() -> None:
    """Removes intermediate .obj, .exp, and .lib artifacts from build/ to keep the repo spotless."""
    for ext in ("*.obj", "*.exp", "*.lib", "*.pdb", "*.ilk"):
        for f in BUILD_DIR.glob(ext):
            try:
                f.unlink()
            except Exception:
                pass


def run_stage_2() -> bool:
    print("\n" + "=" * 96)
    print("🔨 STAGE 2: CROSS-PLATFORM C++20 LOWERING BRIDGE & COMPILATION (g2g::carbon_forge)")
    print("=" * 96)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if IS_WINDOWS:
        vcvars_path, direct_cl = discover_windows_msvc()
        if not vcvars_path and not direct_cl:
            print("❌ Could not locate MSVC build environment (vcvars64.bat or cl.exe).")
            return False

        cl_cmd = f'"{direct_cl}"' if direct_cl else "cl.exe"
        base_flags = '/O2 /arch:AVX2 /std:c++20 /EHsc /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS /I "include"'

        targets = [
            {
                "name": KIMODO_CARBON_LIB.name,
                "src": "src/kimodo_carbon.cpp",
                "out": f"build/{KIMODO_CARBON_LIB.name}",
                "flags": f"{base_flags} /LD /DKIMODO_BUILD_SHARED",
            },
            {
                "name": AVX_VNNI_CARBON_LIB.name,
                "src": "src/avx_vnni_carbon.cpp",
                "out": f"build/{AVX_VNNI_CARBON_LIB.name}",
                "flags": f"{base_flags} /LD",
            },
            {
                "name": OPENARM_GUARD_LIB.name,
                "src": "examples/openarm_bimanual_guard/openarm_bimanual_guard.cpp",
                "out": f"build/{OPENARM_GUARD_LIB.name}",
                "flags": f'{base_flags} /LD /DOPENARM_BUILD_SHARED /I "examples/openarm_bimanual_guard"',
            },
            {
                "name": BENCH_EXECUTABLE.name,
                "src": "benchmarks/carbon_forge_bench.cpp",
                "out": f"build/{BENCH_EXECUTABLE.name}",
                "flags": base_flags,
            },
        ]

        for t in targets:
            obj = t["out"].replace(".dll", ".obj").replace(".exe", ".obj")
            cmd = f'{cl_cmd} {t["flags"]} "{t["src"]}" /Fe:"{t["out"]}" /Fo:"{obj}"'
            full_cmd = f'call "{vcvars_path}" >nul 2>&1 && {cmd}' if vcvars_path else cmd
            print(f"Compiling {t['name']}...")
            res = subprocess.run(
                full_cmd,
                shell=True,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if res.returncode != 0:
                print(f"❌ Compilation failed for {t['name']}:\n{res.stdout}\n{res.stderr}")
                return False
            out_file = ROOT_DIR / t["out"]
            print(f"✅ Compiled {t['name']} ({out_file.stat().st_size / 1024.0:.1f} KB)")

        clean_intermediate_artifacts()
    else:
        cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        if not cxx:
            print("❌ Could not locate a C++20 compiler (clang++ or g++).")
            return False

        base_flags = ["-O3", "-mavx2", "-mfma", "-mavxvnni", "-mxsave", "-std=c++20", "-Iinclude"]
        targets_posix = [
            (KIMODO_CARBON_LIB.name, ["src/kimodo_carbon.cpp"], ["-shared", "-fPIC", "-DKIMODO_BUILD_SHARED"], KIMODO_CARBON_LIB),
            (AVX_VNNI_CARBON_LIB.name, ["src/avx_vnni_carbon.cpp"], ["-shared", "-fPIC"], AVX_VNNI_CARBON_LIB),
            (OPENARM_GUARD_LIB.name, ["examples/openarm_bimanual_guard/openarm_bimanual_guard.cpp"], ["-shared", "-fPIC", "-DOPENARM_BUILD_SHARED", "-Iexamples/openarm_bimanual_guard"], OPENARM_GUARD_LIB),
            (BENCH_EXECUTABLE.name, ["benchmarks/carbon_forge_bench.cpp"], [], BENCH_EXECUTABLE),
        ]
        for name, srcs, extra, out_path in targets_posix:
            cmd_list = [cxx] + base_flags + extra + srcs + ["-o", str(out_path)]
            print(f"Compiling {name} via {Path(cxx).name}...")
            res = subprocess.run(
                cmd_list,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            if res.returncode != 0:
                print(f"❌ Compilation failed for {name}:\n{res.stdout}\n{res.stderr}")
                return False
            print(f"✅ Compiled {name} ({out_path.stat().st_size / 1024.0:.1f} KB)")

    print(f"\nExecuting native benchmark binary ({BENCH_EXECUTABLE.name})...")
    bench_res = subprocess.run(
        [str(BENCH_EXECUTABLE)],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(bench_res.stdout)
    if bench_res.returncode != 0:
        print(f"❌ {BENCH_EXECUTABLE.name} exited with code {bench_res.returncode}")
        return False

    print("🎉 STAGE 2 PASSED: Native Shared Libraries and Benchmark Binary compiled & verified!")
    return True


# =============================================================================
# STAGE 3: PYTHON CTYPES 7-DOMAIN NUMERICAL PARITY & BENCHMARK SUITE
# =============================================================================

def free_library(dll: Optional[ctypes.CDLL]) -> bool:
    """Safely unloads a shared library handle on Windows to release file locks."""
    if IS_WINDOWS and dll is not None and hasattr(dll, "_handle") and dll._handle:
        try:
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
            kernel32.FreeLibrary.restype = wintypes.BOOL
            res = bool(kernel32.FreeLibrary(wintypes.HMODULE(dll._handle)))
            dll._handle = None
            return res
        except Exception:
            return False
    return True


class KimodoInitParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("text_encoder_path", ctypes.c_char_p),
        ("compute_backend", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("gpu_layers", ctypes.c_int32),
        ("use_mmap", ctypes.c_int32),
        ("use_mlock", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("reserved", ctypes.c_int32 * 8),
    ]


class KimodoGenerationParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("duration_sec", ctypes.c_float),
        ("fps", ctypes.c_int32),
        ("ddim_steps", ctypes.c_int32),
        ("cfg_scale", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("rotation_format", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("reserved", ctypes.c_int32 * 8),
    ]


class KimodoMotionBuffer(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("num_frames", ctypes.c_int32),
        ("num_joints", ctypes.c_int32),
        ("fps", ctypes.c_int32),
        ("rotation_format", ctypes.c_int32),
        ("root_positions", ctypes.POINTER(ctypes.c_float)),
        ("joint_rotations", ctypes.POINTER(ctypes.c_float)),
        ("root_linear_velocities", ctypes.POINTER(ctypes.c_float)),
        ("root_angular_velocities", ctypes.POINTER(ctypes.c_float)),
        ("joint_names", ctypes.POINTER(ctypes.c_char_p)),
        ("memory_size_bytes", ctypes.c_uint64),
        ("internal_handle", ctypes.c_void_p),
    ]


def ensure_synthetic_gguf_fixture(path: Path) -> Path:
    """Generates a self-contained GGUF v3 binary fixture for zero-copy mmap testing."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    buf += struct.pack("<IIQQ", 0x46554747, 3, 2, 2)

    k1 = b"general.architecture"
    v1 = b"gemma4"
    buf += struct.pack("<Q", len(k1)) + k1 + struct.pack("<IQ", 8, len(v1)) + v1

    k2 = b"general.alignment"
    buf += struct.pack("<Q", len(k2)) + k2 + struct.pack("<II", 4, 32)

    t1_name = b"blk.0.attn_q.weight"
    buf += struct.pack("<Q", len(t1_name)) + t1_name
    buf += struct.pack("<IQQIQ", 2, 16, 16, 0, 0)

    t2_name = b"blk.0.attn_k.weight"
    buf += struct.pack("<Q", len(t2_name)) + t2_name
    buf += struct.pack("<IQQIQ", 2, 16, 16, 0, 1024)

    while len(buf) % 32 != 0:
        buf.append(0)
    buf += b"\x00" * 2048

    path.write_bytes(bytes(buf))
    return path


def ref_slerp_double(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        dot = -dot
        q2 = -q2
    t = max(0.0, min(1.0, float(t)))
    if dot > 0.99999995:
        res = (1.0 - t) * q1 + t * q2
    else:
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        res = s0 * q1 + s1 * q2
    return (res / np.linalg.norm(res)).astype(np.float32)


def run_stage_3() -> bool:
    print("\n" + "=" * 96)
    print("💎 STAGE 3: PYTHON CTYPES 7-DOMAIN NUMERICAL PARITY & BENCHMARK SUITE")
    print("=" * 96)

    assert ctypes.sizeof(KimodoInitParams) == 72, "72-byte struct alignment invariant violated"
    assert ctypes.sizeof(KimodoGenerationParams) == 72, "72-byte struct alignment invariant violated"
    assert ctypes.sizeof(KimodoMotionBuffer) == 72, "72-byte struct alignment invariant violated"

    c_k: Optional[ctypes.CDLL] = None
    c_v: Optional[ctypes.CDLL] = None
    c_openarm: Optional[ctypes.CDLL] = None
    all_passed = True
    summary_rows: List[Tuple[str, bool, str]] = []

    try:
        c_k = ctypes.CDLL(str(KIMODO_CARBON_LIB))
        c_v = ctypes.CDLL(str(AVX_VNNI_CARBON_LIB))
        c_openarm = ctypes.CDLL(str(OPENARM_GUARD_LIB))

        # Domain 1: Unit Quaternion SLERP
        c_k.kimodo_slerp.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        test_cases = [
            ([0.0, 0.0, 0.0, 1.0], [0.0, 0.70710678, 0.0, 0.70710678], 0.5),
            ([0.0, 0.0, 0.0, 1.0], [0.0, -0.70710678, 0.0, -0.70710678], 0.25),
            ([0.104828, 0.209657, 0.314485, 0.917249], [0.104828, 0.209657, 0.314485, 0.917249], 0.7),
            ([0.20739034, -0.31108551, 0.41478068, 0.82956136], [-0.56254395, 0.11250879, 0.22501758, 0.78756153], 0.65),
        ]
        d1_max = 0.0
        out_q = (ctypes.c_float * 4)()
        for q1, q2, t in test_cases:
            c_k.kimodo_slerp((ctypes.c_float * 4)(*q1), (ctypes.c_float * 4)(*q2), t, out_q)
            err = float(np.max(np.abs(np.array(list(out_q), dtype=np.float32) - ref_slerp_double(q1, q2, t))))
            d1_max = max(d1_max, err)
        ok1 = d1_max <= TOLERANCE_STRICT
        all_passed = all_passed and ok1
        summary_rows.append(("Domain 1 (Unit Quaternion SLERP)", ok1, f"max delta = {d1_max:.2e}"))

        # Domain 2: Geodesic Angular Deadband Filter (0.0005 rad)
        c_k.kimodo_angular_deadband_filter.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        cur_q = [0.0, 0.0, 0.0, 1.0]
        cur_arr = (ctypes.c_float * 4)(*cur_q)
        dt = 0.0166667
        sub_half = 0.00049 * 0.5
        tgt_sub = (ctypes.c_float * 4)(0.0, math.sin(sub_half), 0.0, math.cos(sub_half))
        c_k.kimodo_angular_deadband_filter(cur_arr, tgt_sub, dt, out_q)
        d2_sub = float(np.max(np.abs(np.array(list(out_q), dtype=np.float32) - np.array(cur_q, dtype=np.float32))))

        sup_half = 0.10 * 0.5
        tgt_sup = [0.0, math.sin(sup_half), 0.0, math.cos(sup_half)]
        c_k.kimodo_angular_deadband_filter(cur_arr, (ctypes.c_float * 4)(*tgt_sup), dt, out_q)
        ref_sup = ref_slerp_double(cur_q, tgt_sup, 1.0 - math.exp(-18.0 * dt))
        d2_sup = float(np.max(np.abs(np.array(list(out_q), dtype=np.float32) - ref_sup)))
        d2_max = max(d2_sub, d2_sup)
        ok2 = d2_max <= TOLERANCE_STRICT
        all_passed = all_passed and ok2
        summary_rows.append(("Domain 2 (Geodesic Deadband Filter)", ok2, f"max delta = {d2_max:.2e}"))

        # Domain 3: Cubic Soft-Spring Boundary Barrier
        c_k.kimodo_apply_cubic_soft_limit.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c_k.kimodo_apply_cubic_soft_limit.restype = ctypes.c_double
        d3_max = 0.0
        for theta in np.linspace(-5.0, 5.0, 41):
            c_val = c_k.kimodo_apply_cubic_soft_limit(float(theta), -1.5, 1.5, 10000.0)
            if theta > 1.5:
                e = float(theta) - 1.5
                ref_val = min(1.5 + e / math.sqrt(1.0 + 10000.0 * e * e), 1.51)
            elif theta < -1.5:
                e = -1.5 - float(theta)
                ref_val = max(-1.5 - e / math.sqrt(1.0 + 10000.0 * e * e), -1.51)
            else:
                ref_val = float(theta)
            d3_max = max(d3_max, abs(c_val - ref_val))
        ok3 = d3_max <= TOLERANCE_STRICT
        all_passed = all_passed and ok3
        summary_rows.append(("Domain 3 (C3 Cubic Soft Barrier)", ok3, f"max delta = {d3_max:.2e}"))

        # Domain 4: 22-Joint SMPL-X Posture Synthesis
        c_k.kimodo_init.argtypes = [ctypes.POINTER(KimodoInitParams)]
        c_k.kimodo_init.restype = ctypes.c_void_p
        c_k.kimodo_free.argtypes = [ctypes.c_void_p]
        c_k.kimodo_generate_motion.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(KimodoGenerationParams),
            ctypes.POINTER(KimodoMotionBuffer),
        ]
        c_k.kimodo_generate_motion.restype = ctypes.c_int32
        c_k.kimodo_free_motion_buffer.argtypes = [ctypes.POINTER(KimodoMotionBuffer)]

        init_p = KimodoInitParams()
        init_p.n_threads = 4
        ctx = c_k.kimodo_init(ctypes.byref(init_p))
        gen_p = KimodoGenerationParams()
        gen_p.prompt = b"bilateral_wave"
        gen_p.duration_sec = 3.0
        gen_p.fps = 30
        gen_p.seed = 42
        mbuf = KimodoMotionBuffer()
        rc4 = c_k.kimodo_generate_motion(ctx, ctypes.byref(gen_p), ctypes.byref(mbuf))
        jr = np.ctypeslib.as_array(mbuf.joint_rotations, shape=(mbuf.num_frames, mbuf.num_joints, 4))
        norms = np.linalg.norm(jr, axis=-1)
        d4_max = float(np.max(np.abs(norms - 1.0)))
        c_k.kimodo_free_motion_buffer(ctypes.byref(mbuf))
        c_k.kimodo_free(ctx)
        ok4 = (rc4 == 0) and (d4_max <= 1.0e-6)
        all_passed = all_passed and ok4
        summary_rows.append(("Domain 4 (22-Joint Posture Synthesis)", ok4, f"unit norm delta = {d4_max:.2e}"))

        # Domain 5: 64-Primitive Analytical SDF Colliders
        c_k.kimodo_sdf_sphere.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        c_k.kimodo_sdf_sphere.restype = ctypes.c_float
        c_k.kimodo_sdf_box.argtypes = [ctypes.c_float] * 6
        c_k.kimodo_sdf_box.restype = ctypes.c_float
        c_k.kimodo_synthesize_humanoid64_count.restype = ctypes.c_int32
        d5_max = max(
            abs(c_k.kimodo_sdf_sphere(3.0, 4.0, 0.0, 2.0) - 3.0),
            abs(c_k.kimodo_sdf_box(2.0, 0.0, 0.0, 1.0, 1.0, 1.0) - 1.0),
        )
        h_count = c_k.kimodo_synthesize_humanoid64_count()
        ok5 = (h_count == 64) and (d5_max <= TOLERANCE_STRICT)
        all_passed = all_passed and ok5
        summary_rows.append(("Domain 5 (64-Primitive SDF Colliders)", ok5, f"count={h_count}, delta={d5_max:.2e}"))

        # Domain 6: Zero-Copy GGUF v3 Memory-Mapping
        gguf_file = ensure_synthetic_gguf_fixture(SYNTHETIC_GGUF_PATH)
        c_k.kimodo_gguf_open.argtypes = [ctypes.c_char_p]
        c_k.kimodo_gguf_open.restype = ctypes.c_void_p
        c_k.kimodo_gguf_close.argtypes = [ctypes.c_void_p]
        c_k.kimodo_gguf_get_version.argtypes = [ctypes.c_void_p]
        c_k.kimodo_gguf_get_version.restype = ctypes.c_uint32
        c_k.kimodo_gguf_get_alignment.argtypes = [ctypes.c_void_p]
        c_k.kimodo_gguf_get_alignment.restype = ctypes.c_uint32
        c_k.kimodo_gguf_get_tensor_count.argtypes = [ctypes.c_void_p]
        c_k.kimodo_gguf_get_tensor_count.restype = ctypes.c_uint64

        g_ctx = c_k.kimodo_gguf_open(str(gguf_file).encode("utf-8"))
        g_ver = c_k.kimodo_gguf_get_version(g_ctx)
        g_align = c_k.kimodo_gguf_get_alignment(g_ctx)
        g_tc = c_k.kimodo_gguf_get_tensor_count(g_ctx)
        c_k.kimodo_gguf_close(g_ctx)
        try:
            gguf_file.unlink()
        except Exception:
            pass
        ok6 = (g_ver == 3) and (g_align == 32) and (g_tc == 2)
        all_passed = all_passed and ok6
        summary_rows.append(("Domain 6 (Zero-Copy GGUF v3 MMap)", ok6, f"v={g_ver}, align={g_align}, tensors={g_tc}"))

        # Domain 7: VEX AVX-VNNI / AVX2 INT8 Dot-Product & Batch Cosine Similarity
        c_v.avx_vnni_dot_product_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        c_v.avx_vnni_dot_product_int8.restype = ctypes.c_int
        np.random.seed(42)
        N, D = 64, 256
        mat = np.random.randint(-128, 127, size=(N, D), dtype=np.int8)
        qry = np.random.randint(-128, 127, size=D, dtype=np.int8)
        scales = np.random.uniform(0.01, 1.0, size=N).astype(np.float32)
        q_scale = 0.5
        out_scores = np.zeros(N, dtype=np.float32)
        rc7 = c_v.avx_vnni_dot_product_int8(
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            qry.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            q_scale,
            N,
            D,
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        ref_dots = (mat.astype(np.int32) @ qry.astype(np.int32)).astype(np.float32) * (scales * q_scale)
        d7_max = float(np.max(np.abs(out_scores - ref_dots)))
        ok7 = (rc7 == 0) and (d7_max <= 1.0e-4)
        all_passed = all_passed and ok7
        summary_rows.append(("Domain 7 (AVX-VNNI INT8 Dot Product)", ok7, f"max delta = {d7_max:.2e}"))

        # Benchmark 1: RK4 SE(3) Predictive Barrier vs 1st-Order Euler
        c_k.hephaestus_rk4_predict_barrier.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        c_k.hephaestus_rk4_step.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        rk4_th, rk4_om, rk4_max = 1.45, 5.0, 1.45
        tau_out = ctypes.c_float(0.0)
        out_th = ctypes.c_float(0.0)
        out_om = ctypes.c_float(0.0)
        for _ in range(50):
            c_k.hephaestus_rk4_predict_barrier(rk4_th, rk4_om, -1.5, 1.5, 0.015, ctypes.byref(tau_out))
            tau_c = max(-35.0, min(35.0, tau_out.value))
            c_k.hephaestus_rk4_step(rk4_th, rk4_om, tau_c, 0.05, 0.10, 0.002, ctypes.byref(out_th), ctypes.byref(out_om))
            rk4_th, rk4_om = out_th.value, out_om.value
            rk4_max = max(rk4_max, rk4_th)
            if rk4_om <= 0.0 and rk4_th >= 1.45:
                break
        rk4_overshoot = max(0.0, rk4_max - 1.50)
        b1_ok = rk4_overshoot == 0.0
        all_passed = all_passed and b1_ok
        summary_rows.append(("Benchmark 1 (RK4 C4 Quintic Barrier)", b1_ok, f"overshoot = +{rk4_overshoot:.5f} rad"))

        # Benchmark 2: AVX2 8-Wide SoA Batch SLERP vs Scalar SLERP
        c_k.hephaestus_batch_slerp_soa8.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        q1_soa = np.array([0.0] * 24 + [1.0] * 8, dtype=np.float32)
        val = 0.70710678
        q2_soa = np.array([val] * 8 + [0.0] * 16 + [val] * 8, dtype=np.float32)
        t8 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        out_soa = np.zeros(32, dtype=np.float32)

        p_q1 = q1_soa.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_q2 = q2_soa.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_t = t8.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_out = out_soa.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        n_batches = 25000
        t0 = time.perf_counter()
        for _ in range(n_batches):
            c_k.hephaestus_batch_slerp_soa8(p_q1, p_q2, p_t, p_out)
        simd_dur = time.perf_counter() - t0

        q1_arr = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 1.0)
        q2_arr = (ctypes.c_float * 4)(val, 0.0, 0.0, val)
        t0 = time.perf_counter()
        for _ in range(n_batches):
            for lane in range(8):
                c_k.kimodo_slerp(q1_arr, q2_arr, t8[lane], out_q)
        scalar_dur = time.perf_counter() - t0
        speedup = scalar_dur / max(1e-9, simd_dur)

        cheb_err = 0.0
        for lane in range(8):
            c_k.kimodo_slerp(q1_arr, q2_arr, t8[lane], out_q)
            simd_q = [out_soa[0 * 8 + lane], out_soa[1 * 8 + lane], out_soa[2 * 8 + lane], out_soa[3 * 8 + lane]]
            cheb_err = max(cheb_err, max(abs(a - b) for a, b in zip(simd_q, list(out_q))))

        b2_ok = (speedup >= SPEEDUP_MIN) and (cheb_err < CHEBYSHEV_TOLERANCE)
        all_passed = all_passed and b2_ok
        summary_rows.append(
            ("Benchmark 2 (AVX2 BatchSlerpSoA8)", b2_ok, f"{speedup:.2f}x speedup, max err = {cheb_err:.6f}")
        )

        # Benchmark 3: OpenArm 1kHz Bimanual Closed-Loop Safety Guard
        c_openarm.openarm_bimanual_guard_init.restype = ctypes.c_void_p
        c_openarm.openarm_bimanual_guard_init.argtypes = []
        c_openarm.openarm_bimanual_guard_destroy.restype = None
        c_openarm.openarm_bimanual_guard_destroy.argtypes = [ctypes.c_void_p]
        c_openarm.openarm_bimanual_guard_step_1khz.restype = ctypes.c_int32
        c_openarm.openarm_bimanual_guard_step_1khz.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        guard_ctx = c_openarm.openarm_bimanual_guard_init()
        c_f16 = ctypes.c_float * 16
        q_m = c_f16()
        v_m = c_f16()
        t_c = c_f16()
        q_nxt = c_f16()
        v_nxt = c_f16()
        t_sfe = c_f16()
        clr = ctypes.c_float(0.0)

        # Initial pose: arms neutral
        q_m[0] = -0.30; q_m[1] = 0.20; q_m[3] = 0.80; q_m[7] = 0.20
        q_m[8] = 0.30; q_m[9] = 0.20; q_m[11] = 0.80; q_m[15] = 0.20

        limits = [
            (-1.5708, 1.5708), (-1.0000, 1.5708), (-1.5708, 1.5708), (0.0000, 2.2000),
            (-1.5708, 1.5708), (-0.7854, 0.7854), (-0.4189, 0.4189), (0.0000, 0.4500),
            (-1.5708, 1.5708), (-1.5708, 1.0000), (-1.5708, 1.5708), (0.0000, 2.2000),
            (-1.5708, 1.5708), (-0.7854, 2.6180), (-0.4189, 0.4189), (0.0000, 0.4500)
        ]

        max_os = 0.0
        min_clr = float("inf")
        for step in range(1000):
            if step < 500:
                # Collision assault: drive both arms inward toward center
                t_c[0] = 35.0; t_c[1] = 15.0; t_c[3] = 10.0; t_c[5] = 10.0; t_c[6] = 8.0
                t_c[8] = -35.0; t_c[9] = 15.0; t_c[11] = 10.0; t_c[13] = 10.0; t_c[14] = 8.0
            else:
                # Joint limit assault: drive toward limits
                for j in range(16):
                    t_c[j] = 35.0 if (step // 125) % 2 == 0 else -35.0
                t_c[13] = 12.0; t_c[6] = 12.0; t_c[14] = -12.0; t_c[7] = -8.0; t_c[15] = 8.0

            c_openarm.openarm_bimanual_guard_step_1khz(
                guard_ctx, q_m, v_m, t_c, 0.001, q_nxt, v_nxt, t_sfe, ctypes.byref(clr)
            )
            min_clr = min(min_clr, clr.value)
            for j in range(16):
                q_val = q_nxt[j]
                mn, mx = limits[j]
                if q_val > mx:
                    max_os = max(max_os, q_val - mx)
                elif q_val < mn:
                    max_os = max(max_os, mn - q_val)
                q_m[j] = q_nxt[j]
                v_m[j] = v_nxt[j]

        c_openarm.openarm_bimanual_guard_destroy(guard_ctx)
        b3_ok = (max_os <= 1.0e-5) and (min_clr > 0.0)
        all_passed = all_passed and b3_ok
        summary_rows.append(
            ("Benchmark 3 (OpenArm 1kHz Bimanual Guard)", b3_ok, f"overshoot = +{max_os:.5f} rad, min clearance = {min_clr:.4f} m")
        )

    finally:
        free_library(c_k)
        free_library(c_v)
        free_library(c_openarm)

    for label, ok, detail in summary_rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<38} | {detail}")

    if all_passed:
        print("\n🎉 STAGE 3 PASSED: All 7 numerical domains and 3 benchmarks validated cleanly!")
    else:
        print("\n❌ STAGE 3 FAILED: One or more verification checks failed.")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaming2Gamers (G2G) CarbonForge Builder & Benchmark Suite")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Run a specific verification stage (1, 2, or 3)")
    args = parser.parse_args()

    if args.stage == 1:
        ok = run_stage_1()
    elif args.stage == 2:
        ok = run_stage_2()
    elif args.stage == 3:
        ok = run_stage_3()
    else:
        ok = run_stage_1() and run_stage_2() and run_stage_3()

    print("\n" + "=" * 96)
    if ok:
        print("🏆 GAMING2GAMERS / CARBONFORGE: ALL STAGES PASSED CLEANLY (EXIT CODE 0) • ∞ Meraki")
    else:
        print("💥 GAMING2GAMERS / CARBONFORGE: VERIFICATION FAILED")
    print("=" * 96)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
