#!/usr/bin/env python3
"""
===============================================================================
🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
Standalone Python ctypes 7-Domain Parity & Benchmark Runner
File: benchmarks/bench_carbon_parity.py
===============================================================================
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from build_and_benchmark import run_stage_3

if __name__ == "__main__":
    sys.exit(0 if run_stage_3() else 1)
