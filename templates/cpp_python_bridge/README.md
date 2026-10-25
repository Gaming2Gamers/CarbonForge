# 🔨 CarbonForge Module Extension Template (`cpp_python_bridge`)

> **By Gaming2Gamers (G2G) • ∞ Meraki**

This directory provides a clean, drop-in blueprint for adding new **Google Carbon (`carbon-lang`)** modules to `Gaming2Gamers/CarbonForge` with zero-overhead C++20 lowering and Python `ctypes` bindings.

## How to Add a New Module in 3 Steps

1. **Author the `.carbon` Specification**:
   - Copy [`module_template.carbon`](module_template.carbon) into `modules/kinematics/`, `modules/physics_sdf/`, or `modules/ai_cortex/`.
   - Declare `package CarbonForge api;` and `import Cpp;` at the top of the file.
2. **Add the C++20 Lowering Implementation**:
   - Copy [`module_template.hpp`](module_template.hpp) into `include/` under `namespace g2g::carbon_forge::<domain>` and expose your `extern "C" CARBON_EXPORT` function in `src/`.
3. **Bind via Python `ctypes`**:
   - Use [`ctypes_bridge_template.py`](ctypes_bridge_template.py) to call your compiled shared library (`build/kimodo_carbon.dll` or `build/avx_vnni_carbon.dll`) with zero-copy NumPy/ctypes pointers, and register your module in `build_and_benchmark.py`.

---
**Crafted with ∞ Meraki by Gaming2Gamers (G2G)**
