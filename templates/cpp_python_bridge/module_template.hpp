#ifndef MODULE_TEMPLATE_HPP
#define MODULE_TEMPLATE_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Starter C++20 Lowering Header Template
 * File: templates/cpp_python_bridge/module_template.hpp
 * ============================================================================
 */

#include "../../include/carbon_interop.hpp"
#include <cmath>

namespace g2g::carbon_forge::custom {

class CustomComputeKernel {
public:
    static inline f32 ComputeScaledNorm(f32 x, f32 y, f32 z, f32 scale_factor) noexcept {
        return scale_factor * std::sqrt(x * x + y * y + z * z);
    }
};

} // namespace g2g::carbon_forge::custom

#endif // MODULE_TEMPLATE_HPP
