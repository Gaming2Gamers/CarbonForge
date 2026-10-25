#ifndef SKIN_COLLIDERS_HPP
#define SKIN_COLLIDERS_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Analytical Signed Distance Fields & 64-Primitive Humanoid Ragdoll Synthesizer
 * File: include/skin_colliders.hpp
 * ============================================================================
 */

#include "carbon_interop.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>

namespace g2g::carbon_forge::colliders {

enum class ColliderType : int32_t {
    Cuboid = 0,
    Ball = 1,
    Cylinder = 2,
    Capsule = 3
};

enum class CoordinateFrame : int32_t {
    ThreeJs = 0,    // Right-handed Y-up
    Ros2Rep103 = 1  // Right-handed Z-up
};

struct Vec3 {
    f32 x{0.0f};
    f32 y{0.0f};
    f32 z{0.0f};

    static constexpr Vec3 Zero() noexcept { return {0.0f, 0.0f, 0.0f}; }
    static constexpr Vec3 Make(f32 x, f32 y, f32 z) noexcept { return {x, y, z}; }

    [[nodiscard]] f32 LengthSq() const noexcept { return x * x + y * y + z * z; }
    [[nodiscard]] f32 Length() const noexcept { return std::sqrt(LengthSq()); }

    [[nodiscard]] Vec3 Normalized() const noexcept {
        f32 l = Length();
        return (l > 1.0e-7f) ? Vec3{x / l, y / l, z / l} : Zero();
    }
};

struct ColliderPrimitive {
    std::string name;
    ColliderType type{ColliderType::Capsule};
    Vec3 position{Vec3::Zero()};
    Quaternion orientation{Quaternion::Identity()};
    f32 radius{0.05f};
    f32 half_height{0.10f};
    Vec3 half_extents{Vec3::Zero()};
    f32 density{1000.0f}; // kg/m^3
};

class AnalyticalSDF {
public:
    static f32 SphereSDF(const Vec3& p, f32 r) noexcept {
        return p.Length() - r;
    }

    static f32 BoxSDF(const Vec3& p, const Vec3& b) noexcept {
        f32 dx = std::abs(p.x) - b.x;
        f32 dy = std::abs(p.y) - b.y;
        f32 dz = std::abs(p.z) - b.z;
        f32 ox = std::max(dx, 0.0f);
        f32 oy = std::max(dy, 0.0f);
        f32 oz = std::max(dz, 0.0f);
        f32 outside = std::sqrt(ox * ox + oy * oy + oz * oz);
        f32 inside = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
        return outside + inside;
    }

    static f32 CapsuleSDF(const Vec3& p, f32 h, f32 r) noexcept {
        f32 py = p.y - std::clamp(p.y, -h, h);
        f32 dist = std::sqrt(p.x * p.x + py * py + p.z * p.z);
        return dist - r;
    }

    static f32 CylinderSDF(const Vec3& p, f32 h, f32 r) noexcept {
        f32 d_radial = std::sqrt(p.x * p.x + p.z * p.z) - r;
        f32 d_axial = std::abs(p.y) - h;
        f32 ox = std::max(d_radial, 0.0f);
        f32 oy = std::max(d_axial, 0.0f);
        f32 outside = std::sqrt(ox * ox + oy * oy);
        f32 inside = std::min(std::max(d_radial, d_axial), 0.0f);
        return outside + inside;
    }

    static f32 PrimitiveSDF(const ColliderPrimitive& prim, const Vec3& world_p) noexcept {
        Vec3 rel{world_p.x - prim.position.x, world_p.y - prim.position.y, world_p.z - prim.position.z};
        auto inv_rot = prim.orientation.Conjugate();
        auto local_arr = inv_rot.RotateVector({rel.x, rel.y, rel.z});
        Vec3 local_p{local_arr[0], local_arr[1], local_arr[2]};

        switch (prim.type) {
            case ColliderType::Ball:
                return SphereSDF(local_p, prim.radius);
            case ColliderType::Cuboid:
                return BoxSDF(local_p, prim.half_extents);
            case ColliderType::Capsule:
                return CapsuleSDF(local_p, prim.half_height, prim.radius);
            case ColliderType::Cylinder:
                return CylinderSDF(local_p, prim.half_height, prim.radius);
            default:
                return 1.0e6f;
        }
    }

    static Vec3 ComputeNormalGradient(const ColliderPrimitive& prim, const Vec3& world_p, f32 eps = 1.0e-4f) noexcept {
        f32 dx = PrimitiveSDF(prim, {world_p.x + eps, world_p.y, world_p.z}) -
                 PrimitiveSDF(prim, {world_p.x - eps, world_p.y, world_p.z});
        f32 dy = PrimitiveSDF(prim, {world_p.x, world_p.y + eps, world_p.z}) -
                 PrimitiveSDF(prim, {world_p.x, world_p.y - eps, world_p.z});
        f32 dz = PrimitiveSDF(prim, {world_p.x, world_p.y, world_p.z + eps}) -
                 PrimitiveSDF(prim, {world_p.x, world_p.y, world_p.z - eps});
        return Vec3{dx, dy, dz}.Normalized();
    }
};

class HumanoidRagdollSynthesizer {
public:
    static std::vector<ColliderPrimitive> SynthesizeHumanoid64() {
        std::vector<ColliderPrimitive> prims;
        prims.reserve(64);

        prims.push_back({"head", ColliderType::Ball, {0.0f, 1.62f, 0.0f}, Quaternion::Identity(), 0.095f, 0.0f, Vec3::Zero(), 980.0f});
        prims.push_back({"neck", ColliderType::Capsule, {0.0f, 1.50f, 0.0f}, Quaternion::Identity(), 0.045f, 0.035f, Vec3::Zero(), 1020.0f});
        prims.push_back({"chest", ColliderType::Capsule, {0.0f, 1.34f, 0.0f}, Quaternion::Identity(), 0.125f, 0.080f, Vec3::Zero(), 950.0f});
        prims.push_back({"spine", ColliderType::Capsule, {0.0f, 1.18f, 0.0f}, Quaternion::Identity(), 0.115f, 0.065f, Vec3::Zero(), 1000.0f});
        prims.push_back({"pelvis", ColliderType::Capsule, {0.0f, 1.02f, 0.0f}, Quaternion::Identity(), 0.130f, 0.070f, Vec3::Zero(), 1050.0f});

        prims.push_back({"left_upper_arm", ColliderType::Capsule, {-0.18f, 1.35f, 0.0f}, Quaternion::Identity(), 0.040f, 0.110f, Vec3::Zero(), 1020.0f});
        prims.push_back({"right_upper_arm", ColliderType::Capsule, {0.18f, 1.35f, 0.0f}, Quaternion::Identity(), 0.040f, 0.110f, Vec3::Zero(), 1020.0f});

        prims.push_back({"left_forearm", ColliderType::Capsule, {-0.24f, 1.12f, 0.0f}, Quaternion::Identity(), 0.035f, 0.100f, Vec3::Zero(), 1040.0f});
        prims.push_back({"right_forearm", ColliderType::Capsule, {0.24f, 1.12f, 0.0f}, Quaternion::Identity(), 0.035f, 0.100f, Vec3::Zero(), 1040.0f});

        prims.push_back({"left_palm", ColliderType::Cuboid, {-0.28f, 0.94f, 0.0f}, Quaternion::Identity(), 0.0f, 0.0f, {0.038f, 0.015f, 0.042f}, 1050.0f});
        prims.push_back({"right_palm", ColliderType::Cuboid, {0.28f, 0.94f, 0.0f}, Quaternion::Identity(), 0.0f, 0.0f, {0.038f, 0.015f, 0.042f}, 1050.0f});

        prims.push_back({"left_thigh", ColliderType::Capsule, {-0.09f, 0.78f, 0.0f}, Quaternion::Identity(), 0.075f, 0.170f, Vec3::Zero(), 1050.0f});
        prims.push_back({"right_thigh", ColliderType::Capsule, {0.09f, 0.78f, 0.0f}, Quaternion::Identity(), 0.075f, 0.170f, Vec3::Zero(), 1050.0f});

        prims.push_back({"left_calf", ColliderType::Capsule, {-0.09f, 0.42f, 0.0f}, Quaternion::Identity(), 0.055f, 0.160f, Vec3::Zero(), 1060.0f});
        prims.push_back({"right_calf", ColliderType::Capsule, {0.09f, 0.42f, 0.0f}, Quaternion::Identity(), 0.055f, 0.160f, Vec3::Zero(), 1060.0f});

        prims.push_back({"left_foot", ColliderType::Cuboid, {-0.09f, 0.05f, 0.04f}, Quaternion::Identity(), 0.0f, 0.0f, {0.042f, 0.035f, 0.110f}, 1080.0f});
        prims.push_back({"right_foot", ColliderType::Cuboid, {0.09f, 0.05f, 0.04f}, Quaternion::Identity(), 0.0f, 0.0f, {0.042f, 0.035f, 0.110f}, 1080.0f});

        const char* finger_names[5] = {"thumb", "index", "middle", "ring", "little"};
        for (int hand = 0; hand < 2; ++hand) {
            std::string side = (hand == 0) ? "left" : "right";
            f32 hand_x = (hand == 0) ? -0.28f : 0.28f;
            for (int f = 0; f < 5; ++f) {
                for (int p = 0; p < 3; ++p) {
                    std::string p_name = side + "_" + finger_names[f] + "_phalanx_" + std::to_string(p + 1);
                    f32 p_y = 0.90f - (f * 0.012f) - (p * 0.015f);
                    f32 r = 0.007f - (p * 0.001f);
                    prims.push_back({p_name, ColliderType::Capsule, {hand_x, p_y, 0.0f}, Quaternion::Identity(), r, 0.012f, Vec3::Zero(), 1050.0f});
                }
            }
        }

        while (prims.size() < 64) {
            size_t idx = prims.size();
            std::string name = "volume_filler_" + std::to_string(idx);
            prims.push_back({name, ColliderType::Capsule, {0.0f, 1.0f, 0.0f}, Quaternion::Identity(), 0.030f, 0.040f, Vec3::Zero(), 1000.0f});
        }

        return prims;
    }

    static std::vector<ColliderPrimitive> SynthesizeSkirtRing(f32 waist_height = 0.95f, f32 skirt_length = 0.40f) {
        std::vector<ColliderPrimitive> rings;
        rings.reserve(5);
        for (int i = 0; i < 5; ++i) {
            f32 frac = static_cast<f32>(i) / 4.0f;
            f32 y = waist_height - (frac * skirt_length);
            f32 r = 0.14f + (frac * 0.16f);
            std::string name = "skirt_ring_tier_" + std::to_string(i + 1);
            rings.push_back({name, ColliderType::Capsule, {0.0f, y, 0.0f}, Quaternion::Identity(), r, 0.025f, Vec3::Zero(), 200.0f});
        }
        return rings;
    }

    static std::string ExportRapierJson(const std::vector<ColliderPrimitive>& prims) {
        std::ostringstream ss;
        ss << "[\n";
        for (size_t i = 0; i < prims.size(); ++i) {
            const auto& p = prims[i];
            ss << "  {\"name\": \"" << p.name << "\", \"type\": " << static_cast<int>(p.type)
               << ", \"position\": [" << p.position.x << "," << p.position.y << "," << p.position.z << "]"
               << ", \"radius\": " << p.radius << ", \"half_height\": " << p.half_height << "}"
               << (i + 1 < prims.size() ? ",\n" : "\n");
        }
        ss << "]\n";
        return ss.str();
    }

    static std::string ExportMujocoMjcf(const std::vector<ColliderPrimitive>& prims) {
        std::ostringstream ss;
        ss << "<mujoco model=\"humanoid_colliders\">\n  <worldbody>\n";
        for (const auto& p : prims) {
            std::string geom_type = (p.type == ColliderType::Ball) ? "sphere" :
                                    (p.type == ColliderType::Cuboid) ? "box" : "capsule";
            ss << "    <geom name=\"" << p.name << "\" type=\"" << geom_type
               << "\" pos=\"" << p.position.x << " " << p.position.y << " " << p.position.z << "\"";
            if (p.type == ColliderType::Ball) {
                ss << " size=\"" << p.radius << "\"";
            } else if (p.type == ColliderType::Cuboid) {
                ss << " size=\"" << p.half_extents.x << " " << p.half_extents.y << " " << p.half_extents.z << "\"";
            } else {
                ss << " size=\"" << p.radius << " " << p.half_height << "\"";
            }
            ss << "/>\n";
        }
        ss << "  </worldbody>\n</mujoco>\n";
        return ss.str();
    }

    static std::string ExportRos2Urdf(const std::vector<ColliderPrimitive>& prims) {
        std::ostringstream ss;
        ss << "<?xml version=\"1.0\"?>\n<robot name=\"humanoid_colliders\">\n";
        for (const auto& p : prims) {
            ss << "  <link name=\"" << p.name << "_link\">\n    <collision>\n      <origin xyz=\""
               << p.position.x << " " << p.position.y << " " << p.position.z << "\"/>\n      <geometry>\n";
            if (p.type == ColliderType::Ball) {
                ss << "        <sphere radius=\"" << p.radius << "\"/>\n";
            } else if (p.type == ColliderType::Cuboid) {
                ss << "        <box size=\"" << (2.0f * p.half_extents.x) << " "
                   << (2.0f * p.half_extents.y) << " " << (2.0f * p.half_extents.z) << "\"/>\n";
            } else {
                ss << "        <cylinder radius=\"" << p.radius << "\" length=\"" << (2.0f * p.half_height) << "\"/>\n";
            }
            ss << "      </geometry>\n    </collision>\n  </link>\n";
        }
        ss << "</robot>\n";
        return ss.str();
    }
};

} // namespace g2g::carbon_forge::colliders

#endif // SKIN_COLLIDERS_HPP
