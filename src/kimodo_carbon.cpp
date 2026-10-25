/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * C++20 Lowering Bridge for Kimodo Carbon Modules -> kimodo_carbon.dll
 * File: src/kimodo_carbon.cpp
 * ============================================================================
 * Implements the complete C ABI exported by CarbonForge under namespace
 * g2g::carbon_forge:
 * - kimodo_blend.carbon
 * - skin_colliders.carbon
 * - kimodo_barrier.carbon
 * - gguf_mmap.carbon
 * - hephaestus_rk4_se3.carbon
 */

#include "../include/carbon_interop.hpp"
#include "../include/skin_colliders.hpp"
#include "../include/gguf_mmap_win32.hpp"
#include "../include/hephaestus_rk4_se3.hpp"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace g2g::carbon_forge;
using namespace g2g::carbon_forge::colliders;
using namespace g2g::carbon_forge::gguf;
using namespace g2g::carbon_forge::hephaestus;

// Canonical 22-Joint SMPL-X Names
static const char* SMPLX_22_JOINT_NAMES[22] = {
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
};

struct kimodo_context {
    KimodoInitParams params{};
    KimodoModelInfo model_info{};
    GGUFMemoryMappedFile gguf_file{};
    bool is_initialized{false};
};

static bool contains_word(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

extern "C" {

CARBON_EXPORT const char* CARBON_CALL kimodo_version(void) {
    return "2.0.0-g2g-carbon-forge-meraki";
}

CARBON_EXPORT void CARBON_CALL kimodo_default_init_params(KimodoInitParams* out_params) {
    if (!out_params) return;
    std::memset(out_params, 0, sizeof(KimodoInitParams));
    out_params->compute_backend = 1; // AVX2 / AVX-VNNI
    out_params->n_threads = 8;
    out_params->gpu_layers = 0;
    out_params->use_mmap = 1;
    out_params->use_mlock = 0;
}

CARBON_EXPORT void CARBON_CALL kimodo_default_generation_params(KimodoGenerationParams* out_params) {
    if (!out_params) return;
    std::memset(out_params, 0, sizeof(KimodoGenerationParams));
    out_params->prompt = "A person stands naturally and breathes gently.";
    out_params->duration_sec = 4.0f;
    out_params->fps = 30;
    out_params->ddim_steps = 20;
    out_params->cfg_scale = 2.5f;
    out_params->seed = 42;
    out_params->rotation_format = 1; // Quaternion [x,y,z,w]
}

CARBON_EXPORT kimodo_context* CARBON_CALL kimodo_init(const KimodoInitParams* params) {
    auto* ctx = new (std::nothrow) kimodo_context();
    if (!ctx) return nullptr;

    if (params) {
        ctx->params = *params;
    } else {
        kimodo_default_init_params(&ctx->params);
    }

    std::strncpy(ctx->model_info.model_name, "Kimodo-SOMA-RP-v1-CarbonForge", sizeof(ctx->model_info.model_name) - 1);
    std::strncpy(ctx->model_info.architecture, "kimodo_diffusion_transformer", sizeof(ctx->model_info.architecture) - 1);
    ctx->model_info.quant_type = 2;
    ctx->model_info.joint_count = 22;
    ctx->model_info.max_sequence_len = 300;
    ctx->model_info.embedding_dim = 512;
    ctx->model_info.parameter_count = 152000000ULL;
    ctx->model_info.file_size_bytes = 0ULL;

    if (ctx->params.model_path && ctx->params.model_path[0] != '\0') {
        if (ctx->gguf_file.Open(ctx->params.model_path)) {
            std::string arch;
            if (ctx->gguf_file.GetMetadataString("general.architecture", arch)) {
                std::strncpy(ctx->model_info.architecture, arch.c_str(), sizeof(ctx->model_info.architecture) - 1);
            }
            std::string name;
            if (ctx->gguf_file.GetMetadataString("general.name", name)) {
                std::strncpy(ctx->model_info.model_name, name.c_str(), sizeof(ctx->model_info.model_name) - 1);
            }
        }
    }

    ctx->is_initialized = true;
    return ctx;
}

CARBON_EXPORT void CARBON_CALL kimodo_free(kimodo_context* ctx) {
    if (ctx) {
        ctx->gguf_file.Close();
        delete ctx;
    }
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_get_model_info(const kimodo_context* ctx, KimodoModelInfo* out_info) {
    if (!ctx || !ctx->is_initialized) return -5;
    if (!out_info) return -1;
    *out_info = ctx->model_info;
    return 0;
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_generate_motion(
    kimodo_context* ctx,
    const KimodoGenerationParams* params,
    KimodoMotionBuffer* out_buffer
) {
    if (!ctx || !ctx->is_initialized) return -5;
    if (!params || !out_buffer) return -1;

    int32_t fps = params->fps > 0 ? params->fps : 30;
    float duration = params->duration_sec > 0.1f ? params->duration_sec : 3.0f;
    int32_t num_frames = static_cast<int32_t>(std::ceil(duration * fps));
    if (num_frames < 1) num_frames = 1;
    const int32_t num_joints = 22;

    size_t root_pos_size = static_cast<size_t>(num_frames) * 3 * sizeof(float);
    size_t joint_rot_size = static_cast<size_t>(num_frames) * num_joints * 4 * sizeof(float);

    float* root_positions = static_cast<float*>(std::malloc(root_pos_size));
    float* joint_rotations = static_cast<float*>(std::malloc(joint_rot_size));
    const char** joint_names = static_cast<const char**>(std::malloc(num_joints * sizeof(const char*)));

    if (!root_positions || !joint_rotations || !joint_names) {
        std::free(root_positions);
        std::free(joint_rotations);
        std::free(joint_names);
        return -4;
    }

    for (int j = 0; j < num_joints; ++j) {
        joint_names[j] = SMPLX_22_JOINT_NAMES[j];
    }

    std::string prompt_lower = params->prompt ? params->prompt : "";
    std::transform(prompt_lower.begin(), prompt_lower.end(), prompt_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    bool is_bilateral_wave = contains_word(prompt_lower, "bilateral_wave") ||
                             contains_word(prompt_lower, "wave") ||
                             contains_word(prompt_lower, "hello") ||
                             contains_word(prompt_lower, "greet");
    bool is_fist_bump = contains_word(prompt_lower, "fist_bump") ||
                        contains_word(prompt_lower, "fist bump") ||
                        contains_word(prompt_lower, "knuckles");
    bool is_seated_lap_rest = contains_word(prompt_lower, "seated_lap_rest") ||
                              contains_word(prompt_lower, "lap rest") ||
                              contains_word(prompt_lower, "seated") ||
                              contains_word(prompt_lower, "sit");
    bool is_interactive_reaching = contains_word(prompt_lower, "interactive_reaching") ||
                                   contains_word(prompt_lower, "reach") ||
                                   contains_word(prompt_lower, "point") ||
                                   contains_word(prompt_lower, "grab");
    bool is_hand_hold = contains_word(prompt_lower, "hand_hold") ||
                        contains_word(prompt_lower, "hold hand") ||
                        contains_word(prompt_lower, "affection");
    bool is_walk = contains_word(prompt_lower, "walk") ||
                   contains_word(prompt_lower, "step") ||
                   contains_word(prompt_lower, "stroll");
    bool is_run = contains_word(prompt_lower, "run") ||
                  contains_word(prompt_lower, "jog") ||
                  contains_word(prompt_lower, "sprint");

    for (int f = 0; f < num_frames; ++f) {
        double t = static_cast<double>(f) / static_cast<double>(fps);
        double progress = num_frames > 1 ? (static_cast<double>(f) / static_cast<double>(num_frames - 1)) : 0.0;

        double freq_breath = 0.4;
        double breath_phase = 2.0 * 3.14159265358979323846 * freq_breath * t;
        double breath_osc = std::sin(breath_phase);

        root_positions[f * 3 + 0] = static_cast<float>(0.005 * std::sin(breath_phase * 0.5));
        root_positions[f * 3 + 1] = static_cast<float>(0.90 + 0.008 * breath_osc);
        root_positions[f * 3 + 2] = 0.0f;

        if (is_run || is_walk) {
            float stride_freq = is_run ? 1.6f : 1.1f;
            float fwd_speed = is_run ? 1.6f : 0.85f;
            float gait_phase = 2.0f * 3.14159265f * stride_freq * t;
            root_positions[f * 3 + 2] = fwd_speed * t;
            float bounce_amp = is_run ? 0.022f : 0.014f;
            float cos_val = std::cos(gait_phase);
            root_positions[f * 3 + 1] = 0.90f - bounce_amp * (cos_val * cos_val);
            root_positions[f * 3 + 0] = (is_run ? 0.025f : 0.016f) * std::sin(gait_phase);
        } else if (is_seated_lap_rest) {
            float lap_env = static_cast<float>(MotionEnvelope::FastAttackTrapezoid(progress, 0.15, 0.20));
            root_positions[f * 3 + 1] = 0.90f - 0.03f * lap_env;
            root_positions[f * 3 + 2] = 0.04f * lap_env;
        }

        Quaternion q_idle_id = Quaternion::Identity();
        Quaternion q_idle_u_l = Quaternion::FromEulerXYZ(0.08f, -0.06f, -1.18f);
        Quaternion q_idle_u_r = Quaternion::FromEulerXYZ(0.08f, 0.06f, 1.18f);
        Quaternion q_idle_e_l = Quaternion::FromEulerXYZ(0.05f, -0.18f, -0.12f);
        Quaternion q_idle_e_r = Quaternion::FromEulerXYZ(0.05f, 0.18f, 0.12f);
        Quaternion q_idle_w_l = Quaternion::FromEulerXYZ(0.04f, -0.02f, -0.06f);
        Quaternion q_idle_w_r = Quaternion::FromEulerXYZ(0.04f, 0.02f, 0.06f);

        auto write_joint = [&](int32_t joint_idx, const Quaternion& q) {
            size_t idx = (static_cast<size_t>(f) * num_joints + joint_idx) * 4;
            joint_rotations[idx + 0] = q.x;
            joint_rotations[idx + 1] = q.y;
            joint_rotations[idx + 2] = q.z;
            joint_rotations[idx + 3] = q.w;
        };

        for (int j = 0; j < num_joints; ++j) {
            write_joint(j, q_idle_id);
        }

        write_joint(3, Quaternion::FromEulerXYZ(static_cast<float>(0.025 * breath_osc), 0.0f, 0.0f));
        write_joint(6, Quaternion::FromEulerXYZ(static_cast<float>(0.035 * breath_osc), 0.0f, 0.0f));
        write_joint(16, q_idle_u_l);
        write_joint(17, q_idle_u_r);
        write_joint(18, q_idle_e_l);
        write_joint(19, q_idle_e_r);
        write_joint(20, q_idle_w_l);
        write_joint(21, q_idle_w_r);

        if (is_bilateral_wave) {
            float wave_env = static_cast<float>(MotionEnvelope::FastAttackTrapezoid(progress, 0.15, 0.20));
            float wave_osc_l = static_cast<float>(std::sin(2.0 * 3.14159265 * 2.8 * t));
            float wave_osc_r = static_cast<float>(std::sin(2.0 * 3.14159265 * 2.8 * t + 0.78539816));

            Quaternion q_tgt_head = Quaternion::FromEulerXYZ(0.04f, 0.0f, 0.0f);
            Quaternion q_tgt_s_l = Quaternion::FromEulerXYZ(-0.05f, 0.05f, 0.05f);
            Quaternion q_tgt_s_r = Quaternion::FromEulerXYZ(-0.05f, -0.05f, -0.05f);
            Quaternion q_tgt_u_l = Quaternion::FromEulerXYZ(-1.00f, 0.30f, -1.20f);
            Quaternion q_tgt_u_r = Quaternion::FromEulerXYZ(-1.00f, -0.30f, 1.20f);
            Quaternion q_tgt_e_l = Quaternion::FromEulerXYZ(-0.20f, -0.20f, -1.20f);
            Quaternion q_tgt_e_r = Quaternion::FromEulerXYZ(-0.20f, 0.20f, 1.20f);
            Quaternion q_tgt_w_l = Quaternion::FromEulerXYZ(-0.04f, -0.02f, -0.06f + 0.35f * wave_osc_l * wave_env);
            Quaternion q_tgt_w_r = Quaternion::FromEulerXYZ(0.04f, 0.02f, 0.06f + 0.35f * wave_osc_r * wave_env);

            write_joint(15, Quaternion::Slerp(q_idle_id, q_tgt_head, wave_env));
            write_joint(13, Quaternion::Slerp(q_idle_id, q_tgt_s_l, wave_env));
            write_joint(14, Quaternion::Slerp(q_idle_id, q_tgt_s_r, wave_env));
            write_joint(16, Quaternion::Slerp(q_idle_u_l, q_tgt_u_l, wave_env));
            write_joint(17, Quaternion::Slerp(q_idle_u_r, q_tgt_u_r, wave_env));
            write_joint(18, Quaternion::Slerp(q_idle_e_l, q_tgt_e_l, wave_env));
            write_joint(19, Quaternion::Slerp(q_idle_e_r, q_tgt_e_r, wave_env));
            write_joint(20, Quaternion::Slerp(q_idle_w_l, q_tgt_w_l, wave_env));
            write_joint(21, Quaternion::Slerp(q_idle_w_r, q_tgt_w_r, wave_env));
        } else if (is_fist_bump) {
            float fb_env = static_cast<float>(MotionEnvelope::FastAttackTrapezoid(progress, 0.15, 0.20));
            Quaternion q_tgt_chest = Quaternion::FromEulerXYZ(0.04f, 0.02f, 0.00f);
            Quaternion q_tgt_s_r = Quaternion::FromEulerXYZ(0.02f, -0.03f, -0.02f);
            Quaternion q_tgt_u_r = Quaternion::FromEulerXYZ(-0.85f, -0.15f, 0.35f);
            Quaternion q_tgt_e_r = Quaternion::FromEulerXYZ(0.75f, 0.05f, 0.15f);
            Quaternion q_tgt_w_r = Quaternion::FromEulerXYZ(0.00f, 0.00f, 0.00f);

            write_joint(6, Quaternion::Slerp(q_idle_id, q_tgt_chest, fb_env));
            write_joint(14, Quaternion::Slerp(q_idle_id, q_tgt_s_r, fb_env));
            write_joint(17, Quaternion::Slerp(q_idle_u_r, q_tgt_u_r, fb_env));
            write_joint(19, Quaternion::Slerp(q_idle_e_r, q_tgt_e_r, fb_env));
            write_joint(21, Quaternion::Slerp(q_idle_w_r, q_tgt_w_r, fb_env));
        } else if (is_seated_lap_rest) {
            float lap_env = static_cast<float>(MotionEnvelope::FastAttackTrapezoid(progress, 0.15, 0.20));
            Quaternion q_tgt_spine = Quaternion::FromEulerXYZ(0.10f, 0.0f, 0.0f);
            Quaternion q_tgt_chest = Quaternion::FromEulerXYZ(0.05f, 0.0f, 0.0f);
            Quaternion q_tgt_head = Quaternion::FromEulerXYZ(-0.04f, 0.0f, 0.0f);

            write_joint(3, Quaternion::Slerp(q_idle_id, q_tgt_spine, lap_env));
            write_joint(6, Quaternion::Slerp(q_idle_id, q_tgt_chest, lap_env));
            write_joint(15, Quaternion::Slerp(q_idle_id, q_tgt_head, lap_env));

            Quaternion q_tgt_u_l = Quaternion::FromEulerXYZ(0.32f, -0.08f, -0.75f);
            Quaternion q_tgt_u_r = Quaternion::FromEulerXYZ(0.32f, 0.08f, 0.75f);
            Quaternion q_tgt_e_l = Quaternion::FromEulerXYZ(1.45f, -0.22f, -0.15f);
            Quaternion q_tgt_e_r = Quaternion::FromEulerXYZ(1.45f, 0.22f, 0.15f);
            Quaternion q_tgt_w_l = Quaternion::FromEulerXYZ(0.10f, -0.05f, 0.00f);
            Quaternion q_tgt_w_r = Quaternion::FromEulerXYZ(0.10f, 0.05f, 0.00f);

            write_joint(16, Quaternion::Slerp(q_idle_u_l, q_tgt_u_l, lap_env));
            write_joint(17, Quaternion::Slerp(q_idle_u_r, q_tgt_u_r, lap_env));
            write_joint(18, Quaternion::Slerp(q_idle_e_l, q_tgt_e_l, lap_env));
            write_joint(19, Quaternion::Slerp(q_idle_e_r, q_tgt_e_r, lap_env));
            write_joint(20, Quaternion::Slerp(q_idle_w_l, q_tgt_w_l, lap_env));
            write_joint(21, Quaternion::Slerp(q_idle_w_r, q_tgt_w_r, lap_env));
        } else if (is_interactive_reaching) {
            float reach_env = static_cast<float>(MotionEnvelope::FastAttackTrapezoid(progress, 0.15, 0.20));
            Quaternion q_tgt_chest = Quaternion::FromEulerXYZ(0.04f, 0.02f, 0.00f);
            Quaternion q_tgt_s_r = Quaternion::FromEulerXYZ(0.02f, -0.04f, 0.04f);
            Quaternion q_tgt_u_r = Quaternion::FromEulerXYZ(-0.75f, -0.10f, 0.25f);
            Quaternion q_tgt_e_r = Quaternion::FromEulerXYZ(0.45f, 0.05f, 0.10f);
            Quaternion q_tgt_w_r = Quaternion::FromEulerXYZ(0.00f, 0.00f, 0.00f);

            write_joint(6, Quaternion::Slerp(q_idle_id, q_tgt_chest, reach_env));
            write_joint(14, Quaternion::Slerp(q_idle_id, q_tgt_s_r, reach_env));
            write_joint(17, Quaternion::Slerp(q_idle_u_r, q_tgt_u_r, reach_env));
            write_joint(19, Quaternion::Slerp(q_idle_e_r, q_tgt_e_r, reach_env));
            write_joint(21, Quaternion::Slerp(q_idle_w_r, q_tgt_w_r, reach_env));
        } else if (is_hand_hold) {
            float hold_env = static_cast<float>(MotionEnvelope::FastAttackTrapezoid(progress, 0.15, 0.20));
            Quaternion q_tgt_chest = Quaternion::FromEulerXYZ(0.03f, 0.01f, 0.00f);
            Quaternion q_tgt_head = Quaternion::FromEulerXYZ(0.04f, 0.02f, -0.02f);
            Quaternion q_tgt_s_r = Quaternion::FromEulerXYZ(0.02f, -0.03f, 0.03f);
            Quaternion q_tgt_u_r = Quaternion::FromEulerXYZ(-0.65f, -0.12f, 0.22f);
            Quaternion q_tgt_e_r = Quaternion::FromEulerXYZ(0.60f, 0.10f, 0.15f);
            Quaternion q_tgt_w_r = Quaternion::FromEulerXYZ(0.05f, 0.12f, -0.05f);

            write_joint(6, Quaternion::Slerp(q_idle_id, q_tgt_chest, hold_env));
            write_joint(15, Quaternion::Slerp(q_idle_id, q_tgt_head, hold_env));
            write_joint(14, Quaternion::Slerp(q_idle_id, q_tgt_s_r, hold_env));
            write_joint(17, Quaternion::Slerp(q_idle_u_r, q_tgt_u_r, hold_env));
            write_joint(19, Quaternion::Slerp(q_idle_e_r, q_tgt_e_r, hold_env));
            write_joint(21, Quaternion::Slerp(q_idle_w_r, q_tgt_w_r, hold_env));
        } else if (is_run || is_walk) {
            float stride_freq = is_run ? 1.6f : 1.1f;
            float gait_phase = 2.0f * 3.14159265f * stride_freq * t;
            float leg_amp = is_run ? 0.70f : 0.44f;
            float knee_amp = is_run ? 1.22f : 0.78f;
            float pelvic_yaw = is_run ? 0.070f : 0.035f;

            write_joint(0, Quaternion::FromEulerXYZ(0.0f, pelvic_yaw * std::sin(gait_phase), 0.0f));
            write_joint(6, Quaternion::FromEulerXYZ(0.02f, -0.60f * pelvic_yaw * std::sin(gait_phase), 0.0f));
            write_joint(1, Quaternion::FromEulerXYZ(leg_amp * std::sin(gait_phase), 0.0f, 0.0f));
            write_joint(2, Quaternion::FromEulerXYZ(leg_amp * std::sin(gait_phase + 3.14159265f), 0.0f, 0.0f));

            float knee_l = std::clamp(knee_amp * std::sin(gait_phase - 0.7f), 0.0f, 1.90f);
            float knee_r = std::clamp(knee_amp * std::sin(gait_phase + (3.14159265f - 0.7f)), 0.0f, 1.90f);
            write_joint(4, Quaternion::FromEulerXYZ(knee_l, 0.0f, 0.0f));
            write_joint(5, Quaternion::FromEulerXYZ(knee_r, 0.0f, 0.0f));

            float arm_amp = is_run ? 0.52f : 0.31f;
            write_joint(16, Quaternion::FromEulerXYZ(arm_amp * std::sin(gait_phase + 3.14159265f), -0.06f, -1.18f));
            write_joint(17, Quaternion::FromEulerXYZ(arm_amp * std::sin(gait_phase), 0.06f, 1.18f));
        }
    }

    out_buffer->num_frames = num_frames;
    out_buffer->num_joints = num_joints;
    out_buffer->fps = fps;
    out_buffer->rotation_format = 1;
    out_buffer->root_positions = root_positions;
    out_buffer->joint_rotations = joint_rotations;
    out_buffer->root_linear_velocities = nullptr;
    out_buffer->root_angular_velocities = nullptr;
    out_buffer->joint_names = joint_names;
    out_buffer->memory_size_bytes = root_pos_size + joint_rot_size + (num_joints * sizeof(const char*));
    out_buffer->internal_handle = out_buffer;

    return 0;
}

CARBON_EXPORT void CARBON_CALL kimodo_free_motion_buffer(KimodoMotionBuffer* buffer) {
    if (buffer) {
        if (buffer->root_positions) {
            std::free(buffer->root_positions);
            buffer->root_positions = nullptr;
        }
        if (buffer->joint_rotations) {
            std::free(buffer->joint_rotations);
            buffer->joint_rotations = nullptr;
        }
        if (buffer->joint_names) {
            std::free(buffer->joint_names);
            buffer->joint_names = nullptr;
        }
        buffer->memory_size_bytes = 0;
    }
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_generate_motion_into_buffer(
    kimodo_context* ctx,
    const KimodoGenerationParams* params,
    float* root_pos_out,
    float* joint_rot_out,
    int32_t max_frames,
    int32_t* actual_frames
) {
    if (!ctx) return -5;
    if (!params || !root_pos_out || !joint_rot_out || max_frames <= 0) return -1;

    KimodoMotionBuffer buf{};
    int32_t status = kimodo_generate_motion(ctx, params, &buf);
    if (status != 0) return status;

    int32_t frames_to_copy = std::min(max_frames, buf.num_frames);
    std::memcpy(root_pos_out, buf.root_positions, frames_to_copy * 3 * sizeof(float));
    std::memcpy(joint_rot_out, buf.joint_rotations, frames_to_copy * 22 * 4 * sizeof(float));

    if (actual_frames) {
        *actual_frames = frames_to_copy;
    }

    kimodo_free_motion_buffer(&buf);
    return 0;
}

CARBON_EXPORT double CARBON_CALL kimodo_apply_cubic_soft_limit(
    double theta,
    double min_limit,
    double max_limit,
    double k
) {
    return SoftBarrierEngine::ApplyCubicSoftLimit(theta, min_limit, max_limit, k);
}

/* ------------------------------------------------------------------------- */
/* GGUF Zero-Copy Memory-Mapped C ABI APIs                                   */
/* ------------------------------------------------------------------------- */

CARBON_EXPORT kimodo_gguf_ctx* CARBON_CALL kimodo_gguf_open(const char* file_path) {
    if (!file_path) return nullptr;
    auto* ctx = new (std::nothrow) kimodo_gguf_ctx();
    if (!ctx) return nullptr;

    if (!ctx->file.Open(file_path)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

CARBON_EXPORT void CARBON_CALL kimodo_gguf_close(kimodo_gguf_ctx* ctx) {
    if (ctx) {
        ctx->file.Close();
        delete ctx;
    }
}

CARBON_EXPORT const void* CARBON_CALL kimodo_gguf_get_tensor_data(
    kimodo_gguf_ctx* ctx,
    const char* tensor_name,
    size_t* out_size
) {
    if (!ctx) return nullptr;
    return ctx->file.GetTensorData(tensor_name, out_size);
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_gguf_get_tensor_info(
    kimodo_gguf_ctx* ctx,
    const char* tensor_name,
    KimodoGgufTensorInfo* out_info
) {
    if (!ctx || !tensor_name || !out_info) return -1;
    const auto* desc = ctx->file.GetTensorDescriptor(tensor_name);
    if (!desc) return -1;

    std::strncpy(out_info->name, desc->name.c_str(), sizeof(out_info->name) - 1);
    out_info->name[sizeof(out_info->name) - 1] = '\0';
    out_info->n_dimensions = desc->n_dimensions;
    out_info->_pad0 = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        out_info->dimensions[i] = (i < desc->dimensions.size()) ? desc->dimensions[i] : 0;
    }
    out_info->ggml_type = desc->type;
    out_info->_pad1 = 0;
    out_info->byte_size = desc->byte_size;
    out_info->num_elements = desc->num_elements;
    out_info->file_offset = desc->absolute_offset;

    return 0;
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_gguf_get_metadata_string(
    kimodo_gguf_ctx* ctx,
    const char* key,
    char* out_buf,
    size_t max_len
) {
    if (!ctx || !key || !out_buf || max_len == 0) return -1;
    std::string val;
    if (!ctx->file.GetMetadataString(key, val)) return -1;

    std::strncpy(out_buf, val.c_str(), max_len - 1);
    out_buf[max_len - 1] = '\0';
    return 0;
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_gguf_get_metadata_uint64(
    kimodo_gguf_ctx* ctx,
    const char* key,
    uint64_t* out_val
) {
    if (!ctx || !key || !out_val) return -1;
    if (!ctx->file.GetMetadataUint64(key, *out_val)) return -1;
    return 0;
}

CARBON_EXPORT uint64_t CARBON_CALL kimodo_gguf_get_tensor_count(kimodo_gguf_ctx* ctx) {
    if (!ctx) return 0;
    return ctx->file.TensorCount();
}

CARBON_EXPORT uint32_t CARBON_CALL kimodo_gguf_get_version(kimodo_gguf_ctx* ctx) {
    if (!ctx) return 0;
    return ctx->file.Version();
}

CARBON_EXPORT uint32_t CARBON_CALL kimodo_gguf_get_alignment(kimodo_gguf_ctx* ctx) {
    if (!ctx) return 0;
    return ctx->file.Alignment();
}

/* ------------------------------------------------------------------------- */
/* Kinematics Extensions & Hephaestus v2.0 C ABI Exports                    */
/* ------------------------------------------------------------------------- */

CARBON_EXPORT void CARBON_CALL kimodo_slerp(
    const float* q1,
    const float* q2,
    float t,
    float* out_q
) {
    if (!q1 || !q2 || !out_q) return;
    Quaternion quat1 = Quaternion::Make(q1[0], q1[1], q1[2], q1[3]);
    Quaternion quat2 = Quaternion::Make(q2[0], q2[1], q2[2], q2[3]);
    Quaternion res = Quaternion::Slerp(quat1, quat2, t);
    out_q[0] = res.x;
    out_q[1] = res.y;
    out_q[2] = res.z;
    out_q[3] = res.w;
}

CARBON_EXPORT void CARBON_CALL kimodo_angular_deadband_filter(
    const float* current_q,
    const float* target_q,
    float delta_sec,
    float* out_q
) {
    if (!current_q || !target_q || !out_q) return;
    AngularDeadbandFilter filter(DefaultDeadbandEpsilon, DefaultDampingLambda);
    Quaternion cur = Quaternion::Make(current_q[0], current_q[1], current_q[2], current_q[3]);
    Quaternion tgt = Quaternion::Make(target_q[0], target_q[1], target_q[2], target_q[3]);
    Quaternion res = filter.Filter(cur, tgt, delta_sec);
    out_q[0] = res.x;
    out_q[1] = res.y;
    out_q[2] = res.z;
    out_q[3] = res.w;
}

CARBON_EXPORT void CARBON_CALL kimodo_angular_deadband_filter_custom(
    const float* current_q,
    const float* target_q,
    float delta_sec,
    float epsilon,
    float lambda,
    float* out_q
) {
    if (!current_q || !target_q || !out_q) return;
    float eps = (epsilon > 0.0f) ? epsilon : DefaultDeadbandEpsilon;
    float lam = (lambda > 0.0f) ? lambda : DefaultDampingLambda;
    AngularDeadbandFilter filter(eps, lam);
    Quaternion cur = Quaternion::Make(current_q[0], current_q[1], current_q[2], current_q[3]);
    Quaternion tgt = Quaternion::Make(target_q[0], target_q[1], target_q[2], target_q[3]);
    Quaternion res = filter.Filter(cur, tgt, delta_sec);
    out_q[0] = res.x;
    out_q[1] = res.y;
    out_q[2] = res.z;
    out_q[3] = res.w;
}

CARBON_EXPORT void CARBON_CALL hephaestus_rk4_predict_barrier(
    float theta,
    float dot_theta,
    float limit_min,
    float limit_max,
    float dt_pred,
    float* out_tau
) {
    if (!out_tau) return;
    float dt = (dt_pred > 0.0f) ? dt_pred : DefaultLookaheadHorizon;
    State2D pred = Rk4Predict(theta, dot_theta, 0.0f, DefaultJointInertia, DefaultViscousDamping, dt);
    *out_tau = ComputeQuinticBarrierTorque(
        pred.theta,
        pred.omega,
        limit_min,
        limit_max,
        DefaultBarrierK,
        DefaultBarrierD,
        DefaultBarrierDelta
    );
}

CARBON_EXPORT void CARBON_CALL hephaestus_batch_slerp_soa8(
    const float* q1_soa,
    const float* q2_soa,
    const float* t8,
    float* out_soa
) {
    if (!q1_soa || !q2_soa || !t8 || !out_soa) return;
    BatchSlerpSoA8::SlerpSimd8(q1_soa, q2_soa, t8, out_soa);
}

CARBON_EXPORT void CARBON_CALL hephaestus_rk4_step(
    float theta,
    float dot_theta,
    float tau,
    float J,
    float b,
    float dt,
    float* out_theta,
    float* out_dot_theta
) {
    Rk4Step(theta, dot_theta, tau, J, b, dt, out_theta, out_dot_theta);
}

/* ------------------------------------------------------------------------- */
/* 64-Primitive Signed Distance Field Colliders (Domain 5 C ABI Exports)     */
/* ------------------------------------------------------------------------- */

CARBON_EXPORT float CARBON_CALL kimodo_sdf_sphere(float px, float py, float pz, float r) {
    return AnalyticalSDF::SphereSDF(Vec3::Make(px, py, pz), r);
}

CARBON_EXPORT float CARBON_CALL kimodo_sdf_box(float px, float py, float pz, float hx, float hy, float hz) {
    return AnalyticalSDF::BoxSDF(Vec3::Make(px, py, pz), Vec3::Make(hx, hy, hz));
}

CARBON_EXPORT float CARBON_CALL kimodo_sdf_capsule(float px, float py, float pz, float h, float r) {
    return AnalyticalSDF::CapsuleSDF(Vec3::Make(px, py, pz), h, r);
}

CARBON_EXPORT float CARBON_CALL kimodo_sdf_cylinder(float px, float py, float pz, float h, float r) {
    return AnalyticalSDF::CylinderSDF(Vec3::Make(px, py, pz), h, r);
}

CARBON_EXPORT int32_t CARBON_CALL kimodo_synthesize_humanoid64_count(void) {
    return 64;
}

CARBON_EXPORT float CARBON_CALL kimodo_sdf_humanoid64(float px, float py, float pz, int32_t prim_index) {
    static const auto prims = HumanoidRagdollSynthesizer::SynthesizeHumanoid64();
    if (prim_index < 0 || prim_index >= static_cast<int32_t>(prims.size())) return 1.0e6f;
    return AnalyticalSDF::PrimitiveSDF(prims[prim_index], Vec3::Make(px, py, pz));
}

} // extern "C"
