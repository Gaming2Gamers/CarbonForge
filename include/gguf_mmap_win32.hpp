#ifndef GGUF_MMAP_WIN32_HPP
#define GGUF_MMAP_WIN32_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Zero-Copy Memory-Mapped (mmap) GGUF v2/v3 Engine Header
 * File: include/gguf_mmap_win32.hpp
 * ============================================================================
 */

#include "carbon_interop.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace g2g::carbon_forge::gguf {

constexpr uint32_t GgufMagic = 0x46554747; // "GGUF" in little-endian
constexpr uint32_t GgufDefaultAlignment = 32;

struct TensorDescriptor {
    std::string name;
    uint32_t n_dimensions{0};
    std::vector<uint64_t> dimensions;
    uint32_t type{0};
    uint64_t offset{0};
    uint64_t absolute_offset{0};
    uint64_t byte_size{0};
    uint64_t num_elements{0};
};

static inline uint64_t CalcTensorByteSize(uint32_t ggml_type, uint64_t num_elements) noexcept {
    switch (ggml_type) {
        case 0:  return num_elements * 4;       // F32
        case 1:  return num_elements * 2;       // F16
        case 29: return num_elements * 2;       // BF16
        case 2:  return ((num_elements + 31) / 32) * 18;  // Q4_0
        case 3:  return ((num_elements + 31) / 32) * 20;  // Q4_1
        case 6:  return ((num_elements + 31) / 32) * 22;  // Q5_0
        case 7:  return ((num_elements + 31) / 32) * 24;  // Q5_1
        case 8:  return ((num_elements + 31) / 32) * 34;  // Q8_0
        case 9:  return ((num_elements + 31) / 32) * 36;  // Q8_1
        case 10: return ((num_elements + 255) / 256) * 84; // Q2_K
        case 11: return ((num_elements + 255) / 256) * 110;// Q3_K
        case 12: return ((num_elements + 255) / 256) * 144;// Q4_K
        case 13: return ((num_elements + 255) / 256) * 176;// Q5_K
        case 14: return ((num_elements + 255) / 256) * 210;// Q6_K
        case 15: return ((num_elements + 255) / 256) * 292;// Q8_K
        case 24: return num_elements * 1;       // I8
        case 25: return num_elements * 2;       // I16
        case 26: return num_elements * 4;       // I32
        case 27: return num_elements * 8;       // I64
        case 28: return num_elements * 8;       // F64
        default: return num_elements;
    }
}

class GGUFMemoryMappedFile {
private:
    void* mapped_ptr_{nullptr};
    uint64_t file_size_{0};
    bool is_mapped_{false};

    uint32_t version_{0};
    uint64_t tensor_count_{0};
    uint64_t metadata_kv_count_{0};
    uint32_t alignment_{GgufDefaultAlignment};
    uint64_t tensor_data_base_offset_{0};

    std::unordered_map<std::string, TensorDescriptor> tensors_;
    std::unordered_map<std::string, std::string> metadata_strings_;
    std::unordered_map<std::string, uint64_t> metadata_uint64s_;

#if defined(_WIN32)
    HANDLE file_handle_{INVALID_HANDLE_VALUE};
    HANDLE mapping_handle_{nullptr};
#else
    int fd_{-1};
#endif

public:
    GGUFMemoryMappedFile() = default;
    ~GGUFMemoryMappedFile() { Close(); }

    bool Open(const char* file_path) {
        if (!file_path || file_path[0] == '\0') return false;
        Close();

#if defined(_WIN32)
        int wide_len = MultiByteToWideChar(CP_UTF8, 0, file_path, -1, nullptr, 0);
        if (wide_len <= 0) return false;
        std::vector<wchar_t> wide_path(wide_len);
        MultiByteToWideChar(CP_UTF8, 0, file_path, -1, wide_path.data(), wide_len);

        file_handle_ = CreateFileW(
            wide_path.data(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr
        );
        if (file_handle_ == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER size_val;
        if (!GetFileSizeEx(file_handle_, &size_val)) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
        file_size_ = static_cast<uint64_t>(size_val.QuadPart);
        if (file_size_ < 16) {
            Close();
            return false;
        }

        mapping_handle_ = CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            Close();
            return false;
        }

        mapped_ptr_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
        if (!mapped_ptr_) {
            Close();
            return false;
        }
#else
        fd_ = open(file_path, O_RDONLY);
        if (fd_ < 0) return false;

        struct stat sb;
        if (fstat(fd_, &sb) < 0) {
            close(fd_);
            fd_ = -1;
            return false;
        }
        file_size_ = static_cast<uint64_t>(sb.st_size);
        if (file_size_ < 16) {
            Close();
            return false;
        }

        mapped_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapped_ptr_ == MAP_FAILED) {
            mapped_ptr_ = nullptr;
            Close();
            return false;
        }
#endif
        is_mapped_ = true;
        return ParseHeaderAndDirectory();
    }

    void Close() {
        if (mapped_ptr_) {
#if defined(_WIN32)
            UnmapViewOfFile(mapped_ptr_);
#else
            munmap(mapped_ptr_, file_size_);
#endif
            mapped_ptr_ = nullptr;
        }
#if defined(_WIN32)
        if (mapping_handle_) {
            CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
#endif
        is_mapped_ = false;
        file_size_ = 0;
        tensors_.clear();
        metadata_strings_.clear();
        metadata_uint64s_.clear();
    }

    [[nodiscard]] bool IsMapped() const noexcept { return is_mapped_; }
    [[nodiscard]] uint32_t Version() const noexcept { return version_; }
    [[nodiscard]] uint64_t TensorCount() const noexcept { return tensor_count_; }
    [[nodiscard]] uint32_t Alignment() const noexcept { return alignment_; }

    [[nodiscard]] const void* GetTensorData(const char* name, size_t* out_size = nullptr) const noexcept {
        if (!is_mapped_ || !name) return nullptr;
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return nullptr;
        if (out_size) *out_size = static_cast<size_t>(it->second.byte_size);
        if (it->second.absolute_offset + it->second.byte_size > file_size_) return nullptr;
        return static_cast<const uint8_t*>(mapped_ptr_) + it->second.absolute_offset;
    }

    [[nodiscard]] const TensorDescriptor* GetTensorDescriptor(const char* name) const noexcept {
        if (!name) return nullptr;
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? &it->second : nullptr;
    }

    bool GetMetadataString(const char* key, std::string& out_str) const {
        if (!key) return false;
        auto it = metadata_strings_.find(key);
        if (it == metadata_strings_.end()) return false;
        out_str = it->second;
        return true;
    }

    bool GetMetadataUint64(const char* key, uint64_t& out_val) const {
        if (!key) return false;
        auto it = metadata_uint64s_.find(key);
        if (it == metadata_uint64s_.end()) return false;
        out_val = it->second;
        return true;
    }

private:
    bool ParseHeaderAndDirectory() {
        if (!mapped_ptr_ || file_size_ < 16) return false;
        const auto* data = static_cast<const uint8_t*>(mapped_ptr_);

        uint32_t magic = 0;
        std::memcpy(&magic, data, sizeof(uint32_t));
        if (magic != GgufMagic) return false;

        std::memcpy(&version_, data + 4, sizeof(uint32_t));
        if (version_ != 2 && version_ != 3) return false;

        std::memcpy(&tensor_count_, data + 8, sizeof(uint64_t));
        std::memcpy(&metadata_kv_count_, data + 16, sizeof(uint64_t));

        size_t cursor = 24;

        auto read_string = [&](std::string& str) -> bool {
            if (cursor + sizeof(uint64_t) > file_size_) return false;
            uint64_t len = 0;
            std::memcpy(&len, data + cursor, sizeof(uint64_t));
            cursor += sizeof(uint64_t);
            if (cursor + len > file_size_) return false;
            str.assign(reinterpret_cast<const char*>(data + cursor), static_cast<size_t>(len));
            cursor += static_cast<size_t>(len);
            return true;
        };

        for (uint64_t i = 0; i < metadata_kv_count_; ++i) {
            std::string key;
            if (!read_string(key)) return false;
            if (cursor + sizeof(uint32_t) > file_size_) return false;
            uint32_t val_type = 0;
            std::memcpy(&val_type, data + cursor, sizeof(uint32_t));
            cursor += sizeof(uint32_t);

            auto skip_value = [&](uint32_t type) -> bool {
                switch (type) {
                    case 0: case 1: case 7: cursor += 1; return (cursor <= file_size_);
                    case 2: case 3: cursor += 2; return (cursor <= file_size_);
                    case 4: case 5: case 6: cursor += 4; return (cursor <= file_size_);
                    case 8: {
                        std::string s;
                        if (!read_string(s)) return false;
                        if (key.find("general.name") != std::string::npos ||
                            key.find("general.architecture") != std::string::npos) {
                            metadata_strings_[key] = s;
                        }
                        return true;
                    }
                    case 10: case 11: case 12: {
                        if (cursor + sizeof(uint64_t) > file_size_) return false;
                        uint64_t u64_val = 0;
                        std::memcpy(&u64_val, data + cursor, sizeof(uint64_t));
                        if (key.find("general.alignment") != std::string::npos) {
                            alignment_ = static_cast<uint32_t>(u64_val);
                        }
                        metadata_uint64s_[key] = u64_val;
                        cursor += 8;
                        return (cursor <= file_size_);
                    }
                    case 9: {
                        if (cursor + sizeof(uint32_t) + sizeof(uint64_t) > file_size_) return false;
                        uint32_t elem_type = 0;
                        uint64_t arr_len = 0;
                        std::memcpy(&elem_type, data + cursor, sizeof(uint32_t));
                        cursor += sizeof(uint32_t);
                        std::memcpy(&arr_len, data + cursor, sizeof(uint64_t));
                        cursor += sizeof(uint64_t);
                        for (uint64_t a = 0; a < arr_len; ++a) {
                            switch (elem_type) {
                                case 0: case 1: case 7: cursor += 1; break;
                                case 2: case 3: cursor += 2; break;
                                case 4: case 5: case 6: cursor += 4; break;
                                case 8: { std::string dummy; if (!read_string(dummy)) return false; break; }
                                case 10: case 11: case 12: cursor += 8; break;
                                default: return false;
                            }
                            if (cursor > file_size_) return false;
                        }
                        return true;
                    }
                    default: return false;
                }
            };

            if (!skip_value(val_type)) return false;
        }

        for (uint64_t i = 0; i < tensor_count_; ++i) {
            TensorDescriptor desc;
            if (!read_string(desc.name)) return false;
            if (cursor + sizeof(uint32_t) > file_size_) return false;
            std::memcpy(&desc.n_dimensions, data + cursor, sizeof(uint32_t));
            cursor += sizeof(uint32_t);

            desc.dimensions.resize(desc.n_dimensions);
            desc.num_elements = 1;
            for (uint32_t d = 0; d < desc.n_dimensions; ++d) {
                if (cursor + sizeof(uint64_t) > file_size_) return false;
                std::memcpy(&desc.dimensions[d], data + cursor, sizeof(uint64_t));
                cursor += sizeof(uint64_t);
                desc.num_elements *= desc.dimensions[d];
            }

            if (cursor + sizeof(uint32_t) + sizeof(uint64_t) > file_size_) return false;
            std::memcpy(&desc.type, data + cursor, sizeof(uint32_t));
            cursor += sizeof(uint32_t);
            std::memcpy(&desc.offset, data + cursor, sizeof(uint64_t));
            cursor += sizeof(uint64_t);

            desc.byte_size = CalcTensorByteSize(desc.type, desc.num_elements);
            tensors_[desc.name] = desc;
        }

        uint64_t align = (alignment_ > 0) ? alignment_ : GgufDefaultAlignment;
        tensor_data_base_offset_ = (cursor + align - 1) & ~(static_cast<uint64_t>(align) - 1);

        for (auto& [name, desc] : tensors_) {
            desc.absolute_offset = tensor_data_base_offset_ + desc.offset;
        }

        return true;
    }
};

} // namespace g2g::carbon_forge::gguf

// Opaque GGUF Context for C ABI
struct kimodo_gguf_ctx {
    g2g::carbon_forge::gguf::GGUFMemoryMappedFile file;
};

#endif // GGUF_MMAP_WIN32_HPP
