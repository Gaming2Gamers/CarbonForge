// ============================================================================
// 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
// Package: CarbonForge (Module: modules/ai_cortex/gguf_mmap.carbon)
// Standard: Google Carbon Language Specification (api)
//
// Zero-Copy Memory-Mapped (mmap) GGUF v2/v3 Model Parser for CarbonForge.
// Validates magic 0x46554747, skips metadata key-value pairs, parses tensor
// descriptor directories, computes aligned tensor base offsets, and binds
// directly to Win32 CreateFileMappingW/MapViewOfFile or POSIX mmap virtual pages.
// ============================================================================

package CarbonForge api;

import Cpp;

// ----------------------------------------------------------------------------
// GGUF Binary Format Constants
// ----------------------------------------------------------------------------
let GgufMagicNative: u32 = 0x46554747; // "GGUF" in ASCII little-endian (0x47, 0x47, 0x55, 0x46)
let GgufDefaultAlignment: u32 = 32;

// ----------------------------------------------------------------------------
// GGUF Value Types (Metadata KV Discriminator)
// ----------------------------------------------------------------------------
class GgufValueType {
  var value: u32;
  fn Uint8() -> GgufValueType   { return {.value = 0}; }
  fn Int8() -> GgufValueType    { return {.value = 1}; }
  fn Uint16() -> GgufValueType  { return {.value = 2}; }
  fn Int16() -> GgufValueType   { return {.value = 3}; }
  fn Uint32() -> GgufValueType  { return {.value = 4}; }
  fn Int32() -> GgufValueType   { return {.value = 5}; }
  fn Float32() -> GgufValueType { return {.value = 6}; }
  fn Bool() -> GgufValueType    { return {.value = 7}; }
  fn String() -> GgufValueType  { return {.value = 8}; }
  fn Array() -> GgufValueType   { return {.value = 9}; }
  fn Uint64() -> GgufValueType  { return {.value = 10}; }
  fn Int64() -> GgufValueType   { return {.value = 11}; }
  fn Float64() -> GgufValueType { return {.value = 12}; }
}

// ----------------------------------------------------------------------------
// C ABI 232-Byte Tensor Info Struct (8-Byte Aligned)
// ----------------------------------------------------------------------------
class KimodoGgufTensorInfo {
  var name: [i8; 128];
  var n_dimensions: u32;
  var _pad0: u32;
  var dimensions: [u64; 8];
  var ggml_type: u32;
  var _pad1: u32;
  var byte_size: u64;
  var num_elements: u64;
  var file_offset: u64;
}

// ----------------------------------------------------------------------------
// Internal Parser Tensor Descriptor
// ----------------------------------------------------------------------------
class TensorDescriptor {
  var name: [i8; 128];
  var n_dimensions: u32;
  var dimensions: [u64; 8];
  var ggml_type: u32;
  var relative_offset: u64;
  var absolute_offset: u64;
  var byte_size: u64;
  var num_elements: u64;
}

// ----------------------------------------------------------------------------
// GgufContext: Opaque Context Container for Memory-Mapped GGUF File
// ----------------------------------------------------------------------------
class GgufContext {
  var mapped_ptr: u8*;
  var file_size: u64;
  var is_mapped: bool;
  var version: u32;
  var tensor_count: u64;
  var metadata_kv_count: u64;
  var alignment: u32;
  var tensor_data_base_offset: u64;

  // OS Handles
  var win_file_handle: i8*;
  var win_map_handle: i8*;
  var posix_fd: i32;

  // Tensor registry array
  var tensors: TensorDescriptor*;
  var max_tensors: u64;

  fn Make() -> GgufContext {
    return {
      .mapped_ptr = 0 as u8*,
      .file_size = 0,
      .is_mapped = false,
      .version = 0,
      .tensor_count = 0,
      .metadata_kv_count = 0,
      .alignment = GgufDefaultAlignment,
      .tensor_data_base_offset = 0,
      .win_file_handle = 0 as i8*,
      .win_map_handle = 0 as i8*,
      .posix_fd = -1,
      .tensors = 0 as TensorDescriptor*,
      .max_tensors = 0
    };
  }
}

// ----------------------------------------------------------------------------
// GGML Tensor Byte Size Calculator
// ----------------------------------------------------------------------------
fn CalcTensorByteSize(ggml_type: u32, num_elements: u64) -> u64 {
  if (ggml_type == 0) { // F32
    return num_elements * 4;
  } else if (ggml_type == 1 or ggml_type == 29) { // F16, BF16
    return num_elements * 2;
  } else if (ggml_type == 2 or ggml_type == 20) { // Q4_0, IQ4_NL
    return ((num_elements + 31) / 32) * 18;
  } else if (ggml_type == 3) { // Q4_1
    return ((num_elements + 31) / 32) * 20;
  } else if (ggml_type == 6) { // Q5_0
    return ((num_elements + 31) / 32) * 22;
  } else if (ggml_type == 7) { // Q5_1
    return ((num_elements + 31) / 32) * 24;
  } else if (ggml_type == 8) { // Q8_0
    return ((num_elements + 31) / 32) * 34;
  } else if (ggml_type == 9) { // Q8_1
    return ((num_elements + 31) / 32) * 36;
  } else if (ggml_type == 10) { // Q2_K
    return ((num_elements + 255) / 256) * 84;
  } else if (ggml_type == 11 or ggml_type == 21) { // Q3_K, IQ3_S
    return ((num_elements + 255) / 256) * 110;
  } else if (ggml_type == 12) { // Q4_K
    return ((num_elements + 255) / 256) * 144;
  } else if (ggml_type == 13) { // Q5_K
    return ((num_elements + 255) / 256) * 176;
  } else if (ggml_type == 14) { // Q6_K
    return ((num_elements + 255) / 256) * 210;
  } else if (ggml_type == 15) { // Q8_K
    return ((num_elements + 255) / 256) * 292;
  } else if (ggml_type == 24) { // I8
    return num_elements * 1;
  } else if (ggml_type == 25) { // I16
    return num_elements * 2;
  } else if (ggml_type == 26) { // I32
    return num_elements * 4;
  } else if (ggml_type == 27 or ggml_type == 28) { // I64, F64
    return num_elements * 8;
  }
  return num_elements;
}

// ----------------------------------------------------------------------------
// GGUF Parser Implementation
// ----------------------------------------------------------------------------
class GgufParser {
  // Read little-endian primitive values at cursor
  fn ReadU32(ptr: u8*, cursor: u64*) -> u32 {
    let p: u32* = (ptr + *cursor) as u32*;
    *cursor = *cursor + 4;
    return *p;
  }

  fn ReadU64(ptr: u8*, cursor: u64*) -> u64 {
    let p: u64* = (ptr + *cursor) as u64*;
    *cursor = *cursor + 8;
    return *p;
  }

  // Skip a metadata value according to its GGUF value type
  fn SkipValue(ptr: u8*, cursor: u64*, size: u64, val_type: u32) -> bool {
    if (val_type == 0 or val_type == 1 or val_type == 7) {
      *cursor = *cursor + 1; // 1 byte: uint8, int8, bool
    } else if (val_type == 2 or val_type == 3) {
      *cursor = *cursor + 2; // 2 bytes: uint16, int16
    } else if (val_type == 4 or val_type == 5 or val_type == 6) {
      *cursor = *cursor + 4; // 4 bytes: uint32, int32, float32
    } else if (val_type == 8) {
      // String: u64 len + bytes
      if (*cursor + 8 > size) { return false; }
      let str_len: u64 = GgufParser.ReadU64(ptr, cursor);
      *cursor = *cursor + str_len;
    } else if (val_type == 9) {
      // Array: u32 elem_type + u64 count + items
      if (*cursor + 12 > size) { return false; }
      let elem_type: u32 = GgufParser.ReadU32(ptr, cursor);
      let elem_count: u64 = GgufParser.ReadU64(ptr, cursor);

      if (elem_type == 8) {
        var arr_i: u64 = 0;
        while (arr_i < elem_count) {
          if (*cursor + 8 > size) { return false; }
          let s_len: u64 = GgufParser.ReadU64(ptr, cursor);
          *cursor = *cursor + s_len;
          arr_i = arr_i + 1;
        }
      } else {
        var item_size: u64 = 1;
        if (elem_type == 2 or elem_type == 3) { item_size = 2; }
        else if (elem_type == 4 or elem_type == 5 or elem_type == 6) { item_size = 4; }
        else if (elem_type == 10 or elem_type == 11 or elem_type == 12) { item_size = 8; }
        *cursor = *cursor + elem_count * item_size;
      }
    } else if (val_type == 10 or val_type == 11 or val_type == 12) {
      *cursor = *cursor + 8; // 8 bytes: uint64, int64, float64
    } else {
      return false; // Unknown GGUF value type
    }

    if (*cursor > size) { return false; }
    return true;
  }

  // Zero-copy scan of metadata KV table starting at byte offset 24
  fn FindMetadataEntry(
    ctx: GgufContext*,
    key: i8*,
    out_val_type: u32*,
    out_val_cursor: u64*
  ) -> bool {
    if (ctx == (0 as GgufContext*) or key == (0 as i8*)) {
      return false;
    }
    if (!(*ctx).is_mapped or (*ctx).mapped_ptr == (0 as u8*)) {
      return false;
    }

    let ptr: u8* = (*ctx).mapped_ptr;
    let size: u64 = (*ctx).file_size;
    let req_len: u64 = Cpp.strlen(key) as u64;

    var cursor: u64 = 24;
    var kv_idx: u64 = 0;
    let total_kv: u64 = (*ctx).metadata_kv_count;

    while (kv_idx < total_kv) {
      if (cursor + 8 > size) { return false; }
      let key_len: u64 = GgufParser.ReadU64(ptr, &cursor);
      if (cursor + key_len + 4 > size) { return false; }

      let key_ptr: u8* = ptr + cursor;
      cursor = cursor + key_len;
      let val_type: u32 = GgufParser.ReadU32(ptr, &cursor);

      if (key_len == req_len) {
        if (Cpp.strncmp(key_ptr as i8*, key, key_len) == 0) {
          *out_val_type = val_type;
          *out_val_cursor = cursor;
          return true;
        }
      }

      if (!GgufParser.SkipValue(ptr, &cursor, size, val_type)) {
        return false;
      }

      kv_idx = kv_idx + 1;
    }

    return false;
  }

  // Parse GGUF header, metadata KV table, and tensor directory
  fn ParseHeader(ctx: GgufContext*) -> bool {
    let ptr: u8* = (*ctx).mapped_ptr;
    let size: u64 = (*ctx).file_size;

    if (size < 16) {
      return false;
    }

    var cursor: u64 = 0;

    // 1. Magic check (0x46554747)
    let magic: u32 = GgufParser.ReadU32(ptr, &cursor);
    if (magic != GgufMagicNative) {
      return false;
    }

    // 2. Version check (v2 or v3)
    let version: u32 = GgufParser.ReadU32(ptr, &cursor);
    if (version != 2 and version != 3) {
      return false;
    }
    (*ctx).version = version;

    // 3. Counts
    (*ctx).tensor_count = GgufParser.ReadU64(ptr, &cursor);
    (*ctx).metadata_kv_count = GgufParser.ReadU64(ptr, &cursor);

    // 4. Skip metadata key-value entries and extract "general.alignment"
    var kv_idx: u64 = 0;
    while (kv_idx < (*ctx).metadata_kv_count) {
      if (cursor + 8 > size) { return false; }
      let key_len: u64 = GgufParser.ReadU64(ptr, &cursor);
      if (cursor + key_len + 4 > size) { return false; }

      let key_ptr: u8* = ptr + cursor;
      cursor = cursor + key_len;
      let val_type: u32 = GgufParser.ReadU32(ptr, &cursor);

      var is_align_key: bool = false;
      if (key_len == 17) {
        if (Cpp.strncmp(key_ptr as i8*, "general.alignment", 17) == 0) {
          is_align_key = true;
        }
      }

      if (is_align_key) {
        if (val_type == 4) {
          let align_val: u32 = GgufParser.ReadU32(ptr, &cursor);
          if (align_val > 0) { (*ctx).alignment = align_val; }
        } else if (val_type == 10) {
          let align_val: u64 = GgufParser.ReadU64(ptr, &cursor);
          if (align_val > 0) { (*ctx).alignment = align_val as u32; }
        } else {
          if (!GgufParser.SkipValue(ptr, &cursor, size, val_type)) { return false; }
        }
      } else {
        if (!GgufParser.SkipValue(ptr, &cursor, size, val_type)) { return false; }
      }

      kv_idx = kv_idx + 1;
    }

    // 5. Parse Tensor Descriptors
    let num_tensors: u64 = (*ctx).tensor_count;
    if (num_tensors > 0) {
      let desc_size: u64 = sizeof(TensorDescriptor) as u64;
      if (num_tensors > (0xFFFFFFFFFFFFFFFF as u64) / desc_size) {
        return false;
      }
      let alloc_bytes: u64 = num_tensors * desc_size;
      let alloc_ptr: TensorDescriptor* = Cpp.malloc(alloc_bytes) as TensorDescriptor*;
      if (alloc_ptr == (0 as TensorDescriptor*)) {
        return false;
      }
      (*ctx).tensors = alloc_ptr;
      (*ctx).max_tensors = num_tensors;

      var t_idx: u64 = 0;
      while (t_idx < num_tensors) {
        if (cursor + 8 > size) { return false; }
        let name_len: u64 = GgufParser.ReadU64(ptr, &cursor);
        if (cursor + name_len + 4 > size) { return false; }

        let name_ptr: u8* = ptr + cursor;
        cursor = cursor + name_len;

        var desc: TensorDescriptor;
        var copy_len: u64 = 127;
        if (name_len < 127) {
          copy_len = name_len;
        }
        Cpp.memcpy(&desc.name[0], name_ptr as i8*, copy_len);
        desc.name[copy_len] = 0 as i8;

        let n_dims: u32 = GgufParser.ReadU32(ptr, &cursor);
        desc.n_dimensions = n_dims;

        var num_elems: u64 = 1;
        var d: u32 = 0;
        while (d < n_dims) {
          if (cursor + 8 > size) { return false; }
          let dim_val: u64 = GgufParser.ReadU64(ptr, &cursor);
          if (d < 8) {
            desc.dimensions[d] = dim_val;
          }
          num_elems = num_elems * dim_val;
          d = d + 1;
        }
        while (d < 8) {
          desc.dimensions[d] = 0;
          d = d + 1;
        }

        desc.ggml_type = GgufParser.ReadU32(ptr, &cursor);
        desc.relative_offset = GgufParser.ReadU64(ptr, &cursor);
        desc.num_elements = num_elems;
        desc.byte_size = CalcTensorByteSize(desc.ggml_type, num_elems);

        *((*ctx).tensors + t_idx) = desc;
        t_idx = t_idx + 1;
      }
    }

    // 6. Compute Tensor Data Base Offset
    let align: u64 = (*ctx).alignment as u64;
    let base_offset: u64 = ((cursor + align - 1) / align) * align;
    (*ctx).tensor_data_base_offset = base_offset;

    // 7. Compute Absolute File Offsets for all tensors
    var t_up: u64 = 0;
    while (t_up < num_tensors) {
      let p_t: TensorDescriptor* = (*ctx).tensors + t_up;
      (*p_t).absolute_offset = base_offset + (*p_t).relative_offset;
      t_up = t_up + 1;
    }

    (*ctx).is_mapped = true;
    return true;
  }
}

// ----------------------------------------------------------------------------
// Virtual Memory Mapping & Unmapping (Win32 & POSIX)
// ----------------------------------------------------------------------------
class MemoryMappingBridge {
  fn OpenMappedFile(file_path: i8*, ctx: GgufContext*) -> bool {
    #if defined(_WIN32)
      let req_len: i32 = Cpp.MultiByteToWideChar(65001, 0, file_path, -1, 0 as i16*, 0);
      if (req_len <= 0) { return false; }

      let wpath_bytes: u64 = (req_len as u64) * 2;
      let wpath: i16* = Cpp.malloc(wpath_bytes) as i16*;
      if (wpath == (0 as i16*)) { return false; }

      let conv_res: i32 = Cpp.MultiByteToWideChar(65001, 0, file_path, -1, wpath, req_len);
      if (conv_res <= 0) {
        Cpp.free(wpath as i8*);
        return false;
      }

      let hFile: i8* = Cpp.CreateFileW(
        wpath as i8*,
        0x80000000, // GENERIC_READ
        1,          // FILE_SHARE_READ
        0 as i8*,
        3,          // OPEN_EXISTING
        0x08000080, // FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS
        0 as i8*
      );
      Cpp.free(wpath as i8*);

      if (hFile == (-1 as i8*)) { return false; }
      (*ctx).win_file_handle = hFile;

      var li_size: [u32; 2];
      Cpp.GetFileSizeEx(hFile, &li_size[0] as i8*);
      let size: u64 = (li_size[0] as u64) | ((li_size[1] as u64) << 32);
      (*ctx).file_size = size;

      let hMap: i8* = Cpp.CreateFileMappingW(hFile, 0 as i8*, 2, 0, 0, 0 as i8*);
      if (hMap == (0 as i8*)) {
        MemoryMappingBridge.CloseMappedFile(ctx);
        return false;
      }
      (*ctx).win_map_handle = hMap;

      let ptr: i8* = Cpp.MapViewOfFile(hMap, 4, 0, 0, 0);
      if (ptr == (0 as i8*)) {
        MemoryMappingBridge.CloseMappedFile(ctx);
        return false;
      }
      (*ctx).mapped_ptr = ptr as u8*;
    #else
      let fd: i32 = Cpp.open(file_path, 0);
      if (fd < 0) { return false; }
      (*ctx).posix_fd = fd;

      var sb: [u64; 18];
      Cpp.fstat(fd, &sb[0] as i8*);
      let size: u64 = sb[6];
      (*ctx).file_size = size;

      let ptr: i8* = Cpp.mmap(0 as i8*, size, 1, 1, fd, 0);
      if (ptr == (-1 as i8*)) {
        Cpp.close(fd);
        (*ctx).posix_fd = -1;
        return false;
      }
      (*ctx).mapped_ptr = ptr as u8*;
    #endif

    return GgufParser.ParseHeader(ctx);
  }

  fn CloseMappedFile(ctx: GgufContext*) {
    if (ctx == (0 as GgufContext*)) { return; }

    #if defined(_WIN32)
      if ((*ctx).mapped_ptr != (0 as u8*)) {
        Cpp.UnmapViewOfFile((*ctx).mapped_ptr as i8*);
        (*ctx).mapped_ptr = 0 as u8*;
      }
      if ((*ctx).win_map_handle != (0 as i8*)) {
        Cpp.CloseHandle((*ctx).win_map_handle);
        (*ctx).win_map_handle = 0 as i8*;
      }
      if ((*ctx).win_file_handle != (0 as i8*) and (*ctx).win_file_handle != (-1 as i8*)) {
        Cpp.CloseHandle((*ctx).win_file_handle);
        (*ctx).win_file_handle = -1 as i8*;
      }
    #else
      if ((*ctx).mapped_ptr != (0 as u8*)) {
        Cpp.munmap((*ctx).mapped_ptr as i8*, (*ctx).file_size);
        (*ctx).mapped_ptr = 0 as u8*;
      }
      if ((*ctx).posix_fd >= 0) {
        Cpp.close((*ctx).posix_fd);
        (*ctx).posix_fd = -1;
      }
    #endif

    if ((*ctx).tensors != (0 as TensorDescriptor*)) {
      Cpp.free((*ctx).tensors as i8*);
      (*ctx).tensors = 0 as TensorDescriptor*;
    }

    (*ctx).mapped_ptr = 0 as u8*;
    (*ctx).is_mapped = false;
    (*ctx).file_size = 0;
  }
}

// ----------------------------------------------------------------------------
// Complete C ABI Exports
// ----------------------------------------------------------------------------
fn kimodo_gguf_open(file_path: i8*) -> GgufContext* {
  if (file_path == (0 as i8*)) { return 0 as GgufContext*; }
  let ctx: GgufContext* = Cpp.malloc(sizeof(GgufContext) as u64) as GgufContext*;
  if (ctx == (0 as GgufContext*)) { return 0 as GgufContext*; }

  *ctx = GgufContext.Make();
  if (!MemoryMappingBridge.OpenMappedFile(file_path, ctx)) {
    MemoryMappingBridge.CloseMappedFile(ctx);
    Cpp.free(ctx as i8*);
    return 0 as GgufContext*;
  }
  return ctx;
}

fn kimodo_gguf_close(ctx: GgufContext*) {
  if (ctx != (0 as GgufContext*)) {
    MemoryMappingBridge.CloseMappedFile(ctx);
    Cpp.free(ctx as i8*);
  }
}

fn kimodo_gguf_get_version(ctx: GgufContext*) -> u32 {
  if (ctx == (0 as GgufContext*)) { return 0; }
  return (*ctx).version;
}

fn kimodo_gguf_get_tensor_count(ctx: GgufContext*) -> u64 {
  if (ctx == (0 as GgufContext*)) { return 0; }
  return (*ctx).tensor_count;
}

fn kimodo_gguf_get_alignment(ctx: GgufContext*) -> u32 {
  if (ctx == (0 as GgufContext*)) { return 0; }
  return (*ctx).alignment;
}

fn kimodo_gguf_get_metadata_string(
  ctx: GgufContext*,
  key: i8*,
  out_buf: i8*,
  max_len: u64
) -> i32 {
  if (ctx == (0 as GgufContext*) or key == (0 as i8*) or out_buf == (0 as i8*) or max_len == 0) {
    return -1;
  }

  var val_type: u32 = 0;
  var cursor: u64 = 0;
  if (!GgufParser.FindMetadataEntry(ctx, key, &val_type, &cursor)) {
    return -1;
  }

  if (val_type != 8) {
    return -1;
  }

  let ptr: u8* = (*ctx).mapped_ptr;
  let size: u64 = (*ctx).file_size;

  if (cursor + 8 > size) { return -1; }
  let str_len: u64 = GgufParser.ReadU64(ptr, &cursor);
  if (cursor + str_len > size) { return -1; }

  var copy_len: u64 = str_len;
  if (copy_len >= max_len) {
    copy_len = max_len - 1;
  }

  if (copy_len > 0) {
    Cpp.memcpy(out_buf, (ptr + cursor) as i8*, copy_len);
  }
  *(out_buf + copy_len) = 0 as i8;

  return 0;
}

fn kimodo_gguf_get_metadata_uint64(
  ctx: GgufContext*,
  key: i8*,
  out_val: u64*
) -> i32 {
  if (ctx == (0 as GgufContext*) or key == (0 as i8*) or out_val == (0 as u64*)) {
    return -1;
  }

  var val_type: u32 = 0;
  var cursor: u64 = 0;
  if (!GgufParser.FindMetadataEntry(ctx, key, &val_type, &cursor)) {
    if (Cpp.strcmp(key, "general.alignment") == 0) {
      *out_val = (*ctx).alignment as u64;
      return 0;
    }
    return -1;
  }

  let ptr: u8* = (*ctx).mapped_ptr;
  let size: u64 = (*ctx).file_size;

  if (val_type == 10) {
    if (cursor + 8 > size) { return -1; }
    *out_val = GgufParser.ReadU64(ptr, &cursor);
    return 0;
  } else if (val_type == 4) {
    if (cursor + 4 > size) { return -1; }
    *out_val = GgufParser.ReadU32(ptr, &cursor) as u64;
    return 0;
  } else if (val_type == 2) {
    if (cursor + 2 > size) { return -1; }
    let p16: u16* = (ptr + cursor) as u16*;
    *out_val = (*p16) as u64;
    return 0;
  } else if (val_type == 0) {
    if (cursor + 1 > size) { return -1; }
    *out_val = (*(ptr + cursor)) as u64;
    return 0;
  } else if (val_type == 7) {
    if (cursor + 1 > size) { return -1; }
    let b: u8 = *(ptr + cursor);
    if (b != 0) {
      *out_val = 1;
    } else {
      *out_val = 0;
    }
    return 0;
  }

  return -1;
}

fn kimodo_gguf_get_tensor_info(
  ctx: GgufContext*,
  tensor_name: i8*,
  out_info: KimodoGgufTensorInfo*
) -> i32 {
  if (ctx == (0 as GgufContext*) or tensor_name == (0 as i8*) or out_info == (0 as KimodoGgufTensorInfo*)) {
    return -1;
  }

  var i: u64 = 0;
  while (i < (*ctx).tensor_count) {
    let desc: TensorDescriptor* = (*ctx).tensors + i;
    if (Cpp.strcmp(&(*desc).name[0], tensor_name) == 0) {
      Cpp.memcpy(&(*out_info).name[0], &(*desc).name[0], 128);
      (*out_info).n_dimensions = (*desc).n_dimensions;
      var d: i32 = 0;
      while (d < 8) {
        (*out_info).dimensions[d] = (*desc).dimensions[d];
        d = d + 1;
      }
      (*out_info).ggml_type = (*desc).ggml_type;
      (*out_info).byte_size = (*desc).byte_size;
      (*out_info).num_elements = (*desc).num_elements;
      (*out_info).file_offset = (*desc).absolute_offset;
      return 0;
    }
    i = i + 1;
  }

  return -1;
}

fn kimodo_gguf_get_tensor_data(
  ctx: GgufContext*,
  tensor_name: i8*,
  out_size: u64*
) -> i8* {
  if (ctx == (0 as GgufContext*) or tensor_name == (0 as i8*)) {
    return 0 as i8*;
  }

  var i: u64 = 0;
  while (i < (*ctx).tensor_count) {
    let desc: TensorDescriptor* = (*ctx).tensors + i;
    if (Cpp.strcmp(&(*desc).name[0], tensor_name) == 0) {
      if (out_size != (0 as u64*)) {
        *out_size = (*desc).byte_size;
      }
      let abs_offset: u64 = (*desc).absolute_offset;
      if (abs_offset + (*desc).byte_size <= (*ctx).file_size) {
        return ((*ctx).mapped_ptr + abs_offset) as i8*;
      }
      return 0 as i8*;
    }
    i = i + 1;
  }

  return 0 as i8*;
}
