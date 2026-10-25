// ============================================================================
// 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
// Package: CarbonForge (Module: modules/ai_cortex/avx_vnni_cortex.carbon)
// Standard: Google Carbon Language Specification (api)
//
// High-performance VEX-encoded SIMD kernel for symmetric INT8 vector dot products
// and batch cosine similarity scoring on Intel Core Ultra 9 285 (Arrow Lake-S).
//
// SILICON ARCHITECTURE INVARIANT:
// Client Arrow Lake-S does NOT support AVX-512.
// EVEX prefix 0x62 causes hardware invalid opcode fault (STATUS_ILLEGAL_INSTRUCTION).
// This module strictly leverages VEX-encoded 0xC4/0xC5 opcodes via
// _mm256_dpbusd_avx_epi32 with fallback to AVX2 _mm256_cvtepi8_epi16 + _mm256_madd_epi16.
// ============================================================================

package CarbonForge api;

import Cpp;

// ----------------------------------------------------------------------------
// SimdCortexEngine: Core Vector Acceleration Routines
// ----------------------------------------------------------------------------
class SimdCortexEngine {
  // Probes CPUID Leaf 1, Leaf 7.0, Leaf 7.1 and XCR0 for AVX2 and AVX-VNNI support
  fn IsVnniSupported() -> i32 {
    var cpuInfo: [i32; 4] = [0, 0, 0, 0];

    // 1. Check Leaf 1 for AVX and OSXSAVE
    Cpp.__cpuid(&cpuInfo[0], 1);
    let osxsave: bool = (cpuInfo[2] & (1 << 27)) != 0;
    let avx: bool = (cpuInfo[2] & (1 << 28)) != 0;
    if (!osxsave or !avx) {
      return 0;
    }

    // 2. Check XGETBV for XMM and YMM state enabled by OS
    let xcr0: u64 = Cpp._xgetbv(0);
    if ((xcr0 & 6) != 6) {
      return 0;
    }

    // 3. Check Leaf 7 Subleaf 0 for AVX2 (EBX bit 5)
    Cpp.__cpuidex(&cpuInfo[0], 7, 0);
    let avx2: bool = (cpuInfo[1] & (1 << 5)) != 0;
    if (!avx2) {
      return 0;
    }

    // 4. Check Leaf 7 Subleaf 1 for AVX-VNNI (EAX bit 4)
    Cpp.__cpuidex(&cpuInfo[0], 7, 1);
    let avx_vnni: bool = (cpuInfo[0] & (1 << 4)) != 0;

    return (avx2 and avx_vnni) ? 1 : 0;
  }

  // Horizontal reduction of 8x int32 lanes in 256-bit register
  fn HorizontalSum256(v: Cpp.__m256i) -> i32 {
    let vlow: Cpp.__m128i = Cpp._mm256_castsi256_si128(v);
    let vhigh: Cpp.__m128i = Cpp._mm256_extracti128_si256(v, 1);
    var sum128: Cpp.__m128i = Cpp._mm_add_epi32(vlow, vhigh);
    sum128 = Cpp._mm_add_epi32(sum128, Cpp._mm_shuffle_epi32(sum128, 0x4E));
    sum128 = Cpp._mm_add_epi32(sum128, Cpp._mm_shuffle_epi32(sum128, 0xB1));
    return Cpp._mm_cvtsi128_si32(sum128);
  }

  // Computes raw unscaled integer dot products across all database vectors
  fn ComputeRawDotProducts(
    db_matrix: i8*,
    query_vec: i8*,
    num_vectors: i32,
    dim: i32,
    out_raw: i32*
  ) {
    let has_vnni: i32 = SimdCortexEngine.IsVnniSupported();

    if (has_vnni != 0) {
      // High-speed VEX-encoded AVX-VNNI Kernel
      // Query pre-shift: u = q + 128 into unsigned bytes [0, 255]
      var q_u8_stack: [u8; 4096];
      var query_u8: u8* = &q_u8_stack[0];
      if (dim > 4096) {
        query_u8 = Cpp.malloc(dim as u64) as u8*;
      }

      var j_prep: i32 = 0;
      while (j_prep < dim) {
        *(query_u8 + j_prep) = ((*(query_vec + j_prep) as i32) + 128) as u8;
        j_prep = j_prep + 1;
      }

      let ones_byte: Cpp.__m256i = Cpp._mm256_set1_epi8(1);

      var i: i32 = 0;
      while (i < num_vectors) {
        let row: i8* = db_matrix + ((i as u64) * (dim as u64));
        var dot0: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var dot1: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var dot2: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var dot3: Cpp.__m256i = Cpp._mm256_setzero_si256();

        var sum0: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var sum1: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var sum2: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var sum3: Cpp.__m256i = Cpp._mm256_setzero_si256();

        var j: i32 = 0;
        // 4x unrolled loop: 128 bytes per cycle
        while (j + 127 < dim) {
          let q0: Cpp.__m256i = Cpp._mm256_loadu_si256((query_u8 + j) as Cpp.__m256i*);
          let r0: Cpp.__m256i = Cpp._mm256_loadu_si256((row + j) as Cpp.__m256i*);
          dot0 = Cpp._mm256_dpbusd_avx_epi32(dot0, q0, r0);
          sum0 = Cpp._mm256_dpbusd_avx_epi32(sum0, ones_byte, r0);

          let q1: Cpp.__m256i = Cpp._mm256_loadu_si256((query_u8 + j + 32) as Cpp.__m256i*);
          let r1: Cpp.__m256i = Cpp._mm256_loadu_si256((row + j + 32) as Cpp.__m256i*);
          dot1 = Cpp._mm256_dpbusd_avx_epi32(dot1, q1, r1);
          sum1 = Cpp._mm256_dpbusd_avx_epi32(sum1, ones_byte, r1);

          let q2: Cpp.__m256i = Cpp._mm256_loadu_si256((query_u8 + j + 64) as Cpp.__m256i*);
          let r2: Cpp.__m256i = Cpp._mm256_loadu_si256((row + j + 64) as Cpp.__m256i*);
          dot2 = Cpp._mm256_dpbusd_avx_epi32(dot2, q2, r2);
          sum2 = Cpp._mm256_dpbusd_avx_epi32(sum2, ones_byte, r2);

          let q3: Cpp.__m256i = Cpp._mm256_loadu_si256((query_u8 + j + 96) as Cpp.__m256i*);
          let r3: Cpp.__m256i = Cpp._mm256_loadu_si256((row + j + 96) as Cpp.__m256i*);
          dot3 = Cpp._mm256_dpbusd_avx_epi32(dot3, q3, r3);
          sum3 = Cpp._mm256_dpbusd_avx_epi32(sum3, ones_byte, r3);

          j = j + 128;
        }

        // Remainder 32-byte blocks
        while (j + 31 < dim) {
          let q0: Cpp.__m256i = Cpp._mm256_loadu_si256((query_u8 + j) as Cpp.__m256i*);
          let r0: Cpp.__m256i = Cpp._mm256_loadu_si256((row + j) as Cpp.__m256i*);
          dot0 = Cpp._mm256_dpbusd_avx_epi32(dot0, q0, r0);
          sum0 = Cpp._mm256_dpbusd_avx_epi32(sum0, ones_byte, r0);
          j = j + 32;
        }

        let dot_tot: Cpp.__m256i = Cpp._mm256_add_epi32(Cpp._mm256_add_epi32(dot0, dot1), Cpp._mm256_add_epi32(dot2, dot3));
        let sum_tot: Cpp.__m256i = Cpp._mm256_add_epi32(Cpp._mm256_add_epi32(sum0, sum1), Cpp._mm256_add_epi32(sum2, sum3));

        var raw_dot: i32 = SimdCortexEngine.HorizontalSum256(dot_tot);
        var raw_sum: i32 = SimdCortexEngine.HorizontalSum256(sum_tot);

        // Scalar tail
        while (j < dim) {
          raw_dot = raw_dot + (*(query_u8 + j) as i32) * (*(row + j) as i32);
          raw_sum = raw_sum + (*(row + j) as i32);
          j = j + 1;
        }

        // Apply algebraic decomposition: sum(q*r) = sum(u*r) - 128 * sum(r)
        *(out_raw + i) = raw_dot - 128 * raw_sum;
        i = i + 1;
      }

      if (query_u8 != &q_u8_stack[0]) {
        Cpp.free(query_u8 as i8*);
      }
    } else {
      // Universal AVX2 Kernel (_mm256_cvtepi8_epi16 + _mm256_madd_epi16)
      var i: i32 = 0;
      while (i < num_vectors) {
        let row: i8* = db_matrix + ((i as u64) * (dim as u64));
        var acc0: Cpp.__m256i = Cpp._mm256_setzero_si256();
        var acc1: Cpp.__m256i = Cpp._mm256_setzero_si256();

        var j: i32 = 0;
        while (j + 31 < dim) {
          let q_low: Cpp.__m128i = Cpp._mm_loadu_si128((query_vec + j) as Cpp.__m128i*);
          let q_high: Cpp.__m128i = Cpp._mm_loadu_si128((query_vec + j + 16) as Cpp.__m128i*);
          let r_low: Cpp.__m128i = Cpp._mm_loadu_si128((row + j) as Cpp.__m128i*);
          let r_high: Cpp.__m128i = Cpp._mm_loadu_si128((row + j + 16) as Cpp.__m128i*);

          let q_low16: Cpp.__m256i = Cpp._mm256_cvtepi8_epi16(q_low);
          let q_high16: Cpp.__m256i = Cpp._mm256_cvtepi8_epi16(q_high);
          let r_low16: Cpp.__m256i = Cpp._mm256_cvtepi8_epi16(r_low);
          let r_high16: Cpp.__m256i = Cpp._mm256_cvtepi8_epi16(r_high);

          let prod_low: Cpp.__m256i = Cpp._mm256_madd_epi16(q_low16, r_low16);
          let prod_high: Cpp.__m256i = Cpp._mm256_madd_epi16(q_high16, r_high16);

          acc0 = Cpp._mm256_add_epi32(acc0, prod_low);
          acc1 = Cpp._mm256_add_epi32(acc1, prod_high);

          j = j + 32;
        }

        let total_acc: Cpp.__m256i = Cpp._mm256_add_epi32(acc0, acc1);
        var sum: i32 = SimdCortexEngine.HorizontalSum256(total_acc);

        while (j < dim) {
          sum = sum + (*(query_vec + j) as i32) * (*(row + j) as i32);
          j = j + 1;
        }

        *(out_raw + i) = sum;
        i = i + 1;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// C ABI Export Wrappers
// ----------------------------------------------------------------------------
fn avx_vnni_is_supported() -> i32 {
  return SimdCortexEngine.IsVnniSupported();
}

fn avx_vnni_dot_product_int8(
  db_matrix: i8*,
  query_vec: i8*,
  db_scales: f32*,
  query_scale: f32,
  num_vectors: i32,
  dim: i32,
  out_scores: f32*
) -> i32 {
  if (db_matrix == (0 as i8*) or query_vec == (0 as i8*) or db_scales == (0 as f32*) or out_scores == (0 as f32*) or num_vectors <= 0 or dim <= 0) {
    return -1;
  }

  var raw_stack: [i32; 1024];
  var raw_dots: i32* = &raw_stack[0];
  if (num_vectors > 1024) {
    raw_dots = Cpp.malloc((num_vectors as u64) * (sizeof(i32) as u64)) as i32*;
    if (raw_dots == (0 as i32*)) {
      return -2;
    }
  }

  SimdCortexEngine.ComputeRawDotProducts(db_matrix, query_vec, num_vectors, dim, raw_dots);

  var i: i32 = 0;
  while (i < num_vectors) {
    let raw: f32 = *(raw_dots + i) as f32;
    let scale: f32 = query_scale * (*(db_scales + i));
    *(out_scores + i) = raw * scale;
    i = i + 1;
  }

  if (raw_dots != &raw_stack[0]) {
    Cpp.free(raw_dots as i8*);
  }

  return 0;
}

fn avx_vnni_batch_cosine_similarity(
  db_matrix: i8*,
  query_vec: i8*,
  db_norms: f32*,
  query_norm: f32,
  db_scales: f32*,
  query_scale: f32,
  num_vectors: i32,
  dim: i32,
  out_sims: f32*
) -> i32 {
  if (db_matrix == (0 as i8*) or query_vec == (0 as i8*) or db_norms == (0 as f32*) or db_scales == (0 as f32*) or out_sims == (0 as f32*) or num_vectors <= 0 or dim <= 0) {
    return -1;
  }

  var raw_stack: [i32; 1024];
  var raw_dots: i32* = &raw_stack[0];
  if (num_vectors > 1024) {
    raw_dots = Cpp.malloc((num_vectors as u64) * (sizeof(i32) as u64)) as i32*;
    if (raw_dots == (0 as i32*)) {
      return -2;
    }
  }

  SimdCortexEngine.ComputeRawDotProducts(db_matrix, query_vec, num_vectors, dim, raw_dots);

  var i: i32 = 0;
  while (i < num_vectors) {
    let dot_float: f32 = (*(raw_dots + i) as f32) * (query_scale * (*(db_scales + i)));
    let denom: f32 = query_norm * (*(db_norms + i));
    var sim: f32 = (denom > 1.0e-9) ? (dot_float / denom) : 0.0;
    if (sim > 1.0) { sim = 1.0; }
    else if (sim < -1.0) { sim = -1.0; }
    *(out_sims + i) = sim;
    i = i + 1;
  }

  if (raw_dots != &raw_stack[0]) {
    Cpp.free(raw_dots as i8*);
  }

  return 0;
}
