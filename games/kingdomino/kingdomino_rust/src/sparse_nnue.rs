//! Sparse Step-3 NNUE: v3 loader, stateless oracle, and reversible accumulators.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::nnue_features;
use super::search::{self, Eval, Game};
use super::{Kingdomino, N, RustGameState, UndoRecord};

const MAGIC: &[u8; 4] = b"KNSP";
const FORMAT_VERSION: u32 = 3;
const HEADER_SIZE: usize = 44;
const EXPECTED_CORE_HASH: u64 = 0xf4a6_81bf_7fa8_950c;
const EXPECTED_SUMMARY_HASH: u64 = 0x0eca_00b1_9211_1097;
const MAX_DIM: usize = 1 << 24;
const MAX_ACTIVE_FEATURES: usize = 112;
const SUMMARY_QUANT_SCALE: f32 = 32767.0;

pub(super) struct SparseNnueWeights {
    summary_size: usize,
    acc_width: usize,
    tail_hidden: usize,
    margin_scale: f32,
    // Feature-major EmbeddingBag layout: each feature row is contiguous.
    acc_w: Vec<f32>,
    acc_b: Vec<f32>,
    // Dense tail weights are transposed once at load time to input-major
    // layout.  For each scalar input, all output weights are contiguous, so
    // the hot loop is an SIMD-friendly AXPY across independent outputs.
    t0_w_by_input: Vec<f32>,
    t0_b: Vec<f32>,
    t1_w_by_input: Vec<f32>,
    t1_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: f32,
    margin_w: Vec<f32>,
    margin_b: f32,
}

impl SparseNnueWeights {
    pub(super) fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| PyValueError::new_err(format!("sparse nnue load '{path}': {e}")))?;
        if bytes.len() < HEADER_SIZE {
            return Err(PyValueError::new_err("sparse nnue: truncated header"));
        }
        if &bytes[..4] != MAGIC {
            return Err(PyValueError::new_err("sparse nnue: bad magic"));
        }
        let u32_at =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        if u32_at(4) != FORMAT_VERSION {
            return Err(PyValueError::new_err(format!(
                "sparse nnue: unsupported format version {}",
                u32_at(4)
            )));
        }
        let feature_count = u32_at(8) as usize;
        let summary_size = u32_at(12) as usize;
        let acc_width = u32_at(16) as usize;
        let tail_hidden = u32_at(20) as usize;
        let margin_scale = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let core_hash = u64::from_le_bytes(bytes[28..36].try_into().unwrap());
        let summary_hash = u64::from_le_bytes(bytes[36..44].try_into().unwrap());
        if feature_count != nnue_features::CORE_SIZE
            || summary_size != nnue_features::SUMMARY_SIZE
            || core_hash != EXPECTED_CORE_HASH
            || summary_hash != EXPECTED_SUMMARY_HASH
        {
            return Err(PyValueError::new_err(format!(
                "sparse nnue: encoder contract mismatch (features={feature_count}, \
                 summary={summary_size}, core={core_hash:016x}, summary_hash={summary_hash:016x})"
            )));
        }
        if acc_width == 0
            || tail_hidden == 0
            || acc_width > MAX_DIM
            || tail_hidden > MAX_DIM
            || !margin_scale.is_finite()
            || margin_scale <= 0.0
        {
            return Err(PyValueError::new_err(
                "sparse nnue: invalid dimensions or margin scale",
            ));
        }

        let mut off = HEADER_SIZE;
        let read = |off: &mut usize, n: usize| -> PyResult<Vec<f32>> {
            let need = n
                .checked_mul(4)
                .ok_or_else(|| PyValueError::new_err("sparse nnue: tensor size overflow"))?;
            if off.checked_add(need).is_none_or(|end| end > bytes.len()) {
                return Err(PyValueError::new_err("sparse nnue: truncated tensor data"));
            }
            let out = bytes[*off..*off + need]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *off += need;
            Ok(out)
        };
        let tail_input = acc_width + summary_size;
        let acc_w = read(&mut off, feature_count * acc_width)?;
        let acc_b = read(&mut off, acc_width)?;
        let t0_w = read(&mut off, tail_hidden * tail_input)?;
        let t0_b = read(&mut off, tail_hidden)?;
        let t1_w = read(&mut off, tail_hidden * tail_hidden)?;
        let t1_b = read(&mut off, tail_hidden)?;
        let out_w = read(&mut off, tail_hidden)?;
        let out_b = read(&mut off, 1)?[0];
        let margin_w = read(&mut off, tail_hidden)?;
        let margin_b = read(&mut off, 1)?[0];
        if off != bytes.len() {
            return Err(PyValueError::new_err("sparse nnue: trailing bytes"));
        }
        let vectors = [
            &acc_w, &acc_b, &t0_w, &t0_b, &t1_w, &t1_b, &out_w, &margin_w,
        ];
        if !vectors.iter().all(|v| v.iter().all(|x| x.is_finite()))
            || !out_b.is_finite()
            || !margin_b.is_finite()
        {
            return Err(PyValueError::new_err(
                "sparse nnue: non-finite weight or bias",
            ));
        }
        let t0_w_by_input = transpose_output_major(&t0_w, tail_hidden, tail_input);
        let t1_w_by_input = transpose_output_major(&t1_w, tail_hidden, tail_hidden);
        Ok(Self {
            summary_size,
            acc_width,
            tail_hidden,
            margin_scale,
            acc_w,
            acc_b,
            t0_w_by_input,
            t0_b,
            t1_w_by_input,
            t1_b,
            out_w,
            out_b,
            margin_w,
            margin_b,
        })
    }

    fn accumulator(&self, active: &[i32]) -> Vec<f32> {
        let mut z = self.acc_b.clone();
        for &feature in active {
            let start = feature as usize * self.acc_width;
            let row = &self.acc_w[start..start + self.acc_width];
            for (dst, src) in z.iter_mut().zip(row) {
                *dst += *src;
            }
        }
        z
    }

    fn add_feature(&self, z: &mut [f32], feature: i32, sign: f32) {
        let start = feature as usize * self.acc_width;
        let row = &self.acc_w[start..start + self.acc_width];
        for (dst, src) in z.iter_mut().zip(row) {
            *dst += sign * *src;
        }
    }

    fn forward_from_z(&self, z: &[f32], summary: &[f32]) -> (f32, f32) {
        debug_assert_eq!(z.len(), self.acc_width);
        debug_assert_eq!(summary.len(), self.summary_size);
        let mut h0 = self.t0_b.clone();
        for (i, &value) in z.iter().enumerate() {
            let value = value.max(0.0);
            if value != 0.0 {
                let row = &self.t0_w_by_input[i * self.tail_hidden..(i + 1) * self.tail_hidden];
                scaled_add(&mut h0, row, value);
            }
        }
        for (i, &value) in summary.iter().enumerate() {
            if value != 0.0 {
                let input = self.acc_width + i;
                let row =
                    &self.t0_w_by_input[input * self.tail_hidden..(input + 1) * self.tail_hidden];
                scaled_add(&mut h0, row, value);
            }
        }
        h0.iter_mut().for_each(|v| *v = v.max(0.0));

        let mut h1 = self.t1_b.clone();
        for (i, &value) in h0.iter().enumerate() {
            if value != 0.0 {
                let row = &self.t1_w_by_input[i * self.tail_hidden..(i + 1) * self.tail_hidden];
                scaled_add(&mut h1, row, value);
            }
        }
        h1.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut logit = self.out_b;
        let mut margin = self.margin_b;
        for i in 0..self.tail_hidden {
            logit += self.out_w[i] * h1[i];
            margin += self.margin_w[i] * h1[i];
        }
        (1.0 / (1.0 + (-logit).exp()), margin)
    }

    fn value_from_state(&self, state: &RustGameState) -> Result<(f64, f32, f32), String> {
        let actor = state.actor().map_err(|e| e.to_string())?;
        let active = nnue_features::sparse_indices(state, actor)?;
        let summary = nnue_features::summary(state, actor)?;
        let z = self.accumulator(&active);
        let (expected, margin) = self.forward_from_z(&z, &summary);
        let actor_value = 2.0 * expected - 1.0;
        let p0 = if actor == 0 {
            actor_value
        } else {
            -actor_value
        };
        Ok((p0 as f64, expected, margin * self.margin_scale))
    }
}

fn transpose_output_major(weights: &[f32], outputs: usize, inputs: usize) -> Vec<f32> {
    debug_assert_eq!(weights.len(), outputs * inputs);
    let mut transposed = vec![0.0; weights.len()];
    for output in 0..outputs {
        for input in 0..inputs {
            transposed[input * outputs + output] = weights[output * inputs + input];
        }
    }
    transposed
}

#[inline(always)]
fn scaled_add(dst: &mut [f32], weights: &[f32], value: f32) {
    debug_assert_eq!(dst.len(), weights.len());
    for (out, &weight) in dst.iter_mut().zip(weights) {
        *out += weight * value;
    }
}

/// Post-training dynamic quantization of the frozen v3 float artifact.  The
/// sparse accumulator is int16; every dense-tail weight is int8 with a
/// per-output scale; accumulator/summary/tail activations are int16.  Biases and
/// the short dequantization steps remain float32.
pub(super) struct QuantizedSparseWeights {
    summary_size: usize,
    acc_width: usize,
    tail_hidden: usize,
    margin_scale: f32,
    acc_scale: Vec<f32>, // integer units per real unit, one per accumulator channel
    acc_bound: i32,
    acc_w: Vec<i16>,
    acc_b: Vec<i16>,
    // Dense rows remain output-major: the AVX2 kernel dots across contiguous inputs.
    t0_w: Vec<i8>,
    t0_scale: Vec<f32>, // accumulator-group + summary-group scale per output
    t0_b: Vec<f32>,
    t1_w: Vec<i8>,
    t1_scale: Vec<f32>,
    t1_b: Vec<f32>,
    out_w: Vec<i8>,
    out_scale: f32,
    out_b: f32,
    margin_w: Vec<i8>,
    margin_weight_scale: f32,
    margin_b: f32,
}

impl QuantizedSparseWeights {
    pub(super) fn load(path: &str) -> PyResult<Self> {
        let float = SparseNnueWeights::load(path)?;
        Self::from_float(&float).map_err(PyValueError::new_err)
    }

    fn from_float(float: &SparseNnueWeights) -> Result<Self, String> {
        let acc_scale = choose_accumulator_scales(float)?;
        let mut acc_w = Vec::with_capacity(float.acc_w.len());
        for row in float.acc_w.chunks_exact(float.acc_width) {
            acc_w.extend(
                row.iter()
                    .zip(&acc_scale)
                    .map(|(&v, &scale)| quantize_i16(v, scale)),
            );
        }
        let acc_b: Vec<i16> = float
            .acc_b
            .iter()
            .zip(&acc_scale)
            .map(|(&v, &scale)| quantize_i16(v, scale))
            .collect();
        let acc_bound = exact_accumulator_bound(&acc_w, &acc_b, float.acc_width);
        if acc_bound > i16::MAX as i32 {
            return Err(format!(
                "quantized accumulator bound {acc_bound} exceeds int16"
            ));
        }

        let tail_input = float.acc_width + float.summary_size;
        let mut t0_effective =
            output_major_from_input_major(&float.t0_w_by_input, float.tail_hidden, tail_input);
        let t1_float = output_major_from_input_major(
            &float.t1_w_by_input,
            float.tail_hidden,
            float.tail_hidden,
        );
        // Fold each input's activation scale into tail0 before int8
        // quantization. The integer dot then directly approximates the real
        // contribution even though accumulator channels use different scales.
        for row in t0_effective.chunks_exact_mut(tail_input) {
            for (i, weight) in row.iter_mut().enumerate() {
                *weight /= if i < float.acc_width {
                    acc_scale[i]
                } else {
                    SUMMARY_QUANT_SCALE
                };
            }
        }
        let (t0_w, t0_scale) = quantize_i8_rows_two_groups(
            &t0_effective,
            float.tail_hidden,
            tail_input,
            float.acc_width,
        );
        let (t1_w, t1_scale) = quantize_i8_rows(&t1_float, float.tail_hidden, float.tail_hidden);
        let (out_w, out_scale) = quantize_i8_vector(&float.out_w);
        let (margin_w, margin_weight_scale) = quantize_i8_vector(&float.margin_w);
        Ok(Self {
            summary_size: float.summary_size,
            acc_width: float.acc_width,
            tail_hidden: float.tail_hidden,
            margin_scale: float.margin_scale,
            acc_scale,
            acc_bound,
            acc_w,
            acc_b,
            t0_w,
            t0_scale,
            t0_b: float.t0_b.clone(),
            t1_w,
            t1_scale,
            t1_b: float.t1_b.clone(),
            out_w,
            out_scale,
            out_b: float.out_b,
            margin_w,
            margin_weight_scale,
            margin_b: float.margin_b,
        })
    }

    fn accumulator(&self, active: &[i32]) -> Result<Vec<i16>, String> {
        if active.len() > MAX_ACTIVE_FEATURES {
            return Err(format!(
                "{} active NNUE features exceeds quantized bound {MAX_ACTIVE_FEATURES}",
                active.len()
            ));
        }
        let mut sums: Vec<i32> = self.acc_b.iter().map(|&v| v as i32).collect();
        for &feature in active {
            let start = feature as usize * self.acc_width;
            for (dst, &src) in sums
                .iter_mut()
                .zip(&self.acc_w[start..start + self.acc_width])
            {
                *dst += src as i32;
            }
        }
        sums.into_iter()
            .map(|v| i16::try_from(v).map_err(|_| format!("quantized accumulator overflow: {v}")))
            .collect()
    }

    fn add_feature(&self, z: &mut [i16], feature: i32, sign: i32) {
        let start = feature as usize * self.acc_width;
        for (dst, &src) in z.iter_mut().zip(&self.acc_w[start..start + self.acc_width]) {
            let value = *dst as i32 + sign * src as i32;
            debug_assert!((i16::MIN as i32..=i16::MAX as i32).contains(&value));
            *dst = value as i16;
        }
    }

    fn forward_from_z(&self, z: &[i16], summary: &[f32]) -> (f32, f32) {
        debug_assert_eq!(z.len(), self.acc_width);
        debug_assert_eq!(summary.len(), self.summary_size);
        let summary_q: Vec<i16> = summary
            .iter()
            .map(|&v| quantize_i16(v.clamp(-1.0, 1.0), SUMMARY_QUANT_SCALE))
            .collect();
        let tail_input = self.acc_width + self.summary_size;
        let mut tail_q = Vec::with_capacity(tail_input);
        tail_q.extend(z.iter().map(|&value| value.max(0)));
        tail_q.extend_from_slice(&summary_q);
        let mut h0 = vec![0.0f32; self.tail_hidden];
        for (o, out) in h0.iter_mut().enumerate() {
            let row = &self.t0_w[o * tail_input..(o + 1) * tail_input];
            let value = self.t0_b[o]
                + dot_i16_i8(&tail_q[..self.acc_width], &row[..self.acc_width]) as f32
                    * self.t0_scale[o * 2]
                + dot_i16_i8(&tail_q[self.acc_width..], &row[self.acc_width..]) as f32
                    * self.t0_scale[o * 2 + 1];
            *out = value.max(0.0);
        }
        let (h0_q, h0_scale) = dynamic_quantize_nonnegative(&h0);

        let mut h1 = vec![0.0f32; self.tail_hidden];
        for (o, out) in h1.iter_mut().enumerate() {
            let row = &self.t1_w[o * self.tail_hidden..(o + 1) * self.tail_hidden];
            let value = self.t1_b[o] + dot_i16_i8(&h0_q, row) as f32 * self.t1_scale[o] / h0_scale;
            *out = value.max(0.0);
        }
        let (h1_q, h1_scale) = dynamic_quantize_nonnegative(&h1);
        let logit = self.out_b + dot_i16_i8(&h1_q, &self.out_w) as f32 * self.out_scale / h1_scale;
        let margin = self.margin_b
            + dot_i16_i8(&h1_q, &self.margin_w) as f32 * self.margin_weight_scale / h1_scale;
        (1.0 / (1.0 + (-logit).exp()), margin)
    }

    fn value_from_state(&self, state: &RustGameState) -> Result<(f64, f32, f32), String> {
        let actor = state.actor().map_err(|e| e.to_string())?;
        let active = nnue_features::sparse_indices(state, actor)?;
        let summary = nnue_features::summary(state, actor)?;
        let z = self.accumulator(&active)?;
        let (expected, margin) = self.forward_from_z(&z, &summary);
        let actor_value = 2.0 * expected - 1.0;
        let p0 = if actor == 0 {
            actor_value
        } else {
            -actor_value
        };
        Ok((p0 as f64, expected, margin * self.margin_scale))
    }
}

fn choose_accumulator_scales(float: &SparseNnueWeights) -> Result<Vec<f32>, String> {
    let mut scales = Vec::with_capacity(float.acc_width);
    let mut values = Vec::with_capacity(nnue_features::CORE_SIZE);
    for dim in 0..float.acc_width {
        values.clear();
        values.extend(
            (0..nnue_features::CORE_SIZE)
                .map(|feature| float.acc_w[feature * float.acc_width + dim].abs()),
        );
        values.sort_unstable_by(|a, b| b.total_cmp(a));
        let bound = float.acc_b[dim].abs()
            + values
                .iter()
                .take(MAX_ACTIVE_FEATURES)
                .copied()
                .sum::<f32>();
        if !bound.is_finite() || bound <= 0.0 {
            return Err(format!("invalid float accumulator bound in channel {dim}"));
        }
        // Rounding each selected row can add at most 0.5 integer units.
        let rounding_guard = 0.5 * (MAX_ACTIVE_FEATURES + 1) as f32;
        let safe = (i16::MAX as f32 - rounding_guard) / bound;
        let mut scale = 1.0f32;
        while scale * 2.0 <= safe && scale < 4096.0 {
            scale *= 2.0;
        }
        if scale < 1.0 {
            return Err(format!(
                "float accumulator channel {dim} bound {bound} cannot fit int16"
            ));
        }
        scales.push(scale);
    }
    Ok(scales)
}

fn exact_accumulator_bound(weights: &[i16], bias: &[i16], width: usize) -> i32 {
    let mut max_bound = 0i32;
    let mut values = Vec::with_capacity(nnue_features::CORE_SIZE);
    for dim in 0..width {
        values.clear();
        values.extend(
            (0..nnue_features::CORE_SIZE)
                .map(|feature| (weights[feature * width + dim] as i32).unsigned_abs() as i32),
        );
        values.sort_unstable_by(|a, b| b.cmp(a));
        let bound = (bias[dim] as i32).unsigned_abs() as i32
            + values.iter().take(MAX_ACTIVE_FEATURES).sum::<i32>();
        max_bound = max_bound.max(bound);
    }
    max_bound
}

#[inline]
fn quantize_i16(value: f32, units_per_real: f32) -> i16 {
    (value * units_per_real)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn quantize_i8_rows(values: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut out = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(rows);
    for row in values.chunks_exact(cols) {
        let (q, scale) = quantize_i8_vector(row);
        out.extend(q);
        scales.push(scale);
    }
    (out, scales)
}

fn quantize_i8_rows_two_groups(
    values: &[f32],
    rows: usize,
    cols: usize,
    split: usize,
) -> (Vec<i8>, Vec<f32>) {
    debug_assert!(split > 0 && split < cols);
    let mut out = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(rows * 2);
    for row in values.chunks_exact(cols) {
        let (left, left_scale) = quantize_i8_vector(&row[..split]);
        let (right, right_scale) = quantize_i8_vector(&row[split..]);
        out.extend(left);
        out.extend(right);
        scales.extend([left_scale, right_scale]);
    }
    (out, scales)
}

fn quantize_i8_vector(values: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let real_per_integer = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let out = values
        .iter()
        .map(|&v| (v / real_per_integer).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (out, real_per_integer)
}

fn output_major_from_input_major(values: &[f32], outputs: usize, inputs: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for input in 0..inputs {
        for output in 0..outputs {
            out[output * inputs + input] = values[input * outputs + output];
        }
    }
    out
}

fn dynamic_quantize_nonnegative(values: &[f32]) -> (Vec<i16>, f32) {
    let max = values.iter().copied().fold(0.0f32, f32::max);
    let units_per_real = if max > 0.0 {
        i16::MAX as f32 / max
    } else {
        1.0
    };
    let quantized = values
        .iter()
        .map(|&v| quantize_i16(v.max(0.0), units_per_real))
        .collect();
    (quantized, units_per_real)
}

#[inline]
fn dot_i16_i8(a: &[i16], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by runtime AVX2 detection; the implementation uses
        // unaligned loads and remains within both slices.
        return unsafe { dot_i16_i8_avx2(a, b) };
    }
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i16_i8_avx2(a: &[i16], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 16 <= a.len() {
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb8 = unsafe { _mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>()) };
        let vb = _mm256_cvtepi8_epi16(vb8);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
        i += 16;
    }
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };
    let mut sum: i32 = lanes.into_iter().sum();
    while i < a.len() {
        sum += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i16_i8_avx2(a: &[i16], b: &[i8]) -> i32 {
    use std::arch::x86::*;
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 16 <= a.len() {
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb8 = unsafe { _mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>()) };
        let vb = _mm256_cvtepi8_epi16(vb8);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
        i += 16;
    }
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };
    let mut sum: i32 = lanes.into_iter().sum();
    while i < a.len() {
        sum += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    sum
}

#[derive(Clone)]
struct AccumulatorSnapshot {
    active: [Vec<i32>; 2],
    z: [Vec<f32>; 2],
}

#[derive(Clone)]
struct DualAccumulator {
    active: [Vec<i32>; 2],
    z: [Vec<f32>; 2],
}

impl DualAccumulator {
    fn refresh(state: &RustGameState, weights: &SparseNnueWeights) -> Result<Self, String> {
        let a0 = nnue_features::sparse_indices(state, 0)?;
        let a1 = nnue_features::sparse_indices(state, 1)?;
        let z0 = weights.accumulator(&a0);
        let z1 = weights.accumulator(&a1);
        Ok(Self {
            active: [a0, a1],
            z: [z0, z1],
        })
    }

    fn restore(&mut self, snapshot: AccumulatorSnapshot) {
        self.active = snapshot.active;
        self.z = snapshot.z;
    }

    fn derive_next(
        state: &RustGameState,
        undo: &UndoRecord,
        current: &[Vec<i32>; 2],
    ) -> Result<[Vec<i32>; 2], String> {
        // Board features are monotonic additions: read the two newly written
        // cells from the placement undo. Every other bank is small and dynamic,
        // so re-derive it after the move; this automatically handles sampled
        // chance rows, round promotion, actor/slot changes, and forced discards.
        let placement = match undo {
            UndoRecord::Move {
                player,
                place: Some(place),
                ..
            } => Some((*player, [place.i1, place.i2])),
            _ => None,
        };
        let mut next = [Vec::with_capacity(160), Vec::with_capacity(160)];
        for perspective in 0..2u8 {
            next[perspective as usize].extend(
                current[perspective as usize]
                    .iter()
                    .take_while(|&&i| i < nnue_features::BOARD_FEATURE_END)
                    .copied(),
            );
            if let Some((owner, cells)) = placement {
                for cell in cells {
                    let x = (cell % N) as i8;
                    let y = (cell / N) as i8;
                    next[perspective as usize].push(nnue_features::board_feature_index(
                        perspective,
                        owner,
                        x,
                        y,
                        state.boards[owner as usize].terrain[cell],
                        state.boards[owner as usize].crowns[cell],
                    )?);
                }
            }
            next[perspective as usize]
                .extend(nnue_features::non_board_indices(state, perspective)?);
            next[perspective as usize].sort_unstable();
        }
        Ok(next)
    }

    fn transition_to(
        &mut self,
        next: [Vec<i32>; 2],
        weights: &SparseNnueWeights,
    ) -> AccumulatorSnapshot {
        // Snapshot only z. The old active vectors are moved (not cloned) into the
        // undo record after the diff, removing two allocations from every edge.
        let old_z = self.z.clone();
        for perspective in 0..2 {
            let old = &self.active[perspective];
            let new = &next[perspective];
            let mut i = 0;
            let mut j = 0;
            while i < old.len() || j < new.len() {
                if j == new.len() || (i < old.len() && old[i] < new[j]) {
                    weights.add_feature(&mut self.z[perspective], old[i], -1.0);
                    i += 1;
                } else if i == old.len() || new[j] < old[i] {
                    weights.add_feature(&mut self.z[perspective], new[j], 1.0);
                    j += 1;
                } else {
                    i += 1;
                    j += 1;
                }
            }
        }
        let old_active = std::mem::replace(&mut self.active, next);
        AccumulatorSnapshot {
            active: old_active,
            z: old_z,
        }
    }
}

pub(super) struct SparseStatelessEval {
    pub(super) weights: Arc<SparseNnueWeights>,
}

impl Eval<Kingdomino> for SparseStatelessEval {
    fn eval(&self, state: &RustGameState) -> f64 {
        self.weights
            .value_from_state(state)
            .map(|x| x.0)
            .expect("stateless sparse NNUE forward failed")
    }
}

pub(super) struct SparseSearchState {
    pub(super) game: RustGameState,
    weights: Arc<SparseNnueWeights>,
    accumulator: DualAccumulator,
}

impl SparseSearchState {
    pub(super) fn new(game: RustGameState, weights: Arc<SparseNnueWeights>) -> PyResult<Self> {
        let accumulator =
            DualAccumulator::refresh(&game, &weights).map_err(PyValueError::new_err)?;
        Ok(Self {
            game,
            weights,
            accumulator,
        })
    }
}

pub(super) struct SparseUndo {
    game: UndoRecord,
    accumulator: AccumulatorSnapshot,
}

pub(super) struct SparseKingdomino;

impl Game for SparseKingdomino {
    type State = SparseSearchState;
    type Action = <Kingdomino as Game>::Action;
    type Chance = <Kingdomino as Game>::Chance;
    type Undo = SparseUndo;

    fn to_move(s: &Self::State) -> PyResult<search::Turn> {
        <Kingdomino as Game>::to_move(&s.game)
    }
    fn is_terminal(s: &Self::State) -> bool {
        <Kingdomino as Game>::is_terminal(&s.game)
    }
    fn terminal_value_p0(s: &Self::State) -> f64 {
        <Kingdomino as Game>::terminal_value_p0(&s.game)
    }
    fn bounded_margin(s: &Self::State) -> f64 {
        <Kingdomino as Game>::bounded_margin(&s.game)
    }
    fn legal_actions(s: &Self::State, out: &mut Vec<Self::Action>) {
        <Kingdomino as Game>::legal_actions(&s.game, out)
    }
    fn is_stochastic(s: &Self::State, a: Self::Action) -> bool {
        <Kingdomino as Game>::is_stochastic(&s.game, a)
    }
    fn chance_children(
        s: &Self::State,
        a: Self::Action,
        cfg: &search::SearchConfig,
    ) -> Vec<(Self::Chance, f64)> {
        <Kingdomino as Game>::chance_children(&s.game, a, cfg)
    }
    fn make(s: &mut Self::State, a: Self::Action) -> PyResult<Self::Undo> {
        let game = <Kingdomino as Game>::make(&mut s.game, a)?;
        let next = match DualAccumulator::derive_next(&s.game, &game, &s.accumulator.active) {
            Ok(next) => next,
            Err(err) => {
                <Kingdomino as Game>::unmake(&mut s.game, game);
                return Err(PyValueError::new_err(err));
            }
        };
        let snapshot = s.accumulator.transition_to(next, &s.weights);
        Ok(SparseUndo {
            game,
            accumulator: snapshot,
        })
    }
    fn make_with_chance(
        s: &mut Self::State,
        a: Self::Action,
        c: &Self::Chance,
    ) -> PyResult<Self::Undo> {
        let game = <Kingdomino as Game>::make_with_chance(&mut s.game, a, c)?;
        let next = match DualAccumulator::derive_next(&s.game, &game, &s.accumulator.active) {
            Ok(next) => next,
            Err(err) => {
                <Kingdomino as Game>::unmake(&mut s.game, game);
                return Err(PyValueError::new_err(err));
            }
        };
        let snapshot = s.accumulator.transition_to(next, &s.weights);
        Ok(SparseUndo {
            game,
            accumulator: snapshot,
        })
    }
    fn unmake(s: &mut Self::State, undo: Self::Undo) {
        <Kingdomino as Game>::unmake(&mut s.game, undo.game);
        s.accumulator.restore(undo.accumulator);
    }

    fn exact_remaining_plies(s: &Self::State) -> Option<u32> {
        <Kingdomino as Game>::exact_remaining_plies(&s.game)
    }

    fn position_key(s: &Self::State, scratch: &mut Vec<u8>) -> Option<u128> {
        <Kingdomino as Game>::position_key(&s.game, scratch)
    }
}

pub(super) struct SparseIncrementalEval;

impl Eval<SparseKingdomino> for SparseIncrementalEval {
    fn eval(&self, state: &SparseSearchState) -> f64 {
        let actor = state.game.actor().expect("incremental eval on terminal");
        let summary = nnue_features::summary(&state.game, actor)
            .expect("incremental sparse NNUE summary failed");
        let (expected, _) = state
            .weights
            .forward_from_z(&state.accumulator.z[actor as usize], &summary);
        let actor_value = 2.0 * expected - 1.0;
        if actor == 0 {
            actor_value as f64
        } else {
            -actor_value as f64
        }
    }
}

#[derive(Clone)]
struct QuantizedAccumulatorSnapshot {
    active: [Vec<i32>; 2],
    z: [Vec<i16>; 2],
}

#[derive(Clone)]
struct QuantizedDualAccumulator {
    active: [Vec<i32>; 2],
    z: [Vec<i16>; 2],
}

impl QuantizedDualAccumulator {
    fn refresh(state: &RustGameState, weights: &QuantizedSparseWeights) -> Result<Self, String> {
        let a0 = nnue_features::sparse_indices(state, 0)?;
        let a1 = nnue_features::sparse_indices(state, 1)?;
        let z0 = weights.accumulator(&a0)?;
        let z1 = weights.accumulator(&a1)?;
        Ok(Self {
            active: [a0, a1],
            z: [z0, z1],
        })
    }

    fn restore(&mut self, snapshot: QuantizedAccumulatorSnapshot) {
        self.active = snapshot.active;
        self.z = snapshot.z;
    }

    fn transition_to(
        &mut self,
        next: [Vec<i32>; 2],
        weights: &QuantizedSparseWeights,
    ) -> Result<QuantizedAccumulatorSnapshot, String> {
        if next.iter().any(|active| active.len() > MAX_ACTIVE_FEATURES) {
            return Err(format!(
                "quantized transition exceeds {MAX_ACTIVE_FEATURES} active features"
            ));
        }
        let old_z = self.z.clone();
        for perspective in 0..2 {
            let old = &self.active[perspective];
            let new = &next[perspective];
            // Removals first, then additions. Every intermediate is therefore a
            // subset of old or new and remains covered by the top-K int16 proof.
            let mut i = 0;
            let mut j = 0;
            while i < old.len() {
                while j < new.len() && new[j] < old[i] {
                    j += 1;
                }
                if j == new.len() || old[i] < new[j] {
                    weights.add_feature(&mut self.z[perspective], old[i], -1);
                }
                i += 1;
            }
            i = 0;
            j = 0;
            while j < new.len() {
                while i < old.len() && old[i] < new[j] {
                    i += 1;
                }
                if i == old.len() || new[j] < old[i] {
                    weights.add_feature(&mut self.z[perspective], new[j], 1);
                }
                j += 1;
            }
        }
        let old_active = std::mem::replace(&mut self.active, next);
        Ok(QuantizedAccumulatorSnapshot {
            active: old_active,
            z: old_z,
        })
    }
}

pub(super) struct QuantizedSparseSearchState {
    pub(super) game: RustGameState,
    weights: Arc<QuantizedSparseWeights>,
    accumulator: QuantizedDualAccumulator,
}

impl QuantizedSparseSearchState {
    pub(super) fn new(game: RustGameState, weights: Arc<QuantizedSparseWeights>) -> PyResult<Self> {
        let accumulator =
            QuantizedDualAccumulator::refresh(&game, &weights).map_err(PyValueError::new_err)?;
        Ok(Self {
            game,
            weights,
            accumulator,
        })
    }
}

pub(super) struct QuantizedSparseUndo {
    game: UndoRecord,
    accumulator: QuantizedAccumulatorSnapshot,
}

pub(super) struct QuantizedSparseKingdomino;

impl Game for QuantizedSparseKingdomino {
    type State = QuantizedSparseSearchState;
    type Action = <Kingdomino as Game>::Action;
    type Chance = <Kingdomino as Game>::Chance;
    type Undo = QuantizedSparseUndo;

    fn to_move(s: &Self::State) -> PyResult<search::Turn> {
        <Kingdomino as Game>::to_move(&s.game)
    }
    fn is_terminal(s: &Self::State) -> bool {
        <Kingdomino as Game>::is_terminal(&s.game)
    }
    fn terminal_value_p0(s: &Self::State) -> f64 {
        <Kingdomino as Game>::terminal_value_p0(&s.game)
    }
    fn bounded_margin(s: &Self::State) -> f64 {
        <Kingdomino as Game>::bounded_margin(&s.game)
    }
    fn legal_actions(s: &Self::State, out: &mut Vec<Self::Action>) {
        <Kingdomino as Game>::legal_actions(&s.game, out)
    }
    fn is_stochastic(s: &Self::State, a: Self::Action) -> bool {
        <Kingdomino as Game>::is_stochastic(&s.game, a)
    }
    fn chance_children(
        s: &Self::State,
        a: Self::Action,
        cfg: &search::SearchConfig,
    ) -> Vec<(Self::Chance, f64)> {
        <Kingdomino as Game>::chance_children(&s.game, a, cfg)
    }
    fn make(s: &mut Self::State, a: Self::Action) -> PyResult<Self::Undo> {
        let game = <Kingdomino as Game>::make(&mut s.game, a)?;
        let next = match DualAccumulator::derive_next(&s.game, &game, &s.accumulator.active) {
            Ok(next) => next,
            Err(err) => {
                <Kingdomino as Game>::unmake(&mut s.game, game);
                return Err(PyValueError::new_err(err));
            }
        };
        let accumulator = match s.accumulator.transition_to(next, &s.weights) {
            Ok(snapshot) => snapshot,
            Err(err) => {
                <Kingdomino as Game>::unmake(&mut s.game, game);
                return Err(PyValueError::new_err(err));
            }
        };
        Ok(QuantizedSparseUndo { game, accumulator })
    }
    fn make_with_chance(
        s: &mut Self::State,
        a: Self::Action,
        c: &Self::Chance,
    ) -> PyResult<Self::Undo> {
        let game = <Kingdomino as Game>::make_with_chance(&mut s.game, a, c)?;
        let next = match DualAccumulator::derive_next(&s.game, &game, &s.accumulator.active) {
            Ok(next) => next,
            Err(err) => {
                <Kingdomino as Game>::unmake(&mut s.game, game);
                return Err(PyValueError::new_err(err));
            }
        };
        let accumulator = match s.accumulator.transition_to(next, &s.weights) {
            Ok(snapshot) => snapshot,
            Err(err) => {
                <Kingdomino as Game>::unmake(&mut s.game, game);
                return Err(PyValueError::new_err(err));
            }
        };
        Ok(QuantizedSparseUndo { game, accumulator })
    }
    fn unmake(s: &mut Self::State, undo: Self::Undo) {
        <Kingdomino as Game>::unmake(&mut s.game, undo.game);
        s.accumulator.restore(undo.accumulator);
    }

    fn exact_remaining_plies(s: &Self::State) -> Option<u32> {
        <Kingdomino as Game>::exact_remaining_plies(&s.game)
    }

    fn position_key(s: &Self::State, scratch: &mut Vec<u8>) -> Option<u128> {
        <Kingdomino as Game>::position_key(&s.game, scratch)
    }
}

pub(super) struct QuantizedSparseEval;

impl Eval<QuantizedSparseKingdomino> for QuantizedSparseEval {
    fn eval(&self, state: &QuantizedSparseSearchState) -> f64 {
        let actor = state.game.actor().expect("quantized eval on terminal");
        let summary = nnue_features::summary(&state.game, actor)
            .expect("quantized sparse NNUE summary failed");
        let (expected, _) = state
            .weights
            .forward_from_z(&state.accumulator.z[actor as usize], &summary);
        let actor_value = 2.0 * expected - 1.0;
        if actor == 0 {
            actor_value as f64
        } else {
            -actor_value as f64
        }
    }
}

#[pyclass]
pub(super) struct QuantizedSparseNnueEvaluator {
    weights: Arc<QuantizedSparseWeights>,
}

#[pymethods]
impl QuantizedSparseNnueEvaluator {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            weights: Arc::new(QuantizedSparseWeights::load(path)?),
        })
    }

    fn evaluate(&self, state: &RustGameState) -> PyResult<(f64, f64, f64)> {
        let (p0, expected, margin) = self
            .weights
            .value_from_state(state)
            .map_err(PyValueError::new_err)?;
        Ok((p0, expected as f64, margin as f64))
    }

    /// `(min_acc_scale, max_acc_scale, conservative_abs_bound,
    /// max_active_features, avx2)`.
    fn quantization_info(&self) -> (f64, f64, i32, usize, bool) {
        (
            self.weights
                .acc_scale
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min) as f64,
            self.weights
                .acc_scale
                .iter()
                .copied()
                .fold(0.0f32, f32::max) as f64,
            self.weights.acc_bound,
            MAX_ACTIVE_FEATURES,
            std::arch::is_x86_feature_detected!("avx2"),
        )
    }
}

#[pyclass]
pub(super) struct SparseNnueEvaluator {
    weights: Arc<SparseNnueWeights>,
}

#[pymethods]
impl SparseNnueEvaluator {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            weights: Arc::new(SparseNnueWeights::load(path)?),
        })
    }

    /// Stateless reference evaluation: (p0_value, actor expected score, actor margin points).
    fn evaluate(&self, state: &RustGameState) -> PyResult<(f64, f64, f64)> {
        let (p0, expected, margin) = self
            .weights
            .value_from_state(state)
            .map_err(PyValueError::new_err)?;
        Ok((p0, expected as f64, margin as f64))
    }

    /// Micro-profile one state. Returns seconds for
    /// `(sparse_indices+accumulator_sum, summary, tail)` over `iterations`.
    fn benchmark_components(
        &self,
        state: &RustGameState,
        iterations: usize,
    ) -> PyResult<(f64, f64, f64)> {
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be >= 1"));
        }
        let actor = state.actor()?;
        let active = nnue_features::sparse_indices(state, actor).map_err(PyValueError::new_err)?;
        let summary = nnue_features::summary(state, actor).map_err(PyValueError::new_err)?;
        let z = self.weights.accumulator(&active);

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let a = nnue_features::sparse_indices(state, actor).map_err(PyValueError::new_err)?;
            std::hint::black_box(self.weights.accumulator(&a));
        }
        let feature_secs = start.elapsed().as_secs_f64();

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(
                nnue_features::summary(state, actor).map_err(PyValueError::new_err)?,
            );
        }
        let summary_secs = start.elapsed().as_secs_f64();

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(self.weights.forward_from_z(&z, &summary));
        }
        let tail_secs = start.elapsed().as_secs_f64();
        Ok((feature_secs, summary_secs, tail_secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{new_game, solver_state_bytes};

    fn weights() -> Arc<SparseNnueWeights> {
        let aw = 8;
        let th = 4;
        let patterned = |n: usize, scale: f32| {
            (0..n)
                .map(|i| (((i * 37 + 11) % 101) as f32 - 50.0) * scale)
                .collect()
        };
        Arc::new(SparseNnueWeights {
            summary_size: nnue_features::SUMMARY_SIZE,
            acc_width: aw,
            tail_hidden: th,
            margin_scale: 40.0,
            acc_w: patterned(nnue_features::CORE_SIZE * aw, 0.0002),
            acc_b: patterned(aw, 0.001),
            t0_w_by_input: transpose_output_major(
                &patterned(th * (aw + nnue_features::SUMMARY_SIZE), 0.0005),
                th,
                aw + nnue_features::SUMMARY_SIZE,
            ),
            t0_b: patterned(th, 0.001),
            t1_w_by_input: transpose_output_major(&patterned(th * th, 0.001), th, th),
            t1_b: patterned(th, 0.001),
            out_w: patterned(th, 0.002),
            out_b: 0.03,
            margin_w: patterned(th, 0.002),
            margin_b: -0.02,
        })
    }

    fn max_refresh_error(s: &SparseSearchState) -> f32 {
        let fresh = DualAccumulator::refresh(&s.game, &s.weights).unwrap();
        s.accumulator
            .z
            .iter()
            .flatten()
            .zip(fresh.z.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn quantized_weights() -> Arc<QuantizedSparseWeights> {
        Arc::new(QuantizedSparseWeights::from_float(&weights()).unwrap())
    }

    #[test]
    fn quantized_dot_matches_scalar_and_bound_is_safe() {
        for len in 0..96usize {
            let a: Vec<i16> = (0..len)
                .map(|i| ((i * 7919 + 17) % 65535) as i32 - 32767)
                .map(|v| v as i16)
                .collect();
            let b: Vec<i8> = (0..len)
                .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
                .collect();
            let scalar: i32 = a.iter().zip(&b).map(|(&x, &y)| x as i32 * y as i32).sum();
            assert_eq!(dot_i16_i8(&a, &b), scalar, "length {len}");
        }
        let q = quantized_weights();
        assert!(q.acc_bound <= i16::MAX as i32);
        assert_eq!(q.acc_scale.len(), q.acc_width);
    }

    #[test]
    fn quantized_incremental_full_playout_and_unwind() -> PyResult<()> {
        let weights = quantized_weights();
        let mut s = QuantizedSparseSearchState::new(new_game(29, true, true), weights.clone())?;
        let root_acc = s.accumulator.clone();
        let mut root_bytes = Vec::new();
        solver_state_bytes(&s.game, &mut root_bytes);
        let mut stack = Vec::new();
        let mut seed = 41u64;
        while !<QuantizedSparseKingdomino as Game>::is_terminal(&s) {
            let mut actions = Vec::new();
            <QuantizedSparseKingdomino as Game>::legal_actions(&s, &mut actions);
            let pick = search::splitmix64(&mut seed) as usize % actions.len();
            let undo = <QuantizedSparseKingdomino as Game>::make(&mut s, actions[pick])?;
            for perspective in 0..2 {
                assert_eq!(
                    s.accumulator.active[perspective],
                    nnue_features::sparse_indices(&s.game, perspective as u8).unwrap()
                );
                assert_eq!(
                    s.accumulator.z[perspective],
                    weights
                        .accumulator(&s.accumulator.active[perspective])
                        .unwrap()
                );
            }
            stack.push(undo);
        }
        while let Some(undo) = stack.pop() {
            <QuantizedSparseKingdomino as Game>::unmake(&mut s, undo);
        }
        let mut end_bytes = Vec::new();
        solver_state_bytes(&s.game, &mut end_bytes);
        assert_eq!(root_bytes, end_bytes);
        assert_eq!(root_acc.active, s.accumulator.active);
        assert_eq!(root_acc.z, s.accumulator.z);
        Ok(())
    }

    #[test]
    fn incremental_full_playout_and_unwind() -> PyResult<()> {
        let mut s = SparseSearchState::new(new_game(17, true, true), weights())?;
        let root_acc = s.accumulator.clone();
        let mut root_bytes = Vec::new();
        solver_state_bytes(&s.game, &mut root_bytes);
        let mut stack = Vec::new();
        let mut seed = 9u64;
        while !<SparseKingdomino as Game>::is_terminal(&s) {
            let mut actions = Vec::new();
            <SparseKingdomino as Game>::legal_actions(&s, &mut actions);
            let pick = search::splitmix64(&mut seed) as usize % actions.len();
            let undo = <SparseKingdomino as Game>::make(&mut s, actions[pick])?;
            assert_eq!(
                s.accumulator.active[0],
                nnue_features::sparse_indices(&s.game, 0).unwrap()
            );
            assert_eq!(
                s.accumulator.active[1],
                nnue_features::sparse_indices(&s.game, 1).unwrap()
            );
            assert!(max_refresh_error(&s) < 2e-5);
            stack.push(undo);
        }
        while let Some(undo) = stack.pop() {
            <SparseKingdomino as Game>::unmake(&mut s, undo);
        }
        let mut end_bytes = Vec::new();
        solver_state_bytes(&s.game, &mut end_bytes);
        assert_eq!(root_bytes, end_bytes);
        assert_eq!(root_acc.active, s.accumulator.active);
        assert_eq!(root_acc.z, s.accumulator.z);
        Ok(())
    }

    #[test]
    fn sampled_chance_update_matches_refresh() -> PyResult<()> {
        let cfg = search::SearchConfig {
            depth: 2,
            chance_samples: 4,
            enum_cap: 1,
            margin_weight: 0.0,
            seed: 5,
        };
        let mut s = SparseSearchState::new(new_game(3, true, true), weights())?;
        loop {
            let mut actions = Vec::new();
            <SparseKingdomino as Game>::legal_actions(&s, &mut actions);
            if actions
                .iter()
                .any(|&a| <SparseKingdomino as Game>::is_stochastic(&s, a))
            {
                let a = actions
                    .into_iter()
                    .find(|&a| <SparseKingdomino as Game>::is_stochastic(&s, a))
                    .unwrap();
                let chance = <SparseKingdomino as Game>::chance_children(&s, a, &cfg);
                let before = s.accumulator.clone();
                let undo = <SparseKingdomino as Game>::make_with_chance(&mut s, a, &chance[0].0)?;
                assert_eq!(
                    s.accumulator.active[0],
                    nnue_features::sparse_indices(&s.game, 0).unwrap()
                );
                assert_eq!(
                    s.accumulator.active[1],
                    nnue_features::sparse_indices(&s.game, 1).unwrap()
                );
                assert!(max_refresh_error(&s) < 2e-5);
                <SparseKingdomino as Game>::unmake(&mut s, undo);
                assert_eq!(before.active, s.accumulator.active);
                assert_eq!(before.z, s.accumulator.z);
                break;
            }
            let _undo = <SparseKingdomino as Game>::make(&mut s, actions[0])?;
            // Dropping the record does not undo; this test intentionally owns and
            // advances the prefix until it reaches a stochastic boundary.
        }
        Ok(())
    }

    struct NanEval;
    impl Eval<SparseKingdomino> for NanEval {
        fn eval(&self, _s: &SparseSearchState) -> f64 {
            f64::NAN
        }
    }

    #[test]
    fn search_error_restores_game_and_accumulator() -> PyResult<()> {
        let mut s = SparseSearchState::new(new_game(21, true, true), weights())?;
        let root_acc = s.accumulator.clone();
        let mut root_bytes = Vec::new();
        solver_state_bytes(&s.game, &mut root_bytes);
        let cfg = search::SearchConfig {
            depth: 1,
            chance_samples: 4,
            enum_cap: 128,
            margin_weight: 0.0,
            seed: 0,
        };
        let mut nodes = 0;
        assert!(
            search::choose_action::<SparseKingdomino, _>(&mut s, &NanEval, &cfg, None, &mut nodes)
                .is_err()
        );
        let mut end_bytes = Vec::new();
        solver_state_bytes(&s.game, &mut end_bytes);
        assert_eq!(root_bytes, end_bytes);
        assert_eq!(root_acc.active, s.accumulator.active);
        assert_eq!(root_acc.z, s.accumulator.z);
        Ok(())
    }
}
