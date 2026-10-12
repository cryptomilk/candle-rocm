//! `block_q8_1_mmq` activation quantization for the AMD WMMA path.
//!
//! [`super::mmq_wmma`] needs activations in `mmq_gguf`'s `block_q8_1_mmq`
//! layout (144 bytes per 128 values, four `block_q8_1`-equivalent sub-blocks
//! sharing one scale), not the plain `block_q8_1` (34 bytes per 32 values)
//! [`super::q8_1`] produces for the dp4a path. `mmq_quantize.cu` (compiled as
//! its own `mmq_gguf` shared library unit) produces this layout directly
//! from f32 activations.

use super::kernels::MATRIX_ROW_PADDING;
use super::q8_1::pad;
use crate::quantized::GgmlDType;
use crate::rocm_backend::{RocmDevice, SendSyncDeviceMemory, WrapErr};
use crate::Result;
use candle_rocm_kernels::MmqGgufUnit;
use std::ffi::c_void;

/// `sizeof(block_q8_1_mmq)`: four `block_q8_1`-equivalent sub-blocks (128
/// values) sharing one 16-byte scale/sum union, regardless of which scale
/// layout (`D4`/`DS4`/`D2S6`) the union holds.
const BLOCK_BYTES: usize = 144;
/// Values per `block_q8_1_mmq`.
const BLOCK_VALUES: usize = 128;

/// Bytes a `block_q8_1_mmq` buffer of `rows` rows of `k` columns needs,
/// padding included.
pub(super) fn buffer_bytes(k: usize, rows: usize) -> usize {
    rows * (pad(k, MATRIX_ROW_PADDING) / BLOCK_VALUES) * BLOCK_BYTES
}

type QuantizeFn = unsafe extern "C" fn(
    x_f32: *const c_void,
    ids: *const i32,
    vy: *mut c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut c_void,
);

/// The `launch_mmq_quantize_q8_1_*` symbol `dtype`'s `mmq_gguf` kernel reads;
/// see [`super::mmq_wmma::WMMA_DTYPES`] for the layout each dtype uses and why.
fn symbol(dtype: GgmlDType) -> Option<&'static [u8]> {
    super::mmq_wmma::WMMA_DTYPES
        .iter()
        .find(|d| d.dtype == dtype)
        .map(|d| d.quantize_symbol)
}

/// Quantize `rows` rows of `k` f32 columns, read from `src` starting at
/// `src_offset`, into `dst` as `block_q8_1_mmq` with the scale layout
/// `dtype`'s `mmq_gguf` kernel expects.
///
/// ```text
/// launch_mmq_quantize_q8_1_DS4(const void *x_f32, const int32_t *ids, void *vy,
///     int type_x, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
///     int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, void *stream)
/// ```
///
/// `ne00`/`ne0` are the unpadded/padded row width; the kernel zero-fills the
/// padding itself (`i0 < ne00 ? real value : 0`), so `dst` must be sized for
/// the padded width ([`buffer_bytes`]). `s01` is the row stride of `src` in
/// f32 elements, `ne1` the row count. `s02`/`s03`/`ne2`/`ne3` are the
/// channel/sample batching this call does not use, and `ids` is the MoE
/// token-gather index, also unused here.
pub(super) fn quantize(
    dtype: GgmlDType,
    src: &SendSyncDeviceMemory<f32>,
    src_offset: usize,
    dst: &SendSyncDeviceMemory<u8>,
    k: usize,
    rows: usize,
    dev: &RocmDevice,
) -> Result<()> {
    let symbol = match symbol(dtype) {
        Some(symbol) => symbol,
        None => crate::bail!("mmq_gguf: no q8_1_mmq quantize layout for {dtype:?}"),
    };
    let lib = dev.mmq_gguf_library(MmqGgufUnit::Quantize)?;
    // SAFETY: `symbol` names an `extern "C"` launcher in the compiled
    // `mmq_gguf` quantize library whose signature is exactly `QuantizeFn`;
    // see `candle-kernels/src/mmq_gguf/mmq_quantize.cu`.
    let launch: libloading::Symbol<QuantizeFn> = unsafe { lib.get(symbol) }.w()?;

    // SAFETY: `src`/`dst` are live device allocations sized for `k`/`rows`,
    // and `stream` is this device's own HIP stream.
    unsafe {
        launch(
            src.ptr_at(src_offset).cast_const(),
            std::ptr::null(),
            dst.as_ptr(),
            0,
            k as i64,
            k as i64,
            0,
            0,
            pad(k, MATRIX_ROW_PADDING) as i64,
            rows as i64,
            1,
            1,
            dev.stream().as_raw().cast(),
        );
    }
    let hip_err = crate::rocm_backend::rocm_rs::hip::get_last_error();
    if !hip_err.is_success() {
        return Err(hip_err).w();
    }
    Ok(())
}
