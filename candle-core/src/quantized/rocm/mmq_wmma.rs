//! The AMD WMMA MMQ dispatch for [`super::QRocmStorage::fwd`], covering every
//! GGML quant type that has an `mmq_instance_*.cu`.
//!
//! Mirrors [`super::mmq`]'s shape (requantize the activations, then a tiled
//! integer matmul against the packed weights) but against the `mmq_gguf`
//! kernels compiled as a shared library (see
//! [`candle_rocm_kernels::MmqGgufUnit`]): those run AMD's WMMA matrix-core
//! instructions instead of `__dp4a`, and read activations as
//! [`super::q8_1_mmq`]'s `block_q8_1_mmq` rather than plain `block_q8_1`.
//!
//! Only RDNA3/RDNA4 devices take this path at all (see
//! [`crate::rocm_backend::RocmDevice::has_wmma`]); everything else falls
//! through to [`super::mmq`]'s `__dp4a` kernel unchanged.

use super::kernels::WARP_SIZE;
use super::q8_1::force_dmmv;
use super::q8_1_mmq;
use super::QRocmStorage;
use crate::backend::BackendStorage;
use crate::quantized::GgmlDType;
use crate::rocm_backend::{
    RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory, WrapErr,
};
use crate::{DType, Layout, Result};
use candle_rocm_kernels::MmqGgufUnit;
use std::ffi::c_void;

/// One GGML quant type WMMA is wired up for: its `mmq_gguf` unit, the
/// `launch_mmq_gguf_*` matmul symbol, and the `launch_mmq_quantize_q8_1_*`
/// symbol its activations need (matching `mmq_get_q8_1_ds_layout` in
/// `mmq_gguf.cuh`: `DS4` is a `half2` scale and sum per 32 values, `D4` is a
/// float scale per 32 values with no sum, and `D2S6` (Q2_K only) is a float
/// scale per 64 values plus a 6-bit sum per 16 values).
///
/// The single source of truth for which dtypes WMMA covers: [`unit`],
/// [`symbol`], [`super::q8_1_mmq::symbol`], the benchmark in
/// [`super::bench::bench_wmma_vs_dp4a`], and the hardware tests in
/// `tests_mmq_wmma.rs` all derive from this table, so wiring up a new dtype
/// only needs one new row here (plus its `MmqGgufUnit` variant and source
/// constant in `candle-rocm-kernels`).
pub(super) struct WmmaDtype {
    pub(super) dtype: GgmlDType,
    pub(super) unit: MmqGgufUnit,
    pub(super) gguf_symbol: &'static [u8],
    pub(super) quantize_symbol: &'static [u8],
}

pub(super) const WMMA_DTYPES: &[WmmaDtype] = &[
    WmmaDtype {
        dtype: GgmlDType::Q4_0,
        unit: MmqGgufUnit::Q4_0,
        gguf_symbol: b"launch_mmq_gguf_q4_0\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_DS4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q4_1,
        unit: MmqGgufUnit::Q4_1,
        gguf_symbol: b"launch_mmq_gguf_q4_1\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_DS4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q5_0,
        unit: MmqGgufUnit::Q5_0,
        gguf_symbol: b"launch_mmq_gguf_q5_0\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_D4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q5_1,
        unit: MmqGgufUnit::Q5_1,
        gguf_symbol: b"launch_mmq_gguf_q5_1\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_DS4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q8_0,
        unit: MmqGgufUnit::Q8_0,
        gguf_symbol: b"launch_mmq_gguf_q8_0\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_D4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q2K,
        unit: MmqGgufUnit::Q2K,
        gguf_symbol: b"launch_mmq_gguf_q2_k\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_D2S6\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q3K,
        unit: MmqGgufUnit::Q3K,
        gguf_symbol: b"launch_mmq_gguf_q3_k\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_D4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q4K,
        unit: MmqGgufUnit::Q4K,
        gguf_symbol: b"launch_mmq_gguf_q4_k\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_DS4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q5K,
        unit: MmqGgufUnit::Q5K,
        gguf_symbol: b"launch_mmq_gguf_q5_k\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_DS4\0",
    },
    WmmaDtype {
        dtype: GgmlDType::Q6K,
        unit: MmqGgufUnit::Q6K,
        gguf_symbol: b"launch_mmq_gguf_q6_k\0",
        quantize_symbol: b"launch_mmq_quantize_q8_1_D4\0",
    },
];

/// The `mmq_gguf` unit for `dtype`, or `None` if WMMA has not been wired up
/// for it yet.
pub(super) fn unit(dtype: GgmlDType) -> Option<MmqGgufUnit> {
    WMMA_DTYPES
        .iter()
        .find(|d| d.dtype == dtype)
        .map(|d| d.unit)
}

/// The `launch_mmq_gguf_*` symbol for `unit`.
fn symbol(unit: MmqGgufUnit) -> &'static [u8] {
    WMMA_DTYPES
        .iter()
        .find(|d| d.unit == unit)
        .map(|d| d.gguf_symbol)
        .unwrap_or_else(|| unreachable!("Quantize is q8_1_mmq's unit, not a matmul one"))
}

type LaunchFn = unsafe extern "C" fn(
    tmp_fixup_ptr: *mut c_void,
    x: *const c_void,
    y_q8_1_mmq: *const c_void,
    dst: *mut c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size_host: i32,
    stream: *mut c_void,
);

/// `data` (the `(nrows, ncols)` packed weights) times `b_size` activation
/// rows, the WMMA counterpart of [`super::mmq::mul_mat_via_q8_1`].
///
/// ```text
/// launch_mmq_gguf_q4_k(void *tmp_fixup_ptr,
///     const void *x, const void *y_q8_1_mmq, void *dst,
///     int64_t ncols_x, int64_t nrows_x, int64_t ncols_y,
///     int64_t stride_row_x, int64_t stride_col_dst,
///     int cc, int nsm, int64_t smpbo, int warp_size_host, void *stream)
/// ```
///
/// `tmp_fixup_ptr` is only read on the stream-k path, which the generated
/// `use_stream_k` is always `false` for an AMD `cc`, so a null pointer is
/// safe. `stride_row_x` is the weight's row stride in *blocks* of `dtype`
/// (`GGML_CUDA_CC`'s own convention, matching how `nb[1]`/`ggml_type_size`
/// combine upstream), which for the contiguous weight storage this backend
/// always uses is `ncols / dtype.block_size()`. `stride_col_dst` is accepted
/// by the signature but unused by this non-MoE launcher body; `nrows` is
/// passed for it regardless, matching the convention a future MoE-aware
/// variant would need. The output lands as `dst[col*nrows + row]`, i.e. the
/// contiguous `(b_size, nrows)` the caller wants, the same write-back layout
/// [`super::mmq::mul_mat_via_q8_1`] already relies on.
#[allow(clippy::too_many_arguments)]
pub(super) fn mul_mat_via_q8_1_mmq(
    data: &SendSyncDeviceMemory<u8>,
    data_len: usize,
    y: &SendSyncDeviceMemory<f32>,
    y_offset: usize,
    dtype: GgmlDType,
    unit: MmqGgufUnit,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f32>> {
    let data_elems = data_len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("quantized mmq (wmma): data holds {data_elems} elems, need {ncols}x{nrows}")
    }

    let y_q8_1_mmq = dev.alloc_zeros::<u8>(q8_1_mmq::buffer_bytes(ncols, b_size))?;
    q8_1_mmq::quantize(dtype, y, y_offset, &y_q8_1_mmq, ncols, b_size, dev)?;

    let lib = dev.mmq_gguf_library(unit)?;
    // SAFETY: the symbol names an `extern "C"` launcher in the compiled
    // `mmq_gguf` library whose signature is exactly `LaunchFn`; see
    // `candle-kernels/src/mmq_gguf/mmq_instance_q4_k.cu`/`_q6_k.cu`.
    let launch: libloading::Symbol<LaunchFn> = unsafe { lib.get(symbol(unit)) }.w()?;

    let cc = match dev.amd_cc() {
        Some(cc) => cc,
        None => crate::bail!(
            "mmq_gguf: could not derive an AMD cc value for this device's architecture"
        ),
    };
    let stride_row_x = (ncols / dtype.block_size()) as i64;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;

    // SAFETY: every pointer is a live device allocation sized for the shapes
    // passed alongside it, and `stream` is this device's own HIP stream.
    unsafe {
        launch(
            std::ptr::null_mut(),
            data.as_ptr().cast_const(),
            y_q8_1_mmq.as_ptr().cast_const(),
            dst.as_ptr(),
            ncols as i64,
            nrows as i64,
            b_size as i64,
            stride_row_x,
            nrows as i64,
            cc,
            dev.multiprocessor_count() as i32,
            dev.shared_mem_per_block_optin(),
            WARP_SIZE as i32,
            dev.stream().as_raw().cast(),
        );
    }
    let hip_err = crate::rocm_backend::rocm_rs::hip::get_last_error();
    if !hip_err.is_success() {
        return Err(hip_err).w();
    }
    Ok(dst)
}

/// The WMMA dispatch for [`super::QRocmStorage::fwd`]: `None` when this
/// device has no WMMA, when `dtype` has no `mmq_gguf` unit wired up yet, or
/// when the activation layout [`super::mmq::try_fwd`] would also decline.
pub(super) fn try_fwd(
    q: &QRocmStorage,
    n: usize,
    k: usize,
    b_size: usize,
    storage: &RocmStorage,
    layout: &Layout,
) -> Result<Option<SendSyncDeviceMemory<f32>>> {
    if force_dmmv() || !q.device.has_wmma() || b_size == 0 {
        return Ok(None);
    }
    let Some(unit) = unit(q.dtype) else {
        return Ok(None);
    };
    let (o1, o2) = match layout.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };
    if o2 - o1 != b_size * k {
        return Ok(None);
    }

    let cast;
    let (y, y_offset) = match &storage.slice {
        RocmStorageSlice::F32(y) => (y, o1),
        RocmStorageSlice::F16(_) | RocmStorageSlice::BF16(_) => {
            cast = storage.to_dtype(layout, DType::F32)?;
            match &cast.slice {
                RocmStorageSlice::F32(y) => (y, 0usize),
                _ => return Ok(None),
            }
        }
        _ => return Ok(None),
    };

    let dst = mul_mat_via_q8_1_mmq(
        &q.data, q.len, y, y_offset, q.dtype, unit, k, n, b_size, &q.device,
    )?;
    Ok(Some(dst))
}
