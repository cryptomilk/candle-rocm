//! Hardware correctness tests for the AMD WMMA MMQ path, all wired-up dtypes.
//!
//! Mirrors `tests_mmq.rs`'s shape: WMMA requantizes the activations to
//! `block_q8_1_mmq`, a numerically *different* computation from the dense
//! dequantize, so every test compares it against that same
//! dequantize-then-matmul reference rather than against the dp4a `mmq`
//! path. [`check`] launches the kernel directly rather than going through
//! `QRocmStorage::fwd`, so the shapes it covers are free of the dispatch
//! policy. These tests only run anything on an RDNA3/RDNA4 device; elsewhere
//! (including a ROCm device with no WMMA) they skip silently, same as a
//! missing ROCm device does.

use super::mmq_wmma;
use crate::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{Device, Module, Result, Tensor};

macro_rules! rocm_wmma_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) if dev.has_wmma() => Device::Rocm(dev),
            // No ROCm device, or one without WMMA (RDNA2 and older).
            _ => return Ok(()),
        }
    };
}

/// Every dtype WMMA is wired up for; see `mmq_wmma::WMMA_DTYPES`.
fn wmma_dtypes() -> impl Iterator<Item = GgmlDType> {
    mmq_wmma::WMMA_DTYPES.iter().map(|d| d.dtype)
}

/// Same tolerance `tests_mmq.rs` holds the dp4a path to, for the same reason:
/// the `q8_1`-family activations carry seven bits of mantissa and the
/// residual accumulates over `k` terms.
const TOL: f32 = 2e-2;

fn assert_close(got: &[f32], want: &[f32], row: usize, what: &str) {
    let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
    for (col, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= TOL * scale,
            "{what} at ({row}, {col}): wmma {a} vs dequantized {b} (row scale {scale})"
        );
    }
}

/// Launch the WMMA path for `(m, k) x (n, k)^T` and assert it matches the
/// same product computed from the dequantized weights.
fn check(device: &Device, dtype: GgmlDType, m: usize, k: usize, n: usize) -> Result<()> {
    let dev = match device {
        Device::Rocm(dev) => dev.clone(),
        _ => return Ok(()),
    };
    let unit = mmq_wmma::unit(dtype).unwrap_or_else(|| panic!("no mmq_gguf unit for {dtype:?}"));
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    // `QTensor` stores the weights transposed, i.e. `(n, k)`.
    let rhs = Tensor::from_slice(&rhs, (n, k), device)?;
    let qt = QTensor::quantize(&rhs, dtype)?;

    let want = Tensor::from_slice(&lhs, (m, k), device)?
        .matmul(&qt.dequantize(device)?.t()?)?
        .to_vec2::<f32>()?;

    let q = match &qt.storage {
        QStorage::Rocm(q) => q,
        _ => return Ok(()),
    };
    let y = dev.clone_htod(&lhs)?;
    let dst = mmq_wmma::mul_mat_via_q8_1_mmq(&q.data, q.len, &y, 0, dtype, unit, k, n, m, &dev)?;
    let got = dev.clone_dtoh(&dst)?;

    assert_eq!(got.len(), m * n, "{dtype:?} m={m} k={k} n={n}");
    for row in 0..m {
        assert_close(
            &got[row * n..(row + 1) * n],
            &want[row],
            row,
            &format!("{dtype:?} m={m} k={k} n={n}"),
        );
    }
    Ok(())
}

/// The prefill shape, for every wired-up dtype.
#[test]
fn wmma_matches_a_dequantized_matmul() -> Result<()> {
    let device = rocm_wmma_device!();
    for dtype in wmma_dtypes() {
        for (m, n) in [(9usize, 64usize), (64, 256)] {
            check(&device, dtype, m, 256, n)?;
        }
    }
    Ok(())
}

/// Shapes that do not divide evenly by the kernel's tile width, which is
/// where a host/device tile-size mismatch (the exact bug found and fixed in
/// the ported `mmq_instance_*.cu` launchers) would show up as out-of-bounds
/// reads or a silently wrong answer rather than a clean failure.
#[test]
fn wmma_handles_partial_tiles() -> Result<()> {
    let device = rocm_wmma_device!();
    for dtype in wmma_dtypes() {
        for m in [1usize, 9, 65, 130] {
            for n in [1usize, 3, 65, 129] {
                check(&device, dtype, m, 256, n)?;
            }
        }
    }
    Ok(())
}

/// More than one step of the kernel's column loop, and a `k` past the
/// 512-wide row padding the `block_q8_1_mmq` activations carry.
#[test]
fn wmma_handles_a_long_shared_dimension() -> Result<()> {
    let device = rocm_wmma_device!();
    for dtype in wmma_dtypes() {
        for k in [1024usize, 2048, 4096] {
            check(&device, dtype, 32, k, 64)?;
        }
    }
    Ok(())
}

/// End to end through `QRocmStorage::fwd`, at a shape large enough that
/// `mmq`'s own `is_faster` would also take it: `dense::preferred` and the
/// WMMA-before-dp4a ordering in `rocm.rs`'s `fwd` both have to agree this
/// lands on WMMA, not dp4a or the dense fallback. f16 and bf16 activations
/// reach the kernel through a cast to f32, the other half of
/// `mmq_wmma::try_fwd`.
#[test]
fn the_dispatch_reaches_wmma() -> Result<()> {
    let device = rocm_wmma_device!();
    let (m, k, n) = (16usize, 2048usize, 2048usize);

    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;
    let qt = QTensor::quantize(&rhs, GgmlDType::Q4K)?;
    let deq = qt.dequantize(&device)?.t()?;
    let mm = QMatMul::from_qtensor(qt)?;

    for act in [crate::DType::F32, crate::DType::F16, crate::DType::BF16] {
        let lhs = Tensor::from_slice(&lhs, (m, k), &device)?.to_dtype(act)?;
        let want = lhs
            .to_dtype(crate::DType::F32)?
            .matmul(&deq)?
            .to_vec2::<f32>()?;
        let got = mm
            .forward(&lhs)?
            .to_dtype(crate::DType::F32)?
            .to_vec2::<f32>()?;
        for (row, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            assert_close(got, want, row, &format!("dispatch {act:?}"));
        }
    }
    Ok(())
}

/// The `(b, m, k)` spelling flattens to the same batch as `(b * m, k)`,
/// which is the shape a transformer actually hands the matmul at prefill.
#[test]
fn three_dimensional_prefill_activations_match() -> Result<()> {
    let device = rocm_wmma_device!();
    let (b, m, k, n) = (2usize, 24usize, 2048usize, 2048usize);
    let lhs: Vec<f32> = (0..b * m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (b, m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

    let qt = QTensor::quantize(&rhs, GgmlDType::Q6K)?;
    let want = lhs.matmul(
        &qt.dequantize(&device)?
            .t()?
            .broadcast_as((b, k, n))?
            .contiguous()?,
    )?;
    let got = QMatMul::from_qtensor(qt)?.forward(&lhs)?;
    assert_eq!(got.dims(), &[b, m, n]);

    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let want = want.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&got, &want, 0, "3-D prefill");
    Ok(())
}
