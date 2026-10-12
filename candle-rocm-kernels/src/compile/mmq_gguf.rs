//! Compiling and loading the AMD WMMA MMQ kernels in
//! `candle-kernels/src/mmq_gguf`.
//!
//! Unlike this crate's other kernels, these sources compile to a shared
//! library rather than a `--genco` device-only code object: see
//! [`hipcc::compile_shared_library`] for why. The result is opened with
//! `libloading` and its `extern "C"` launcher/quantize symbols are resolved
//! by name per call, the same "resolve fresh, cache only the loaded unit"
//! shape [`super::KernelCache::function`] already uses for
//! `hipModuleGetFunction`.

use super::{cache, detect, hipcc};
use crate::error::KernelError;
use libloading::Library;
use std::fs;
use std::path::Path;

/// One `mmq_gguf` translation unit this crate knows how to compile.
///
/// Each variant is its own shared library: the quantize kernel and the
/// per-dtype instance files were already separate `.cu` files upstream, and
/// nothing is gained by combining them into one compile.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmqGgufUnit {
    /// `mmq_quantize.cu`. `f32 -> block_q8_1_mmq` activation requantization,
    /// shared by every quant type below.
    Quantize,
    /// `mmq_instance_q4_0.cu`.
    Q4_0,
    /// `mmq_instance_q4_1.cu`.
    Q4_1,
    /// `mmq_instance_q5_0.cu`.
    Q5_0,
    /// `mmq_instance_q5_1.cu`.
    Q5_1,
    /// `mmq_instance_q8_0.cu`.
    Q8_0,
    /// `mmq_instance_q2_k.cu`.
    Q2K,
    /// `mmq_instance_q3_k.cu`.
    Q3K,
    /// `mmq_instance_q4_k.cu`.
    Q4K,
    /// `mmq_instance_q5_k.cu`.
    Q5K,
    /// `mmq_instance_q6_k.cu`.
    Q6K,
}

impl MmqGgufUnit {
    /// `(source, name)` for this unit, the only place the two are tied
    /// together: adding a unit means adding one arm here plus its source
    /// constant in `lib.rs`, instead of keeping two matches in sync.
    fn source_and_name(self) -> (&'static str, &'static str) {
        match self {
            MmqGgufUnit::Quantize => (crate::MMQ_GGUF_QUANTIZE_SRC, "mmq_gguf_quantize"),
            MmqGgufUnit::Q4_0 => (crate::MMQ_GGUF_Q4_0_SRC, "mmq_gguf_q4_0"),
            MmqGgufUnit::Q4_1 => (crate::MMQ_GGUF_Q4_1_SRC, "mmq_gguf_q4_1"),
            MmqGgufUnit::Q5_0 => (crate::MMQ_GGUF_Q5_0_SRC, "mmq_gguf_q5_0"),
            MmqGgufUnit::Q5_1 => (crate::MMQ_GGUF_Q5_1_SRC, "mmq_gguf_q5_1"),
            MmqGgufUnit::Q8_0 => (crate::MMQ_GGUF_Q8_0_SRC, "mmq_gguf_q8_0"),
            MmqGgufUnit::Q2K => (crate::MMQ_GGUF_Q2_K_SRC, "mmq_gguf_q2_k"),
            MmqGgufUnit::Q3K => (crate::MMQ_GGUF_Q3_K_SRC, "mmq_gguf_q3_k"),
            MmqGgufUnit::Q4K => (crate::MMQ_GGUF_Q4_K_SRC, "mmq_gguf_q4_k"),
            MmqGgufUnit::Q5K => (crate::MMQ_GGUF_Q5_K_SRC, "mmq_gguf_q5_k"),
            MmqGgufUnit::Q6K => (crate::MMQ_GGUF_Q6_K_SRC, "mmq_gguf_q6_k"),
        }
    }

    fn source(self) -> &'static str {
        self.source_and_name().0
    }

    pub(crate) fn name(self) -> &'static str {
        self.source_and_name().1
    }
}

/// A loaded `mmq_gguf` shared library.
///
/// The field is private and [`compile_and_load`] is the only constructor,
/// the same shape [`crate::wrappers::SendSyncModule`] uses for the same
/// reason: the `unsafe impl`s below assert a property of *this* wrapper, not
/// of `libloading::Library` in general.
pub struct MmqGgufLibrary(Library);

// SAFETY: this wrapper's only state is a `dlopen` handle, process-wide and
// not tied to the thread that opened it. `dlsym` is safe to call from any
// thread against a handle another thread opened, and `Library::get`'s
// returned `Symbol` borrows `&self` rather than carrying interior
// mutability, so concurrent lookups through a shared `&MmqGgufLibrary`
// cannot race.
unsafe impl Send for MmqGgufLibrary {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for MmqGgufLibrary {}

impl MmqGgufLibrary {
    /// Resolve `symbol` (NUL-terminated, e.g. `b"launch_mmq_gguf_q4_k\0"`) to
    /// a typed function pointer.
    ///
    /// Not cached. Like [`super::KernelCache::function`], a `dlsym` lookup is
    /// cheap next to the kernel launch it is about to drive, and caching the
    /// resolved pointer would need the same unsafe lifetime extension
    /// `KernelCache` deliberately avoids there.
    ///
    /// # Safety
    /// `T` must match the C function's real signature exactly: argument
    /// types, argument count and calling convention. A mismatch is undefined
    /// behavior that will not be caught at the call site.
    pub unsafe fn get<T>(&self, symbol: &[u8]) -> Result<libloading::Symbol<'_, T>, KernelError> {
        self.0.get(symbol).map_err(|e| {
            KernelError::Compilation(format!(
                "symbol `{}` not found in mmq_gguf library: {e}",
                String::from_utf8_lossy(symbol)
            ))
        })
    }
}

/// Headers staged for an `mmq_gguf` compile: the `hip_shim` entries every
/// kernel needs (`mmq_common.cuh` itself `#include`s `cuda_fp16.h`/
/// `cuda_bf16.h`, which the shim shadows) plus `mmq_gguf`'s own four headers.
fn staged_headers() -> Vec<(&'static str, &'static str)> {
    crate::HEADERS
        .iter()
        .filter(|(name, _)| name.starts_with("hip_shim/"))
        .copied()
        .chain(crate::MMQ_GGUF_HEADERS.iter().copied())
        .collect()
}

/// The hipcc flags an `mmq_gguf` compile needs beyond `-O3 -std=c++17`.
///
/// `GGML_USE_HIP` goes on unconditionally: every `defined(GGML_USE_HIP)`
/// branch in these sources is how they tell "compiling for AMD" apart from
/// "compiling for NVIDIA", matching upstream llama.cpp. `__CUDA_ARCH__` is set
/// to the same value as [`cache::COMPILE_FLAGS`], for the same reason: several
/// `__CUDA_ARCH__`-gated branches in `mmq_common.cuh` (`ggml_cuda_dp4a`,
/// `warp_reduce_sum`/`warp_reduce_max`, `ggml_cuda_get_max_cpy_bytes`) are not
/// behind a `GGML_USE_HIP` guard and fall back to a slower path if left
/// undefined; the intrinsics they gate towards (`__dp4a`, `__reduce_add_sync`,
/// `__reduce_max_sync`) are all shimmed in `hip_compat.h`. The `RDNA*`
/// architecture define follows [`detect::rdna_define`], the same one
/// `quantized.cu` uses. `AMD_WMMA_AVAILABLE` is added only for the
/// architectures [`detect::has_wmma`] reports, which gates the actual
/// matrix-core device code in `mmq_mma.cuh` and the host-side tile sizing in
/// `mmq_common.cuh`'s `amd_wmma_available`.
fn compile_flags(arch: &str) -> Vec<String> {
    let mut flags = vec![
        "-DGGML_USE_HIP".to_string(),
        "-D__CUDA_ARCH__=890".to_string(),
    ];
    if let Some(define) = detect::rdna_define(arch) {
        flags.push(format!("-D{define}"));
    }
    if detect::has_wmma(arch) {
        flags.push("-DAMD_WMMA_AVAILABLE".to_string());
    }
    flags
}

/// Compile (or read from the disk cache) `unit`'s shared library for `arch`,
/// then `dlopen` it.
pub(crate) fn compile_and_load(
    cache_dir: &Path,
    arch: &str,
    toolchain: &str,
    unit: MmqGgufUnit,
) -> Result<MmqGgufLibrary, KernelError> {
    let headers = staged_headers();
    let src_dir = cache_dir.join(format!("mmq-gguf-src-{}", cache::headers_key(&headers)));
    hipcc::stage_headers(&src_dir, &headers)?;

    let flags = compile_flags(arch);
    let flag_refs: Vec<&str> = flags.iter().map(String::as_str).collect();

    let key = cache::mmq_gguf_key(unit.source(), &flag_refs, &headers, arch, toolchain);
    let entry = format!("{}_{key}", unit.name());
    let so_path = cache_dir.join(format!("{entry}.so"));
    let forced = cache::force_recompile();

    if forced || fs::metadata(&so_path).is_err() {
        let _lock = cache::lock_entry(cache_dir, &entry)?;
        if forced || fs::metadata(&so_path).is_err() {
            let tmp_so = cache_dir.join(format!("{entry}.{}.so.tmp", std::process::id()));
            hipcc::compile_shared_library(
                &src_dir,
                &tmp_so,
                arch,
                unit.name(),
                &entry,
                unit.source(),
                &flag_refs,
            )?;
            fs::rename(&tmp_so, &so_path).map_err(|e| {
                let _ = fs::remove_file(&tmp_so);
                KernelError::Io(format!("could not place {}: {e}", so_path.display()))
            })?;
        }
    }

    // SAFETY: `so_path` is a shared library this process just built (or
    // previously built and verified it could build) from candle-rocm's own
    // vendored, trusted source, not an arbitrary path. `dlopen` runs the
    // library's static initializers, which is the inherent risk any
    // `Library::new` call carries; there is nothing library-specific beyond
    // that here.
    unsafe { Library::new(&so_path) }
        .map(MmqGgufLibrary)
        .map_err(|e| {
            KernelError::Compilation(format!("could not dlopen {}: {e}", so_path.display()))
        })
}
