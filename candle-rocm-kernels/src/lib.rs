//! ROCm/HIP kernels for candle.
//!
//! The kernel sources are *shared verbatim* with the CUDA backend: this crate
//! embeds `candle-kernels/src/*.cu` and compiles them with `hipcc`. Nothing in
//! those sources is ROCm-specific; the CUDA-to-HIP bridging lives entirely in
//! `src/hip_shim`, which shadows the CUDA header names the sources include and
//! supplies the few intrinsics HIP lacks.
//!
//! Compilation happens at runtime on first use and is cached on disk under
//! `~/.cache/candle-rocm/<arch>-<rocm-version>/` (override with
//! `CANDLE_ROCM_CACHE_DIR`), so one binary runs on any GPU architecture without
//! a rebuild.
//!
//! That machinery is open to downstream crates: [`KernelCache::custom_function`]
//! compiles and launches a source this crate does not own, in the same dialect
//! and with the same disk cache. `RocmDevice::get_or_load_custom_func` is the
//! candle-level entry point.

mod compile;
mod error;
mod wrappers;

pub use compile::{KernelCache, MmqGgufLibrary, MmqGgufUnit, MmqTiles};
pub use error::KernelError;
pub use wrappers::SendSyncModule;

macro_rules! shared_source {
    ($file:literal) => {
        include_str!(concat!("../../candle-kernels/src/", $file))
    };
}

macro_rules! shim_source {
    ($file:literal) => {
        include_str!(concat!("hip_shim/", $file))
    };
}

macro_rules! mmq_gguf_source {
    ($file:literal) => {
        include_str!(concat!("../../candle-kernels/src/mmq_gguf/", $file))
    };
}

/// Identifies one shared kernel translation unit.
///
/// Mirrors `candle_kernels::Id` so the two backends index modules the same way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

/// A compilable translation unit: a stable index, a name used for cache
/// filenames, and the source itself.
pub struct Module {
    index: usize,
    name: &'static str,
    source: &'static str,
}

impl Module {
    /// Position in [`ALL_IDS`], i.e. the module's own [`Id`] as a `usize`.
    ///
    /// A dense index rather than the name, so a per-device store can be an
    /// array instead of a map. [`KernelCache`] itself keys on the name — it has
    /// to, since a custom module has no slot here — and this is for callers
    /// that want the cheaper lookup.
    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub const fn source(&self) -> &'static str {
        self.source
    }
}

macro_rules! mdl {
    ($const_name:ident, $id:ident, $name:literal, $file:literal) => {
        pub const $const_name: Module = Module {
            index: Id::$id as usize,
            name: $name,
            source: shared_source!($file),
        };
    };
}

mdl!(AFFINE, Affine, "affine", "affine.cu");
mdl!(BINARY, Binary, "binary", "binary.cu");
mdl!(CAST, Cast, "cast", "cast.cu");
mdl!(CONV, Conv, "conv", "conv.cu");
mdl!(FILL, Fill, "fill", "fill.cu");
mdl!(INDEXING, Indexing, "indexing", "indexing.cu");
mdl!(QUANTIZED, Quantized, "quantized", "quantized.cu");
mdl!(REDUCE, Reduce, "reduce", "reduce.cu");
mdl!(SORT, Sort, "sort", "sort.cu");
mdl!(TERNARY, Ternary, "ternary", "ternary.cu");
mdl!(UNARY, Unary, "unary", "unary.cu");

/// Headers the shared sources `#include`, staged alongside them before
/// compilation. The `hip_shim/` entries deliberately carry CUDA header names so
/// they shadow the real ones; the shim directory is placed first on the include
/// path.
pub(crate) const HEADERS: &[(&str, &str)] = &[
    ("compatibility.cuh", shared_source!("compatibility.cuh")),
    ("cuda_utils.cuh", shared_source!("cuda_utils.cuh")),
    (
        "binary_op_macros.cuh",
        shared_source!("binary_op_macros.cuh"),
    ),
    ("hip_shim/hip_compat.h", shim_source!("hip_compat.h")),
    ("hip_shim/cuda_fp16.h", shim_source!("cuda_fp16.h")),
    ("hip_shim/cuda_bf16.h", shim_source!("cuda_bf16.h")),
    ("hip_shim/cuda_fp8.h", shim_source!("cuda_fp8.h")),
    ("hip_shim/cuda.h", shim_source!("cuda.h")),
    ("hip_shim/cuda/std/limits", shim_source!("cuda/std/limits")),
];

/// Headers the `mmq_gguf` AMD WMMA kernels `#include`, staged into their own
/// directory before compilation.
///
/// These are compiled separately from [`HEADERS`]/[`ALL_IDS`]: the
/// `mmq_gguf` sources are built as shared libraries
/// ([`compile::KernelCache::mmq_gguf_library`]), not `--genco` device-only
/// code objects, because their `extern "C"` launcher functions use `<<<>>>`
/// kernel-launch syntax that `--genco` strips along with the rest of the
/// host code.
pub(crate) const MMQ_GGUF_HEADERS: &[(&str, &str)] = &[
    ("mmq_common.cuh", mmq_gguf_source!("mmq_common.cuh")),
    ("mmq_gguf.cuh", mmq_gguf_source!("mmq_gguf.cuh")),
    ("mmq_mma.cuh", mmq_gguf_source!("mmq_mma.cuh")),
    ("mmq_vecdotq.cuh", mmq_gguf_source!("mmq_vecdotq.cuh")),
];

/// `mmq_quantize.cu`'s source: the `f32 -> block_q8_1_mmq` activation
/// requantization the `mmq_gguf` kernels read from. Compiled once, shared by
/// every [`MmqGgufUnit`].
pub(crate) const MMQ_GGUF_QUANTIZE_SRC: &str = mmq_gguf_source!("mmq_quantize.cu");

/// `mmq_instance_q4_0.cu`'s source.
pub(crate) const MMQ_GGUF_Q4_0_SRC: &str = mmq_gguf_source!("mmq_instance_q4_0.cu");

/// `mmq_instance_q4_1.cu`'s source.
pub(crate) const MMQ_GGUF_Q4_1_SRC: &str = mmq_gguf_source!("mmq_instance_q4_1.cu");

/// `mmq_instance_q5_0.cu`'s source.
pub(crate) const MMQ_GGUF_Q5_0_SRC: &str = mmq_gguf_source!("mmq_instance_q5_0.cu");

/// `mmq_instance_q5_1.cu`'s source.
pub(crate) const MMQ_GGUF_Q5_1_SRC: &str = mmq_gguf_source!("mmq_instance_q5_1.cu");

/// `mmq_instance_q8_0.cu`'s source.
pub(crate) const MMQ_GGUF_Q8_0_SRC: &str = mmq_gguf_source!("mmq_instance_q8_0.cu");

/// `mmq_instance_q2_k.cu`'s source.
pub(crate) const MMQ_GGUF_Q2_K_SRC: &str = mmq_gguf_source!("mmq_instance_q2_k.cu");

/// `mmq_instance_q3_k.cu`'s source.
pub(crate) const MMQ_GGUF_Q3_K_SRC: &str = mmq_gguf_source!("mmq_instance_q3_k.cu");

/// `mmq_instance_q4_k.cu`'s source.
pub(crate) const MMQ_GGUF_Q4_K_SRC: &str = mmq_gguf_source!("mmq_instance_q4_k.cu");

/// `mmq_instance_q5_k.cu`'s source.
pub(crate) const MMQ_GGUF_Q5_K_SRC: &str = mmq_gguf_source!("mmq_instance_q5_k.cu");

/// `mmq_instance_q6_k.cu`'s source.
pub(crate) const MMQ_GGUF_Q6_K_SRC: &str = mmq_gguf_source!("mmq_instance_q6_k.cu");
