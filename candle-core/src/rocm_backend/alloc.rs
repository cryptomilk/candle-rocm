//! Device allocation for the ROCm backend.
//!
//! `hipFree` blocks until *all* pending device work completes. Every dropped
//! temporary — the dims/strides buffer at the end of a launcher, the output of
//! every intermediate tensor — therefore used to drain the pipeline, so a chain
//! of twenty pointwise ops paid twenty full device synchronisations and the
//! backend had essentially no CPU/GPU overlap.
//!
//! [`RocmAllocator`] fixes that by not handing buffers back to the driver on
//! the hot path: freed blocks go on a per-size free list and are reused by the
//! next allocation of that size. Freeing becomes a host-side push and allocating
//! becomes a host-side pop. The cache is bounded — parked bytes above
//! [`default_cache_limit`] are evicted back to the driver, largest block first —
//! so a workload whose shapes drift every step (autoregressive decode above
//! all) cannot hoard the card's VRAM in blocks nothing will request again.
//!
//! # Why not `hipMallocAsync`
//!
//! HIP's own stream-ordered allocator is the textbook answer and was tried
//! first. On ROCm 7.2.4 it is marked Beta and behaves like it: it is not
//! thread-safe (two threads allocating and freeing against one pool hand back
//! overlapping buffers), a process that creates and destroys ~100 `RocmDevice`s
//! faults partway through when each gets its own `hipMemPoolCreate`, and using
//! the device's default pool instead faults sooner still under a mixed
//! pointwise/conv workload. Every one of those was reproduced on gfx1101; the
//! reuse below is ordinary `hipMalloc` memory with no driver bookkeeping to get
//! wrong.
//!
//! # The ordering invariant
//!
//! Reusing a block for a new tensor is safe because **every** device operation
//! this backend issues that touches allocator memory is queued on the owning
//! `RocmDevice`'s single stream: kernel launches, the D2D copies in
//! `copy_strided_src`/`copy2d`, `alloc_zeros`' memset, the host transfers in
//! `clone_htod`/`clone_dtoh`, rocBLAS, MIOpen and rocRAND. Work queued for the
//! new owner therefore runs strictly after the work of the old one. Anything
//! added later that issues on another stream — or on the null stream — must
//! either take the same stream or synchronise before the buffer can be recycled.
//!
//! # Address-stable allocations
//!
//! [`VmmPool`] is a separate allocator for a different requirement: HIP
//! graph replay bakes device addresses into recorded kernel launches, so
//! anything a captured graph touches must get the same address on every
//! replay. [`RocmAllocator`]'s free list cannot promise that. `VmmPool`
//! reserves one virtual address range and bump-allocates within it instead,
//! so a repeated allocation sequence against a pool reset to the same
//! watermark always lands at the same addresses.
//!
//! [`GraphCaptureSession`] is the switch between the two: while one is
//! alive, [`SendSyncDeviceMemory::new`] routes through the supplied
//! `VmmPool` instead of the bucketed free list, with no change at any of
//! its call sites. Freeing a `VmmPool`-backed allocation is a no-op rather
//! than a recycle, since the pool's own watermark reset is what reclaims
//! that space.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use rocm_rs::hip::bindings;
use rocm_rs::hip::error::Error as HipError;
use rocm_rs::hip::Stream;

use super::hip_graph::{stream_is_capturing, HipGraph, HipGraphCapture};
use super::wrappers::SendSyncStream;

fn hip_check(status: bindings::hipError_t) -> Result<(), HipError> {
    if status == bindings::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(HipError::new(status))
    }
}

/// Bounds on the allocation granularity.
///
/// Rounding up means two tensors of *similar* size share a bucket instead of
/// each forcing a fresh `hipMalloc`. The granularity is *relative* — an eighth
/// of the request's power-of-two floor — because the workloads that defeat a
/// fixed granularity are the shape-drifting ones: autoregressive decode grows
/// its attention buffers by a few KB every step, and with a fixed step a
/// slightly-larger request lands in a fresh bucket while the previous step's
/// block is parked forever. An eighth keeps the overshoot at or below 12.5%
/// and puts every request between consecutive powers of two into one of eight
/// buckets, so a growing buffer reuses its block for many steps before
/// stepping up.
///
/// 512 B is the floor — the driver's own alignment, so small buffers waste
/// nothing that was not already padding. 32 MiB is the ceiling, so a
/// multi-GiB weight tensor overshoots by at most 32 MiB rather than by 12.5%.
const SMALL_GRANULARITY: usize = 512;
const MAX_GRANULARITY: usize = 32 << 20;

fn bucket_size(size: usize) -> usize {
    // `size` is never 0 here, but `max(1)` keeps `ilog2` total anyway.
    let granularity =
        ((1usize << size.max(1).ilog2()) / 8).clamp(SMALL_GRANULARITY, MAX_GRANULARITY);
    size.div_ceil(granularity) * granularity
}

/// A device pointer parked on the free list.
///
/// The raw pointer is not `Send`, but the memory it addresses belongs to the
/// GPU and is reachable from any host thread; the wrapper exists only to say so
/// once rather than at every use.
#[derive(Clone, Copy)]
struct Block(*mut std::ffi::c_void);

// SAFETY: a device address is process-wide and carries no thread affinity. The
// free list that holds these is behind a `Mutex`, so no two threads can observe
// the same block as available.
unsafe impl Send for Block {}

/// The free list plus the running total of the bytes parked on it, kept
/// together under one lock so the total can never drift from the map.
#[derive(Default)]
struct FreeLists {
    map: HashMap<usize, Vec<Block>>,
    cached_bytes: usize,
    /// Set by [`GraphCaptureSession::new`]; while `Some`, `alloc_bytes`
    /// routes through this pool instead of `map`, giving a HIP graph
    /// capture the address-stable allocations it needs.
    capture_pool: Option<Arc<VmmPool>>,
}

/// Caching device allocator for one [`super::RocmDevice`].
///
/// Holds an `Arc` of the device's stream so that the ordering invariant
/// documented at the top of this module is expressible: the allocator, and
/// every buffer it hands out, keeps that stream alive.
pub struct RocmAllocator {
    stream: Arc<SendSyncStream>,
    free: Mutex<FreeLists>,
    /// Ceiling on [`FreeLists::cached_bytes`]; parking a block above it evicts
    /// parked blocks back to the driver, largest first. `usize::MAX` disables
    /// the cap. Atomic only so tests can tighten it through the `Arc`.
    cache_limit: AtomicUsize,
}

// SAFETY: the state is a stream handle — a process-wide driver object with no
// thread affinity — and a `Mutex`-guarded map of device addresses.
unsafe impl Send for RocmAllocator {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for RocmAllocator {}

impl RocmAllocator {
    pub(crate) fn new(stream: Arc<SendSyncStream>) -> Self {
        Self {
            stream,
            free: Mutex::new(FreeLists::default()),
            cache_limit: AtomicUsize::new(default_cache_limit()),
        }
    }

    pub(crate) fn raw_stream(&self) -> bindings::hipStream_t {
        self.stream.0.as_raw()
    }

    /// Rejects a blocking host/device copy issued while this allocator's
    /// stream is mid-capture, instead of letting it fail deep inside the
    /// driver with an unrelated-looking error.
    fn check_not_capturing(&self) -> Result<(), HipError> {
        if stream_is_capturing(&self.stream.0)? {
            return Err(HipError::new(
                bindings::hipError_t_hipErrorStreamCaptureUnsupported,
            ));
        }
        Ok(())
    }

    /// Poisoning is ignored: the map is a pure cache, so the worst a panic
    /// mid-update can leave behind is a block that is never reused. Refusing to
    /// allocate afterwards would be strictly worse.
    fn lock_free(&self) -> MutexGuard<'_, FreeLists> {
        self.free.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A block of at least `size` bytes, from the free list if one is
    /// parked, or from the capture-mode `VmmPool` if a
    /// [`GraphCaptureSession`] is currently alive on this allocator.
    ///
    /// A `VmmPool`-backed allocation is reported as [`AllocOrigin::Pool`],
    /// which [`Self::recycle`] matches to skip parking it on the bucketed
    /// free list.
    fn alloc_bytes(&self, size: usize) -> Result<(*mut std::ffi::c_void, AllocOrigin), HipError> {
        if size == 0 {
            return Ok((std::ptr::null_mut(), AllocOrigin::Bucketed(0)));
        }
        let bucket = {
            let mut free = self.lock_free();
            if let Some(pool) = free.capture_pool.clone() {
                drop(free);
                let slice = pool.alloc(size)?;
                let (ptr, epoch) = (slice.ptr, slice.birth_epoch);
                return Ok((ptr, AllocOrigin::Pool { pool, epoch }));
            }
            let bucket = bucket_size(size);
            if let Some(block) = free.map.get_mut(&bucket).and_then(Vec::pop) {
                free.cached_bytes -= bucket;
                return Ok((block.0, AllocOrigin::Bucketed(bucket)));
            }
            bucket
        };

        match raw_malloc(bucket) {
            Ok(ptr) => Ok((ptr, AllocOrigin::Bucketed(bucket))),
            // Out of memory only means the *cache* is holding it. Hand
            // everything back and try once more before reporting failure.
            Err(e) => {
                self.release_all();
                raw_malloc(bucket)
                    .map(|ptr| (ptr, AllocOrigin::Bucketed(bucket)))
                    .map_err(|_| e)
            }
        }
    }

    /// Park a block for reuse; on the hot path this is a host-side push.
    ///
    /// If parking the block lifts the cache past [`Self::cache_limit`], parked
    /// blocks are returned to the driver, largest bucket first, until the
    /// cache fits again. The cap is what keeps a shape-drifting workload — a
    /// decode loop whose buffers grow every step, parking a slightly-too-small
    /// block each time — from hoarding the whole card: the driver, rocRAND,
    /// rocBLAS workspaces and every other process allocate outside this free
    /// list, so `alloc_bytes`' release-and-retry cannot save *them* from an
    /// OOM this cache caused. Largest-first eviction throws out exactly the
    /// outgrown blocks while the small, hot buckets survive.
    fn recycle(&self, ptr: *mut std::ffi::c_void, origin: &AllocOrigin) {
        if ptr.is_null() {
            return;
        }
        // `AllocOrigin::Pool` is a `VmmPool`-backed allocation (see
        // `alloc_bytes`'s doc comment): freeing it is a no-op, since the
        // pool's own watermark reset is what reclaims that space. Parking it
        // here instead would let a later bucketed allocation hand it out as
        // if it were `hipMalloc` memory, and eventually `hipFree` an address
        // `hipMemAddressReserve` owns.
        let bucket = match origin {
            AllocOrigin::Pool { .. } => return,
            AllocOrigin::Bucketed(bucket) => *bucket,
        };
        let evicted = {
            let mut free = self.lock_free();
            free.map.entry(bucket).or_default().push(Block(ptr));
            free.cached_bytes += bucket;
            let mut evicted = Vec::new();
            let cache_limit = self.cache_limit.load(Ordering::Relaxed);
            // Every round removes a block or an empty bucket, so this ends.
            while free.cached_bytes > cache_limit {
                let Some(&largest) = free.map.keys().max() else {
                    break;
                };
                // `alloc_bytes` pops without pruning, so a bucket can be empty.
                match free.map.get_mut(&largest).and_then(Vec::pop) {
                    Some(block) => {
                        evicted.push(block);
                        free.cached_bytes -= largest;
                    }
                    None => {
                        free.map.remove(&largest);
                    }
                }
            }
            evicted
        };
        // `hipFree` synchronises the device, so it runs outside the lock; the
        // sync is also what makes freeing sound while queued work may still
        // reference the block.
        for block in evicted {
            // SAFETY: the block came from `hipMalloc` and is no longer
            // reachable from the map.
            unsafe {
                let _ = bindings::hipFree(block.0);
            }
        }
    }

    /// Bytes currently parked on the free list.
    #[cfg(test)]
    pub(crate) fn cached_bytes(&self) -> usize {
        self.lock_free().cached_bytes
    }

    #[cfg(test)]
    pub(crate) fn set_cache_limit(&self, bytes: usize) {
        self.cache_limit.store(bytes, Ordering::Relaxed);
    }

    /// Return every parked block to the driver.
    ///
    /// `hipFree` is documented to synchronise the device, which is exactly what
    /// is wanted here: the blocks may still be referenced by queued work, and
    /// this runs only when the device is being torn down or an allocation has
    /// already failed.
    fn release_all(&self) {
        let mut free = self.lock_free();
        free.cached_bytes = 0;
        for (_, blocks) in free.map.drain() {
            for block in blocks {
                // SAFETY: every block came from `hipMalloc` and is not
                // reachable from anywhere else once drained from the map.
                unsafe {
                    let _ = bindings::hipFree(block.0);
                }
            }
        }
    }
}

/// A single HIP graph capture attempt, tying together the HIP capture
/// session, the allocator's capture-mode routing, and the `VmmPool`
/// watermark they share.
///
/// Dropping an uncommitted session (i.e. one [`Self::end`] was never called
/// on) rolls `pool` back to the watermark recorded at [`Self::new`], so a
/// retried capture attempt gets the same addresses as the one it replaces.
/// A committed session (after [`Self::end`] succeeds) leaves the pool's
/// watermark advanced, since the addresses it allocated are baked into the
/// returned [`HipGraph`] and must stay live for replay.
///
/// # Safety contract
/// No other thread may allocate through `alloc` (e.g. via a cloned
/// [`super::RocmDevice`] sharing the same allocator) while a session is
/// alive. Such an allocation would be routed through `pool` like any
/// capture-time allocation, but is not actually part of the capture; if
/// the session is later abandoned, rolling `pool` back past that unrelated
/// allocation's address lets a later allocation alias it. This mirrors the
/// existing requirement, documented on [`HipGraphCapture`], that no
/// unrelated activity runs on the captured stream.
#[must_use = "a capture session does nothing if immediately dropped"]
pub struct GraphCaptureSession<'a> {
    capture: Option<HipGraphCapture<'a>>,
    alloc: &'a RocmAllocator,
    pool: Arc<VmmPool>,
    mark: usize,
    committed: bool,
}

impl<'a> GraphCaptureSession<'a> {
    /// Begin a HIP graph capture on `stream`, routing allocations made on
    /// `alloc` during the capture through `pool` instead of the bucketed
    /// free list.
    ///
    /// # Errors
    /// Returns an error if `pool` already backs another capture session (on
    /// this or any other allocator), if a capture is already in progress on
    /// `alloc` (nested capture is not supported), or if
    /// `hipStreamBeginCapture` itself fails.
    pub fn new(
        alloc: &'a RocmAllocator,
        stream: &'a Stream,
        pool: Arc<VmmPool>,
    ) -> Result<Self, HipError> {
        // Claims `pool` for this session before anything else runs, so two
        // sessions on two different allocators can never share one pool:
        // the `capture_pool` check below is per-allocator and would not
        // catch that case.
        if pool
            .in_use
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(HipError::new(
                bindings::hipError_t_hipErrorStreamCaptureUnsupported,
            ));
        }
        let mark = pool.watermark();
        if alloc.lock_free().capture_pool.is_some() {
            pool.in_use.store(false, Ordering::Release);
            return Err(HipError::new(
                bindings::hipError_t_hipErrorStreamCaptureUnsupported,
            ));
        }
        let capture = match HipGraphCapture::begin(stream) {
            Ok(capture) => capture,
            Err(e) => {
                pool.in_use.store(false, Ordering::Release);
                return Err(e);
            }
        };
        // Only routed through `pool` once the real HIP capture has actually
        // begun, so `alloc_bytes` never diverts to `pool` for a buffer that
        // isn't being captured.
        alloc.lock_free().capture_pool = Some(pool.clone());
        Ok(Self {
            capture: Some(capture),
            alloc,
            pool,
            mark,
            committed: false,
        })
    }

    /// End the capture and return the recorded graph, committing the
    /// `VmmPool` allocations made during it so `Drop` does not roll them
    /// back.
    ///
    /// # Pool lifetime
    /// The returned graph's instantiated `HipGraphExec`s launch kernels
    /// that reference addresses this session allocated from `pool`. The
    /// caller must keep the same `Arc<VmmPool>` passed to
    /// [`RocmDevice::begin_graph_capture`](super::RocmDevice::begin_graph_capture)
    /// alive for as long as any such `HipGraphExec` is alive — dropping the
    /// last `Arc` unmaps those addresses out from under any exec that still
    /// references them.
    ///
    /// # Errors
    /// Returns an error if the underlying `hipStreamEndCapture` fails.
    ///
    /// # Panics
    /// Never in practice: `self.capture` is only ever cleared by this
    /// method, which consumes `self`, so no other call can run first.
    pub fn end(mut self) -> Result<Option<HipGraph>, HipError> {
        let capture = self.capture.take().expect("capture already ended");
        let graph = capture.end()?;
        // Stop routing through `pool` as soon as the real capture has
        // ended, rather than waiting for `Drop`.
        self.alloc.lock_free().capture_pool = None;
        self.committed = true;
        Ok(graph)
    }
}

impl Drop for GraphCaptureSession<'_> {
    fn drop(&mut self) {
        // End the real HIP capture (if `end` was never called) before
        // flipping the allocator's routing flag off, so the window where
        // `alloc` routes through `pool` is never wider than the window
        // where the stream is actually capturing.
        drop(self.capture.take());
        self.alloc.lock_free().capture_pool = None;
        if !self.committed {
            // Abandoned without `.end()` (early return, panic, or simply
            // dropped). Roll the pool back to the watermark recorded at
            // `new` so the next attempt reuses the same addresses.
            self.pool.reset_to(self.mark);
        }
        self.pool.in_use.store(false, Ordering::Release);
    }
}

/// Where a block handed out by [`RocmAllocator::alloc_bytes`] came from,
/// and therefore how [`RocmAllocator::recycle`] must treat it when freed.
#[derive(Clone)]
enum AllocOrigin {
    /// Parked on the bucketed free list at this bucket size.
    Bucketed(usize),
    /// Bump-allocated from `pool` during a [`GraphCaptureSession`], at
    /// `epoch`. Freeing it is a no-op, since the pool's own watermark reset
    /// is what reclaims that space. `epoch` is `pool`'s
    /// [`VmmPool::reset_to`] generation at allocation time, checked by
    /// [`SendSyncDeviceMemory::as_ptr`] and [`SendSyncDeviceMemory::ptr_at`]
    /// against `pool`'s current generation, mirroring [`VmmSlice::as_ptr`]'s
    /// own check.
    Pool { pool: Arc<VmmPool>, epoch: u64 },
}

/// The cap on parked bytes: an eighth of the card's VRAM, overridable with
/// `CANDLE_ROCM_CACHE_LIMIT_MB` (`0` disables the cap).
///
/// An eighth is small enough that the cache never starves the driver or a
/// neighbouring library of a 16 GB card, and large enough that a shape-stable
/// workload — whose parked set is only the buffers currently between owners —
/// never reaches it and keeps the old always-cache behaviour.
fn default_cache_limit() -> usize {
    if let Some(mb) = std::env::var("CANDLE_ROCM_CACHE_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return if mb == 0 { usize::MAX } else { mb << 20 };
    }
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: both out-pointers are valid; the values are only read on success.
    let status = unsafe { bindings::hipMemGetInfo(&mut free, &mut total) };
    if status == bindings::hipError_t_hipSuccess && total > 0 {
        total / 8
    } else {
        1 << 30
    }
}

fn raw_malloc(size: usize) -> Result<*mut std::ffi::c_void, HipError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` is only read once the status has been checked.
    hip_check(unsafe { bindings::hipMalloc(&mut ptr, size) })?;
    Ok(ptr)
}

impl Drop for RocmAllocator {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// Total virtual address space reserved per [`VmmPool`]. Reserving address
/// space costs nothing until it is actually mapped to physical memory, so
/// this can be generous: large enough that a capture region's allocation
/// sequence never runs out across a long decode session.
const VMM_POOL_RESERVE_BYTES: usize = 1 << 35; // 32 GiB

/// Every [`VmmPool::alloc`] request is rounded up to this boundary, the same
/// alignment HIP's own allocators use internally.
const VMM_ALIGNMENT: usize = 128;

fn round_up(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// Same as [`round_up`], but returns `None` instead of silently wrapping
/// when `n` is close enough to `usize::MAX` that rounding would overflow.
/// Used wherever the input isn't already bounded by a known-safe limit.
fn checked_round_up(n: usize, to: usize) -> Option<usize> {
    n.div_ceil(to).checked_mul(to)
}

/// Growth-only virtual memory reservation backing one [`VmmPool`].
struct VmmPoolState {
    /// Base of the reserved virtual address range. Null until the first
    /// [`VmmPool::alloc`] call reserves it.
    base: *mut std::ffi::c_void,
    /// Physical bytes mapped into the reservation so far.
    mapped: usize,
    /// Bump-allocator offset; the next [`VmmPool::alloc`] starts here.
    used: usize,
    /// Every `hipMemMap` call's `(ptr, size)`, unmapped individually on
    /// `Drop`. A single bulk `hipMemUnmap` over the whole mapped range hits
    /// a ROCm runtime bug (upstream issue #285) that does not actually
    /// unmap anything past the first physical allocation.
    mappings: Vec<(*mut std::ffi::c_void, usize)>,
}

// SAFETY: the state is virtual and physical device memory addresses, which
// are process-wide and carry no thread affinity, plus a `Vec` recording
// them. The `Mutex<VmmPoolState>` in `VmmPool` is the only way to reach this
// type, so no two threads ever observe it unsynchronized.
unsafe impl Send for VmmPoolState {}

/// A bump allocator over HIP virtual memory, giving every allocation a
/// deterministic, stable device address for a given allocation sequence.
///
/// Unlike [`RocmAllocator`], which recycles same-size blocks from a free
/// list with no address guarantee, `VmmPool` reserves one virtual address
/// range up front and hands out `base + offset` on every call. Replaying
/// the same sequence of [`Self::alloc`] calls against a pool reset to the
/// same watermark always returns the same addresses, which is what HIP
/// graph replay needs: a captured graph's kernel launches bake in the
/// addresses recorded during capture.
///
/// Freeing is watermark-only: a [`VmmSlice`] going out of scope does
/// nothing, and the bump pointer only moves backward in bulk, via
/// [`Self::reset_to`]. This is deliberately looser than a strict LIFO free
/// list, because candle's eager execution does not guarantee tensors drop
/// in allocation order.
pub struct VmmPool {
    device_id: i32,
    granularity: usize,
    /// Bumped on every [`Self::reset_to`]. Lets a [`VmmSlice`] detect, in
    /// debug builds, that it was obtained before a reset that may have
    /// invalidated its address. Outside the mutex since it is read from
    /// [`VmmSlice::as_ptr`] without taking the lock.
    epoch: AtomicU64,
    /// Set while a [`GraphCaptureSession`] holds this pool. Lets `new`
    /// reject a second session on this pool even if it targets a different
    /// [`RocmAllocator`]/device, which the allocator's own `capture_pool`
    /// check cannot see.
    in_use: AtomicBool,
    state: Mutex<VmmPoolState>,
}

impl VmmPool {
    /// Construct a pool for `device_id`.
    ///
    /// Only queries the allocation granularity here. The virtual address
    /// range itself is reserved lazily, on the first [`Self::alloc`] call,
    /// so constructing a pool that never allocates costs nothing beyond
    /// this one query.
    ///
    /// # Errors
    /// Returns an error if [`vmm_usable`] reports this runtime cannot back
    /// a multi-allocation reservation (`ROCm/rocm-systems#2516`), or if the
    /// underlying granularity query fails.
    pub fn new(device_id: i32) -> Result<Self, HipError> {
        if !vmm_usable(device_id) {
            return Err(HipError::new(bindings::hipError_t_hipErrorNotSupported));
        }
        let granularity = vmm_granularity(device_id)?;
        Ok(Self {
            device_id,
            granularity,
            epoch: AtomicU64::new(0),
            in_use: AtomicBool::new(false),
            state: Mutex::new(VmmPoolState {
                base: std::ptr::null_mut(),
                mapped: 0,
                used: 0,
                mappings: Vec::new(),
            }),
        })
    }

    /// Poisoning is ignored, for the same reason as
    /// [`RocmAllocator::lock_free`]: a panic mid-update can at worst leave
    /// `mapped`/`used` inconsistent with the mappings `Vec`, and refusing to
    /// allocate afterwards would be strictly worse than risking that.
    fn lock(&self) -> MutexGuard<'_, VmmPoolState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Bump-allocate `size` bytes at a stable address.
    ///
    /// Reserves the pool's virtual address range on the first call, and
    /// maps in more physical pages whenever the bump pointer outgrows what
    /// is already mapped. Replaying the same sequence of `alloc` calls
    /// against a pool reset to the same watermark always returns the same
    /// addresses.
    ///
    /// # Errors
    /// Returns an error if the underlying HIP VMM calls fail, or if `size`
    /// would grow the pool past its reserved address space.
    pub fn alloc(&self, size: usize) -> Result<VmmSlice<'_>, HipError> {
        if size == 0 {
            return Ok(VmmSlice {
                ptr: std::ptr::null_mut(),
                pool: self,
                birth_epoch: self.epoch.load(Ordering::Relaxed),
            });
        }
        let out_of_memory = || HipError::new(bindings::hipError_t_hipErrorOutOfMemory);
        let size = checked_round_up(size, VMM_ALIGNMENT).ok_or_else(out_of_memory)?;
        let mut state = self.lock();

        if state.base.is_null() {
            state.base = reserve_address_range(VMM_POOL_RESERVE_BYTES)?;
        }

        let offset = state.used;
        let new_used = offset.checked_add(size).ok_or_else(out_of_memory)?;
        if new_used > VMM_POOL_RESERVE_BYTES {
            return Err(out_of_memory());
        }

        if new_used > state.mapped {
            let grow = round_up(new_used - state.mapped, self.granularity);
            // Clamp to what's left in the reservation: `grow` is rounded up
            // to `self.granularity`, which need not evenly divide
            // `VMM_POOL_RESERVE_BYTES`, so the rounding alone could overshoot
            // it. `new_used <= VMM_POOL_RESERVE_BYTES` was just checked
            // above, so this never clamps below what `new_used` needs.
            let grow = grow.min(VMM_POOL_RESERVE_BYTES - state.mapped);
            // SAFETY: `state.base` was reserved for `VMM_POOL_RESERVE_BYTES`
            // bytes, and `state.mapped + grow <= VMM_POOL_RESERVE_BYTES`
            // follows from the clamp above.
            let dst = unsafe { state.base.add(state.mapped) };
            map_physical_pages(self.device_id, dst, grow)?;
            state.mappings.push((dst, grow));
            state.mapped += grow;
        }

        state.used = new_used;
        // SAFETY: `offset` is within `[0, state.mapped)`, since `new_used
        // <= state.mapped` was just ensured above.
        let ptr = unsafe { state.base.add(offset) };
        Ok(VmmSlice {
            ptr,
            pool: self,
            birth_epoch: self.epoch.load(Ordering::Relaxed),
        })
    }

    /// The current bump-pointer offset, to be restored later via
    /// [`Self::reset_to`].
    pub fn watermark(&self) -> usize {
        self.lock().used
    }

    /// Roll the bump pointer back to a watermark saved by [`Self::watermark`].
    ///
    /// A bulk free for everything allocated since the watermark: the
    /// physical pages stay mapped, only the next allocation's offset moves
    /// back. Only valid between capture attempts, never during steady-state
    /// replay, which never runs the Rust allocation code this method rewinds.
    ///
    /// Any [`VmmSlice`] obtained after `mark` but still held past this call
    /// is stale: its address falls at or past the watermark, so a later
    /// [`Self::alloc`] can hand out the same address again. The caller must
    /// not keep such a slice alive across a reset; [`VmmSlice::as_ptr`]
    /// catches this in debug builds, but release builds hand back the
    /// address unchecked.
    ///
    /// # Panics
    /// Panics in debug builds if `mark` is past the pool's current bump
    /// offset, since that would move the pool forward rather than back.
    pub fn reset_to(&self, mark: usize) {
        let mut state = self.lock();
        debug_assert!(mark <= state.used, "reset_to must not grow the pool");
        state.used = mark.min(state.used);
        drop(state);
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for VmmPool {
    fn drop(&mut self) {
        let state = self.lock();
        for &(ptr, size) in &state.mappings {
            // SAFETY: each mapping was created by a successful `hipMemMap`
            // in `alloc` and has not been unmapped yet; no other reference
            // to `self` exists once `Drop::drop` runs.
            unsafe {
                let _ = bindings::hipMemUnmap(ptr, size);
            }
        }
        if !state.base.is_null() {
            // SAFETY: `state.base` was reserved by a successful
            // `hipMemAddressReserve` for `VMM_POOL_RESERVE_BYTES` bytes.
            unsafe {
                let _ = bindings::hipMemAddressFree(state.base, VMM_POOL_RESERVE_BYTES);
            }
        }
    }
}

/// An allocation property descriptor for a pinned, device-local VMM mapping
/// on `device_id`. Every VMM call below (the granularity query, the
/// physical page creation) needs an identical descriptor.
fn vmm_alloc_prop(device_id: i32) -> bindings::hipMemAllocationProp {
    bindings::hipMemAllocationProp {
        type_: bindings::hipMemAllocationType_hipMemAllocationTypePinned,
        __bindgen_anon_1: bindings::hipMemAllocationProp__bindgen_ty_1 {
            requestedHandleType: bindings::hipMemAllocationHandleType_hipMemHandleTypeNone,
        },
        location: bindings::hipMemLocation {
            type_: bindings::hipMemLocationType_hipMemLocationTypeDevice,
            id: device_id,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: bindings::hipMemAllocationProp__bindgen_ty_2 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
        },
    }
}

fn vmm_granularity(device_id: i32) -> Result<usize, HipError> {
    let prop = vmm_alloc_prop(device_id);
    let mut granularity: usize = 0;
    // SAFETY: `prop` is fully initialized and `granularity` is a valid
    // out-pointer for a single write.
    hip_check(unsafe {
        bindings::hipMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            bindings::hipMemAllocationGranularity_flags_hipMemAllocationGranularityRecommended,
        )
    })?;
    Ok(granularity)
}

fn reserve_address_range(size: usize) -> Result<*mut std::ffi::c_void, HipError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `ptr` is only read once the status has been checked.
    hip_check(unsafe {
        bindings::hipMemAddressReserve(&mut ptr, size, 0, std::ptr::null_mut(), 0)
    })?;
    Ok(ptr)
}

/// Map `size` bytes of fresh physical memory at `dst`, which must lie
/// inside a reserved, not-yet-mapped region of exactly that size.
///
/// Leaves nothing behind on failure: a failed `hipMemMap` releases the
/// handle `hipMemCreate` produced, and a failed `hipMemSetAccess` unmaps
/// what `hipMemMap` had just mapped before returning.
fn map_physical_pages(
    device_id: i32,
    dst: *mut std::ffi::c_void,
    size: usize,
) -> Result<(), HipError> {
    let prop = vmm_alloc_prop(device_id);
    let mut handle: bindings::hipMemGenericAllocationHandle_t = std::ptr::null_mut();
    // SAFETY: `handle` is a valid out-pointer and `prop` is fully
    // initialized.
    hip_check(unsafe { bindings::hipMemCreate(&mut handle, size, &prop, 0) })?;

    // SAFETY: `dst` is inside a reserved, unmapped region of exactly `size`
    // bytes, and `handle` was just created for exactly `size` bytes.
    if let Err(e) = hip_check(unsafe { bindings::hipMemMap(dst, size, 0, handle, 0) }) {
        // Nothing was mapped, so only the handle itself needs releasing.
        // SAFETY: `handle` was created by `hipMemCreate` above and has not
        // been released yet.
        unsafe {
            let _ = bindings::hipMemRelease(handle);
        }
        return Err(e);
    }

    let access = bindings::hipMemAccessDesc {
        location: bindings::hipMemLocation {
            type_: bindings::hipMemLocationType_hipMemLocationTypeDevice,
            id: device_id,
        },
        flags: bindings::hipMemAccessFlags_hipMemAccessFlagsProtReadWrite,
    };
    // SAFETY: `dst` is valid for `size` bytes, now mapped by the call above.
    let access_result = hip_check(unsafe { bindings::hipMemSetAccess(dst, size, &access, 1) });

    // The handle is only a temporary reference used to create the mapping;
    // once mapped, HIP keeps the physical pages alive through the mapping
    // itself. Release it on every path, success or failure.
    // SAFETY: `handle` was created above and has not been released yet.
    unsafe {
        let _ = bindings::hipMemRelease(handle);
    }

    if let Err(e) = access_result {
        // The mapping itself succeeded, so it must be torn down before
        // returning the error, or the caller would track a mapping that
        // never actually became accessible.
        // SAFETY: `dst` was mapped by the successful `hipMemMap` above.
        unsafe {
            let _ = bindings::hipMemUnmap(dst, size);
        }
        return Err(e);
    }

    Ok(())
}

/// Whether this process' HIP runtime can back [`VmmPool`]-style allocations,
/// i.e. several allocations sharing one `hipMemAddressReserve`d virtual-address
/// range.
///
/// Probes once per process (the result is cached) and never on a device
/// already known to work, so a caller that checks this before every
/// [`VmmPool::new`] pays for the real HIP calls only the first time.
///
/// # Errors
/// Returns `false` (conservatively: treat as unusable) if the probe itself
/// fails for a reason other than the defect it is checking for, e.g. the
/// granularity query or the initial reservation failing outright.
#[must_use]
pub fn vmm_usable(device_id: i32) -> bool {
    static BROKEN: OnceLock<bool> = OnceLock::new();
    !*BROKEN.get_or_init(|| probe_set_access_defect(device_id).unwrap_or(true))
}

/// Whether this HIP runtime fails `hipMemSetAccess` for an allocation that
/// is not the first in its virtual-address reservation
/// (`ROCm/rocm-systems#2516`, fixed by `ROCm/rocm-systems#2451`).
///
/// Reserves a range sized for a 1-granule and a 2-granule allocation, maps
/// both into it side by side, and calls `hipMemSetAccess` on each in turn.
/// The defective runtime validates the requested size against the
/// reservation's allocations counted from the first one in its own internal
/// list, rather than from the target allocation, so the second of the two
/// mapped here always fails that check regardless of which one happens to
/// be "first" in the runtime's bookkeeping -- one and two granules can never
/// both legitimately cover each other. A fixed runtime passes both. Frees
/// everything it allocates, on every path.
fn probe_set_access_defect(device_id: i32) -> Result<bool, HipError> {
    let granularity = vmm_granularity(device_id)?;
    let sizes = [granularity, 2 * granularity];
    let total: usize = sizes.iter().sum();
    let base = reserve_address_range(total)?;

    let mut broken = false;
    let mut mapped_ok = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;
    for size in sizes {
        // SAFETY: `dst` lies within the `total`-byte range `base` just
        // reserved, at a non-overlapping offset for each `size` in `sizes`.
        let dst = unsafe { base.add(offset) };
        match map_physical_pages(device_id, dst, size) {
            Ok(()) => mapped_ok.push((dst, size)),
            // `map_physical_pages` already unmapped/released this chunk on
            // its own failure path; nothing left to clean up for it here.
            Err(_) => broken = true,
        }
        offset += size;
    }

    for (dst, size) in mapped_ok {
        // SAFETY: each `(dst, size)` here was mapped successfully above and
        // not yet unmapped.
        unsafe {
            let _ = bindings::hipMemUnmap(dst, size);
        }
    }
    // SAFETY: `base` was reserved for exactly `total` bytes above, and every
    // mapping within it has just been released.
    unsafe {
        let _ = bindings::hipMemAddressFree(base, total);
    }
    Ok(broken)
}

/// A pool-scoped device allocation, returned by [`VmmPool::alloc`].
///
/// Borrows [`VmmPool`] so the compiler rejects any use after the pool
/// backing it is dropped. There is no `Drop` impl: freeing is
/// watermark-only, so a slice going out of scope does nothing.
/// [`VmmPool::reset_to`] is what actually reclaims the space.
///
/// A slice held past a [`VmmPool::reset_to`] that rewound its offset is
/// stale: the pool can hand its address out again. The compiler cannot
/// catch that case, since the pool is still alive. [`Self::as_ptr`] checks
/// it against the pool's epoch instead, in debug builds.
pub struct VmmSlice<'pool> {
    ptr: *mut std::ffi::c_void,
    pool: &'pool VmmPool,
    /// The pool's [`VmmPool::reset_to`] generation at the time this slice
    /// was allocated. Checked against the pool's current generation in
    /// [`Self::as_ptr`].
    birth_epoch: u64,
}

impl VmmSlice<'_> {
    /// Device pointer to the start of this allocation.
    ///
    /// # Panics
    /// Panics in debug builds if the pool was reset via
    /// [`VmmPool::reset_to`] since this slice was allocated, since the
    /// returned pointer would then be stale.
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        debug_assert_eq!(
            self.birth_epoch,
            self.pool.epoch.load(Ordering::Relaxed),
            "VmmSlice used after pool reset"
        );
        self.ptr
    }
}

// SAFETY: a device address is process-wide and carries no thread affinity.
// `VmmSlice` exposes only read-only accessors, so sharing `&VmmSlice` across
// threads cannot race.
unsafe impl Send for VmmSlice<'_> {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for VmmSlice<'_> {}

/// A device buffer of `T`, returned to its allocator's free list when dropped.
///
/// The API surface deliberately matches the `rocm_rs::hip::DeviceMemory` this
/// replaces, so no call site had to change; what changed is that `Drop` no
/// longer drains the device.
pub struct SendSyncDeviceMemory<T> {
    ptr: *mut std::ffi::c_void,
    /// Requested size in bytes. The block behind it is at least this size;
    /// see [`AllocOrigin`] for how much more.
    size: usize,
    origin: AllocOrigin,
    alloc: Arc<RocmAllocator>,
    phantom: PhantomData<T>,
}

// SAFETY: the state is a device pointer plus a refcount on the allocator. The
// pointer addresses GPU memory, not host memory, so no host thread can race on
// it; ownership is unique and mutation goes through `&mut self`.
//
// `T` is unbounded on purpose. No value of `T` is ever stored here — it is a
// size and type tag for `PhantomData` and for the `&[T]` the host transfers
// borrow — so `T`'s own thread-safety is irrelevant. A `T: Send` bound would
// read as if it were load-bearing, which is exactly the confusion to avoid.
unsafe impl<T> Send for SendSyncDeviceMemory<T> {}
// SAFETY: see the `Send` impl above.
unsafe impl<T> Sync for SendSyncDeviceMemory<T> {}

impl<T> SendSyncDeviceMemory<T> {
    pub(crate) fn new(alloc: &Arc<RocmAllocator>, count: usize) -> Result<Self, HipError> {
        let size = count * std::mem::size_of::<T>();
        let (ptr, origin) = alloc.alloc_bytes(size)?;
        Ok(Self {
            ptr,
            size,
            origin,
            alloc: alloc.clone(),
            phantom: PhantomData,
        })
    }

    /// Base device pointer. Arithmetic on it advances *bytes*.
    ///
    /// # Panics
    /// Panics in debug builds if this allocation came from a `VmmPool` that
    /// was reset via [`VmmPool::reset_to`] since it was allocated, since the
    /// returned pointer would then be stale. See [`AllocOrigin::Pool`].
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.check_not_stale();
        self.ptr
    }

    /// Panics in debug builds if this allocation is `VmmPool`-backed and the
    /// pool has since been reset past it. Shared by [`Self::as_ptr`] and
    /// [`Self::ptr_at`], the two accessors that hand back the raw pointer.
    fn check_not_stale(&self) {
        if let AllocOrigin::Pool { pool, epoch } = &self.origin {
            debug_assert_eq!(
                *epoch,
                pool.epoch.load(Ordering::Relaxed),
                "SendSyncDeviceMemory used after VmmPool reset"
            );
        }
    }

    /// Size of the allocation in bytes, as requested rather than as bucketed.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of elements the allocation holds.
    pub fn count(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Device pointer to element `offset`.
    ///
    /// [`Self::as_ptr`] hands back a `*mut c_void`, so arithmetic on it
    /// advances *bytes*. Every offset candle deals in — `Layout::start_offset`
    /// above all — counts *elements*, so it has to be scaled by the element
    /// size. Doing this by hand at each call site silently mis-addresses every
    /// tensor whose dtype is wider than a byte.
    ///
    /// # Safety
    /// `offset` must be within the allocation.
    ///
    /// # Panics
    /// Panics in debug builds if this allocation came from a `VmmPool` that
    /// was reset via [`VmmPool::reset_to`] since it was allocated. See
    /// [`Self::as_ptr`].
    pub unsafe fn ptr_at(&self, offset: usize) -> *mut std::ffi::c_void {
        self.check_not_stale();
        self.ptr.add(offset * std::mem::size_of::<T>())
    }

    /// Host-to-device copy, clamped to the shorter of the two buffers.
    ///
    /// Drains the stream first, then copies synchronously. The drain is what
    /// orders this against the queued work of whoever owned the block before —
    /// the module-level invariant, made explicit rather than left to the
    /// blocking stream's legacy synchronisation with the null stream. The copy
    /// itself stays synchronous because `data` is a borrowed host slice the
    /// caller may drop the moment this returns, and because the driver's
    /// pageable-memory path is measurably faster than an async copy that is
    /// immediately waited on.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), HipError> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }
        self.alloc.check_not_capturing()?;
        let bytes = self.transfer_bytes(std::mem::size_of_val(data));
        self.synchronize()?;
        // SAFETY: both pointers are valid for `bytes`, which is clamped to the
        // smaller of the two.
        hip_check(unsafe {
            bindings::hipMemcpy(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyHostToDevice,
            )
        })
    }

    /// Stream-ordered host-to-device copy from an owned pinned host buffer.
    ///
    /// Unlike [`Self::copy_from_host`] this does not drain the stream first:
    /// the copy is enqueued behind whatever is already queued on the owning
    /// device's stream, the same ordering [`Self::copy_from_device`] relies
    /// on for its D2D copy. Returns as soon as the copy is enqueued. The
    /// transfer may still be in flight when this call returns.
    ///
    /// Taking [`PinnedBuffer`] rather than a plain slice makes the
    /// page-locked-memory requirement a type-level fact instead of a doc
    /// comment. `src` owns its allocation, so unlike the borrowed
    /// `PinnedHostRegion` this replaced, there is no lifetime for `Drop` to
    /// protect by draining the stream — the caller must keep `src` alive (or
    /// synchronize) for as long as the transfer may still be in flight,
    /// same as reusing any other async-copy source. `src` must have been
    /// allocated on the same device this buffer was allocated on; a mismatch
    /// is rejected with `hipErrorInvalidValue` rather than trusting the
    /// caller to get it right.
    pub fn copy_from_pinned_async(&mut self, src: &PinnedBuffer<T>) -> Result<(), HipError>
    where
        T: Copy + Send + Sync,
    {
        if self.ptr.is_null() || src.size() == 0 {
            return Ok(());
        }
        if !Arc::ptr_eq(&self.alloc.stream, &src.stream) {
            return hip_check(bindings::hipError_t_hipErrorInvalidValue);
        }
        let bytes = self.transfer_bytes(src.size());
        // SAFETY: both pointers are valid for `bytes`; `src` owns its
        // allocation and the caller is responsible for keeping it alive
        // until the transfer completes.
        hip_check(unsafe {
            bindings::hipMemcpyAsync(
                self.ptr,
                src.ptr,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyHostToDevice,
                self.alloc.raw_stream(),
            )
        })
    }

    /// Device-to-host copy, clamped to the shorter of the two buffers.
    ///
    /// Same shape as [`Self::copy_from_host`]: drain, then copy. The drain is
    /// also what makes the data this reads the data the caller expects.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), HipError> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }
        self.alloc.check_not_capturing()?;
        let bytes = self.transfer_bytes(std::mem::size_of_val(data));
        self.synchronize()?;
        // SAFETY: as above.
        hip_check(unsafe {
            bindings::hipMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        })
    }

    /// Stream-ordered device-to-device copy, clamped to the shorter buffer.
    ///
    /// Both buffers belong to the same device — the only case candle produces —
    /// so one stream orders the copy against whatever produced `src` and
    /// whatever consumes `self`, with no host synchronisation.
    pub fn copy_from_device(&mut self, src: &SendSyncDeviceMemory<T>) -> Result<(), HipError> {
        if self.ptr.is_null() || src.ptr.is_null() {
            return Ok(());
        }
        let bytes = self.transfer_bytes(src.size);
        // SAFETY: both pointers are valid for `bytes`.
        hip_check(unsafe {
            bindings::hipMemcpyAsync(
                self.ptr,
                src.ptr,
                bytes,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                self.alloc.raw_stream(),
            )
        })
    }

    /// Stream-ordered `memset` over the requested extent.
    pub fn memset(&mut self, value: i32) -> Result<(), HipError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // SAFETY: `self.ptr` is valid for `self.size` bytes.
        hip_check(unsafe {
            bindings::hipMemsetAsync(self.ptr, value, self.size, self.alloc.raw_stream())
        })
    }

    fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: the allocator keeps this stream alive.
        hip_check(unsafe { bindings::hipStreamSynchronize(self.alloc.raw_stream()) })
    }

    /// Clamp a transfer to the shorter of this allocation and `other_bytes`.
    fn transfer_bytes(&self, other_bytes: usize) -> usize {
        self.size.min(other_bytes)
    }
}

impl<T> Drop for SendSyncDeviceMemory<T> {
    fn drop(&mut self) {
        self.alloc.recycle(self.ptr, &self.origin);
        self.ptr = std::ptr::null_mut();
    }
}

/// Owned pinned (page-locked) host memory, allocated via `hipHostMalloc` and
/// released via `hipHostFree`, so [`SendSyncDeviceMemory::copy_from_pinned_async`]
/// can DMA it without the driver staging through its own pinned buffer
/// first.
///
/// Unlike the borrowed `PinnedHostRegion` this replaced, `PinnedBuffer` owns
/// its allocation: there is no caller data whose lifetime `Drop` needs to
/// protect, only this buffer's own memory, which `hipHostFree` releases
/// unconditionally. `std::mem::forget` on a value of this type merely leaks
/// the allocation, the same way forgetting any other owned buffer would —
/// not a use-after-free, since nothing borrowed it. That is what lets `new`
/// be reachable through a `pub` constructor ([`super::RocmDevice::new_pinned_buffer`])
/// and the value held in a struct field across many sequential uploads,
/// amortising the `hipHostMalloc` cost instead of paying a fresh
/// `hipHostRegister`/`hipHostUnregister` pair every time.
///
/// # Reuse caveat
///
/// `Drop` does not drain the stream before freeing — there is no borrowed
/// memory whose lifetime the drain protects, only this buffer's own
/// allocation. A caller that overwrites the buffer's contents for a second
/// upload (e.g. via [`Self::as_mut_slice`]) is responsible for synchronizing
/// first, same as reusing any other async-copy source.
pub struct PinnedBuffer<T: Copy + Send + Sync> {
    ptr: *mut std::ffi::c_void,
    len: usize,
    stream: Arc<SendSyncStream>,
    _marker: PhantomData<T>,
}

impl<T: Copy + Send + Sync> PinnedBuffer<T> {
    /// Allocate `len` pinned elements of `T` via `hipHostMalloc`, owned by
    /// `stream`'s device for [`SendSyncDeviceMemory::copy_from_pinned_async`]'s
    /// stream-identity check.
    pub(crate) fn new(len: usize, stream: Arc<SendSyncStream>) -> Result<Self, HipError> {
        let size = len * std::mem::size_of::<T>();
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len,
                stream,
                _marker: PhantomData,
            });
        }
        let mut ptr = std::ptr::null_mut();
        hip_check(unsafe {
            bindings::hipHostMalloc(&mut ptr, size, bindings::hipHostMallocDefault)
        })?;
        Ok(Self {
            ptr,
            len,
            stream,
            _marker: PhantomData,
        })
    }

    /// The pinned host pointer, suitable as a `hipMemcpyAsync` source.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }

    /// Byte size of the pinned buffer.
    pub fn size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Number of elements the pinned buffer holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the pinned buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The pinned buffer's contents as a host slice.
    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: `self.ptr` is valid for `self.len` elements of `T` for the
        // lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }

    /// The pinned buffer's contents as a mutable host slice.
    ///
    /// Overwriting a buffer whose previous contents are still being read by
    /// an in-flight `copy_from_pinned_async` is the caller's responsibility
    /// to avoid — synchronize first if reusing this buffer for a new upload.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            return &mut [];
        }
        // SAFETY: `self.ptr` is valid for `self.len` elements of `T` for the
        // lifetime of `self`, and `self` is borrowed mutably here.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
}

impl<T: Copy + Send + Sync> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was successfully allocated in `new`, and
            // this is the only place that frees it. No stream drain: there
            // is no borrowed caller memory to protect, see the type's doc
            // comment's "Reuse caveat".
            unsafe {
                let _ = bindings::hipHostFree(self.ptr);
            }
        }
    }
}

// SAFETY: the allocation is a process-wide driver mapping keyed by address,
// not thread-affine; `PinnedBuffer` owns it exclusively, with no borrowed
// data whose sharing needs protecting the way `PinnedHostRegion` needed.
// Unlike `SendSyncDeviceMemory<T>`, `T` is *not* a phantom tag here:
// `as_slice`/`as_mut_slice` materialize live `&[T]`/`&mut [T]` over this
// memory, so a `T` that crosses a thread boundary through this type (by
// moving the buffer, or by sharing `&PinnedBuffer<T>`) must itself be sound
// to send/share — hence the `T: Send + Sync` bound. `T: Copy` additionally
// rules out types with drop glue, which would be unsound to abandon when
// `hipHostFree` releases the backing memory without running `T::drop`, and
// rules out niched/non-exhaustive bit patterns being read back before the
// first write. (A `T` that is `Copy + Send + Sync` but still has invalid bit
// patterns, e.g. `NonZeroU32`, is not caught by this bound — but nothing in
// this codebase instantiates `PinnedBuffer` with such a type, and callers
// always write through `as_mut_slice` before any read.)
unsafe impl<T: Copy + Send + Sync> Send for PinnedBuffer<T> {}
// SAFETY: see the `Send` impl above for the `T` bound. `as_slice`/`len`/
// `size`/`is_empty`/`as_ptr` are read-only; the only mutating method,
// `as_mut_slice`, requires `&mut self` and so cannot be called concurrently
// from shared references.
unsafe impl<T: Copy + Send + Sync> Sync for PinnedBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::{bucket_size, MAX_GRANULARITY, SMALL_GRANULARITY};

    #[test]
    fn buckets_round_up_to_the_granularity() {
        assert_eq!(bucket_size(1), SMALL_GRANULARITY);
        assert_eq!(bucket_size(SMALL_GRANULARITY), SMALL_GRANULARITY);
        assert_eq!(bucket_size(SMALL_GRANULARITY + 1), 2 * SMALL_GRANULARITY);
        // At 1 MiB the relative granularity is 128 KiB, so one byte past an
        // exact bucket steps up by 128 KiB rather than by 512 B.
        assert_eq!(bucket_size(1 << 20), 1 << 20);
        assert_eq!(bucket_size((1 << 20) + 1), (1 << 20) + (128 << 10));
    }

    /// Two tensors of the same shape must land in the same bucket, or the free
    /// list never hits during an inference loop.
    #[test]
    fn equal_sizes_share_a_bucket() {
        assert_eq!(bucket_size(4096 * 4), bucket_size(4096 * 4));
        assert_eq!(bucket_size(64 << 20), bucket_size(64 << 20));
    }

    /// A bucket must hold the request, and the granularity choice bounds the
    /// overshoot: 12.5% relative in the mid range, 32 MiB absolute on giants.
    #[test]
    fn overshoot_is_bounded() {
        for size in [1usize, 700, 4 << 10, 1 << 20, 70 << 20, 3 << 30] {
            let bucket = bucket_size(size);
            assert!(bucket >= size);
            assert!(bucket - size <= (size / 8).clamp(SMALL_GRANULARITY, MAX_GRANULARITY));
        }
    }

    /// Growing a buffer across a whole octave visits only a handful of
    /// buckets, which is what stops a decode loop — whose attention buffers
    /// grow by a few KB per generated token — from parking a fresh
    /// never-reused block every step.
    #[test]
    fn an_octave_of_growth_visits_few_buckets() {
        let mut buckets = std::collections::BTreeSet::new();
        let mut size = 1usize << 20;
        while size <= 2 << 20 {
            buckets.insert(bucket_size(size));
            size += 4 << 10;
        }
        assert!(buckets.len() <= 9, "got {} buckets", buckets.len());
    }
}
