//! GPU-level behaviour of the caching allocator in [`super::alloc`].
//!
//! Split from that module's own unit tests, which cover the pure bucket
//! arithmetic and need no device.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};

use rocm_rs::hip::bindings;

use super::{RocmDevice, VmmPool, WrapErr};
use crate::Result;

/// Returns early when the machine has no ROCm GPU, like the other ROCm test
/// modules here.
macro_rules! device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

/// A dropped buffer must come back rather than go to the driver — that is
/// the whole reason this allocator exists.
#[test]
fn a_dropped_block_is_reused() -> Result<()> {
    let dev = device!();
    let first = dev.alloc::<f32>(1024)?.as_ptr();
    let second = dev.alloc::<f32>(1024)?;
    assert_eq!(first, second.as_ptr());
    Ok(())
}

/// Two live buffers of the same size must never alias.
#[test]
fn live_blocks_do_not_alias() -> Result<()> {
    let dev = device!();
    let a = dev.alloc::<f32>(1024)?;
    let b = dev.alloc::<f32>(1024)?;
    assert_ne!(a.as_ptr(), b.as_ptr());
    Ok(())
}

/// Rounding is what lets near-miss sizes share a block, but a *smaller*
/// request must never be handed a block that cannot hold it.
#[test]
fn a_reused_block_is_large_enough() -> Result<()> {
    let dev = device!();
    let big = dev.alloc::<u8>(4096)?.as_ptr();
    // 4000 rounds to the same 4096-byte bucket, so it reuses the block.
    let small = dev.alloc::<u8>(4000)?;
    assert_eq!(big, small.as_ptr());
    assert_eq!(small.count(), 4000);
    // 5000 does not, so it gets a fresh one.
    let other = dev.alloc::<u8>(5000)?;
    assert_ne!(big, other.as_ptr());
    Ok(())
}

/// `count` reports what the caller asked for, not the bucket it landed in;
/// `clone_dtoh` sizes its host `Vec` from it.
#[test]
fn count_reflects_the_request_not_the_bucket() -> Result<()> {
    let dev = device!();
    let mem = dev.alloc::<f32>(7)?;
    assert_eq!(mem.count(), 7);
    assert_eq!(mem.size(), 28);
    Ok(())
}

/// A zero-length allocation is a null pointer, and recycling it must not
/// put a null on the free list for a later request to be handed.
#[test]
fn an_empty_allocation_is_null_and_is_not_recycled() -> Result<()> {
    let dev = device!();
    assert!(dev.alloc::<f32>(0)?.as_ptr().is_null());
    assert!(!dev.alloc::<f32>(1)?.as_ptr().is_null());
    Ok(())
}

/// Parking a block above the cache limit evicts back to the driver, largest
/// first, so the cache stays bounded while the hot small buckets survive.
#[test]
fn the_cache_cap_evicts_largest_first() -> Result<()> {
    let dev = device!();
    dev.allocator.set_cache_limit(4 << 20);
    // Park a small block, then a large one that lifts the cache past the cap:
    // the large block itself is the eviction victim, the small one survives.
    let small_ptr = {
        let small = dev.alloc::<u8>(64 << 10)?;
        let ptr = small.as_ptr();
        drop(small);
        ptr
    };
    drop(dev.alloc::<u8>(8 << 20)?);
    assert!(dev.allocator.cached_bytes() <= 4 << 20);
    // The small bucket is still parked and still hits.
    assert_eq!(dev.alloc::<u8>(64 << 10)?.as_ptr(), small_ptr);
    dev.allocator.set_cache_limit(usize::MAX);
    Ok(())
}

/// A decode-style monotonically growing sequence of allocations must reuse
/// blocks rather than park a fresh one every step: the relative bucket
/// granularity keeps the number of distinct blocks logarithmic-ish, and the
/// cap bounds what the misses can accumulate.
#[test]
fn growing_allocations_do_not_hoard_unboundedly() -> Result<()> {
    let dev = device!();
    dev.allocator.set_cache_limit(64 << 20);
    // 4 KB of growth per step from 1 MiB, like a lengthening KV sequence.
    let mut size = 1usize << 20;
    for _ in 0..2000 {
        let buf = dev.alloc::<u8>(size)?;
        drop(buf);
        size += 4 << 10;
    }
    assert!(dev.allocator.cached_bytes() <= 64 << 20);
    dev.allocator.set_cache_limit(usize::MAX);
    Ok(())
}

/// An async copy from a pinned host buffer must land the same bytes as the
/// synchronous path, once the caller synchronises before reading back.
#[test]
fn async_copy_from_pinned_buffer_matches_sync() -> Result<()> {
    let dev = device!();
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let mut pin = dev.new_pinned_buffer::<f32>(data.len())?;
    pin.as_mut_slice().copy_from_slice(&data);

    let mut dst = dev.alloc::<f32>(data.len())?;
    dst.copy_from_pinned_async(&pin).w()?;
    dev.synchronize()?;

    let readback = dev.clone_dtoh(&dst)?;
    assert_eq!(readback, data);
    Ok(())
}

/// A buffer pinned on one device's stream must not be usable to copy into a
/// buffer allocated on another: there is no cross-stream ordering guarantee
/// between the two devices, so accepting the mismatch would let the copy
/// race whatever the destination device's stream does next.
#[test]
fn async_copy_rejects_pinned_buffer_on_another_stream() -> Result<()> {
    let a = device!();
    let b = match RocmDevice::new_with_stream(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let mut pin = a.new_pinned_buffer::<f32>(data.len())?;
    pin.as_mut_slice().copy_from_slice(&data);

    let mut dst = b.alloc::<f32>(data.len())?;
    assert!(dst.copy_from_pinned_async(&pin).is_err());
    Ok(())
}

/// Synchronising before dropping a `PinnedBuffer` used in an async copy must
/// produce the correct data: this is the documented-safe usage pattern
/// (`copy_from_pinned_async`'s caller must keep the source alive, or
/// synchronize, for as long as the transfer may still be in flight).
#[test]
fn sync_then_drop_pinned_buffer_is_safe() -> Result<()> {
    let dev = device!();
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let mut pin = dev.new_pinned_buffer::<f32>(data.len())?;
    pin.as_mut_slice().copy_from_slice(&data);

    let mut dst = dev.alloc::<f32>(data.len())?;
    dst.copy_from_pinned_async(&pin).w()?;
    dev.synchronize()?;
    drop(pin);

    let readback = dev.clone_dtoh(&dst)?;
    assert_eq!(readback, data);
    Ok(())
}

/// Replaying the same allocation sequence against a `VmmPool` reset to an
/// earlier watermark must return the same addresses every time. This is the
/// property HIP graph replay depends on: a captured graph bakes in the
/// addresses recorded during capture.
#[test]
fn vmm_pool_allocs_return_stable_addresses() -> Result<()> {
    let _dev = device!();
    let pool = VmmPool::new(0).w()?;
    let sizes = [64usize, 256, 1024, 4096];

    let mark = pool.watermark();
    let mut first_pass = Vec::new();
    for &size in &sizes {
        first_pass.push(pool.alloc(size).w()?.as_ptr());
    }

    pool.reset_to(mark);
    let mut second_pass = Vec::new();
    for &size in &sizes {
        second_pass.push(pool.alloc(size).w()?.as_ptr());
    }

    assert_eq!(first_pass, second_pass);
    Ok(())
}

/// An allocation larger than what is currently mapped must still succeed,
/// mapping in more physical pages on demand, and must not alias a live
/// allocation that came before it.
#[test]
fn vmm_pool_grows_past_initial_mapping() -> Result<()> {
    let _dev = device!();
    let pool = VmmPool::new(0).w()?;

    // Comfortably larger than any realistic HIP allocation granularity
    // (typically a few MiB), to force at least one grow, while staying far
    // below the pool's 32 GiB virtual reservation.
    let first = pool.alloc(256 << 20).w()?;
    assert!(!first.as_ptr().is_null());
    let second = pool.alloc(256 << 20).w()?;
    assert_ne!(first.as_ptr(), second.as_ptr());
    Ok(())
}

/// Resetting to a watermark saved mid-sequence must let a later allocation
/// reuse exactly the space freed by the reset, without mapping new physical
/// pages for it.
#[test]
fn vmm_pool_reset_reclaims_space() -> Result<()> {
    let _dev = device!();
    let pool = VmmPool::new(0).w()?;

    let _first = pool.alloc(1024).w()?;
    let mark = pool.watermark();
    let second = pool.alloc(2048).w()?.as_ptr();

    pool.reset_to(mark);
    let third = pool.alloc(2048).w()?;
    assert_eq!(second, third.as_ptr());
    Ok(())
}

/// Dropping a pool that still has more than one outstanding mapping must not
/// crash. The `Drop` impl unmaps each tracked mapping individually rather
/// than in one bulk call, which is the actual ROCm runtime workaround this
/// guards: a single allocation would pass even with a regression to one bulk
/// `hipMemUnmap` over the whole mapped range.
#[test]
fn vmm_pool_drop_does_not_crash() -> Result<()> {
    let _dev = device!();
    let pool = VmmPool::new(0).w()?;
    // Comfortably larger than any realistic HIP allocation granularity, so
    // each call forces its own grow and leaves its own entry in `mappings`.
    let _first = pool.alloc(64 << 20).w()?;
    let _second = pool.alloc(64 << 20).w()?;
    Ok(())
}

/// Data written through a `VmmSlice` must read back unchanged: pointer
/// equality alone (checked by the other tests) wouldn't catch a
/// misconfigured `hipMemSetAccess` that still reports success.
#[test]
fn vmm_pool_write_readback_round_trips() -> Result<()> {
    let _dev = device!();
    let pool = VmmPool::new(0).w()?;
    let data: Vec<u8> = (0..4096u32).map(|i| (i % 256) as u8).collect();

    let slice = pool.alloc(data.len()).w()?;
    // SAFETY: `slice` is a device allocation of at least `data.len()` bytes,
    // mapped read-write by `VmmPool::alloc`; `data` is a valid host slice of
    // the same length.
    let status = unsafe {
        bindings::hipMemcpy(
            slice.as_ptr(),
            data.as_ptr() as *const std::ffi::c_void,
            data.len(),
            bindings::hipMemcpyKind_hipMemcpyHostToDevice,
        )
    };
    assert_eq!(status, bindings::hipError_t_hipSuccess);

    let mut readback = vec![0u8; data.len()];
    // SAFETY: same allocation, reversed direction, same length.
    let status = unsafe {
        bindings::hipMemcpy(
            readback.as_mut_ptr() as *mut std::ffi::c_void,
            slice.as_ptr(),
            data.len(),
            bindings::hipMemcpyKind_hipMemcpyDeviceToHost,
        )
    };
    assert_eq!(status, bindings::hipError_t_hipSuccess);

    assert_eq!(readback, data);
    Ok(())
}

/// Allocations made inside a graph capture session must come from the
/// `VmmPool` (stable addresses across a reset watermark), and allocations
/// outside the session must keep using the bucketed free list (recycled
/// addresses), with the session restoring bucketed routing once ended.
#[test]
fn capture_mode_routes_allocs_through_vmm_pool() -> Result<()> {
    let dev = device!();
    let pool = Arc::new(VmmPool::new(0).w()?);

    // Outside the session: ordinary bucketed reuse.
    let bucketed_first = dev.alloc::<u8>(4096)?.as_ptr();
    drop(dev.alloc::<u8>(4096)?);

    // Inside the session: replaying the same allocation sequence against a
    // reset watermark reproduces the same VMM address. If capture mode had
    // not actually engaged, this would just hit the bucketed free list and
    // reproduce `bucketed_first` instead, which the `assert_ne!` below rules
    // out. Each pass commits via `.end()`, so the watermark has to be reset
    // by hand between passes to get the same addresses again.
    let mark = pool.watermark();
    let session = dev.begin_graph_capture(pool.clone())?;
    let vmm_first = dev.alloc::<u8>(4096)?.as_ptr();
    session.end().w()?;

    pool.reset_to(mark);
    let session = dev.begin_graph_capture(pool.clone())?;
    let vmm_second = dev.alloc::<u8>(4096)?.as_ptr();
    session.end().w()?;

    assert_eq!(vmm_first, vmm_second);
    assert_ne!(vmm_first, bucketed_first);

    // The session restores bucketed routing once ended.
    assert_eq!(dev.alloc::<u8>(4096)?.as_ptr(), bucketed_first);
    Ok(())
}

/// Freeing an allocation made during a capture session must not push it onto
/// the bucketed free list: its address is owned by the `VmmPool`, and
/// recycling it into the bucketed allocator's bookkeeping would eventually
/// `hipFree` an address `hipMemAddressReserve` owns.
#[test]
fn capture_mode_allocs_are_not_recycled() -> Result<()> {
    let dev = device!();
    let pool = Arc::new(VmmPool::new(0).w()?);
    let before = dev.allocator.cached_bytes();

    {
        let _session = dev.begin_graph_capture(pool.clone())?;
        drop(dev.alloc::<u8>(4096)?);
    }

    assert_eq!(dev.allocator.cached_bytes(), before);
    Ok(())
}

/// A second capture attempt started before the first one ends must be
/// rejected with an error rather than silently corrupting the first
/// capture's allocator routing, and must not panic: a multi-request server
/// retrying a failed capture is exactly the caller this has to survive.
#[test]
fn begin_graph_capture_rejects_nested_capture() -> Result<()> {
    let dev = device!();
    let pool = Arc::new(VmmPool::new(0).w()?);

    let _outer = dev.begin_graph_capture(pool.clone())?;
    assert!(dev.begin_graph_capture(pool.clone()).is_err());
    Ok(())
}

/// Two threads racing `begin_graph_capture` against the same pool and
/// device must leave exactly one of them holding the session. The
/// sequential nested-capture test above cannot exercise this: it never
/// gives a second attempt a chance to run while the first is still being
/// set up, which is exactly the window the `VmmPool::in_use` guard and the
/// allocator's `capture_pool` check both have to close.
#[test]
fn begin_graph_capture_rejects_concurrent_capture() -> Result<()> {
    let dev = device!();
    let pool = Arc::new(VmmPool::new(0).w()?);
    let start = Arc::new(Barrier::new(2));
    let done = Arc::new(Barrier::new(2));
    let first_ok = Arc::new(AtomicBool::new(false));

    let other_dev = dev.clone();
    let other_pool = pool.clone();
    let other_start = start.clone();
    let other_done = done.clone();
    let other_first_ok = first_ok.clone();
    let handle = std::thread::spawn(move || {
        let session = other_dev.begin_graph_capture(other_pool);
        other_first_ok.store(session.is_ok(), Ordering::SeqCst);
        other_start.wait();
        // Keep the session alive until the main thread's attempt is done.
        other_done.wait();
        drop(session);
    });

    start.wait();
    let second_ok = dev.begin_graph_capture(pool).is_ok();
    done.wait();
    handle.join().unwrap();

    assert_ne!(
        first_ok.load(Ordering::SeqCst),
        second_ok,
        "exactly one concurrent capture attempt must succeed"
    );
    Ok(())
}

/// Dropping a capture session without calling `.end()` must roll the
/// `VmmPool` watermark back to where it was when the session began, so a
/// retried capture attempt gets the same addresses as the abandoned one
/// instead of drifting forward every time a capture is aborted.
#[test]
fn graph_capture_session_rolls_back_pool_on_abandoned_capture() -> Result<()> {
    let dev = device!();
    let pool = Arc::new(VmmPool::new(0).w()?);
    let mark = pool.watermark();

    {
        let _session = dev.begin_graph_capture(pool.clone())?;
        drop(dev.alloc::<u8>(4096)?);
        // `_session` is dropped here without `.end()`.
    }

    assert_eq!(pool.watermark(), mark);
    Ok(())
}

/// A capture session that ends successfully must leave the `VmmPool`
/// watermark advanced rather than rolling it back: the addresses it
/// allocated are baked into the returned graph and must stay live for
/// replay.
#[test]
fn graph_capture_session_keeps_watermark_after_successful_end() -> Result<()> {
    let dev = device!();
    let pool = Arc::new(VmmPool::new(0).w()?);
    let mark = pool.watermark();

    let session = dev.begin_graph_capture(pool.clone())?;
    drop(dev.alloc::<u8>(4096)?);
    session.end().w()?;

    assert!(pool.watermark() > mark);
    Ok(())
}

/// A `VmmSlice` held across the [`VmmPool::reset_to`] that invalidated it is
/// stale: `as_ptr` must catch that in debug builds rather than silently
/// handing back an address the pool may have already reused.
///
/// `#[should_panic]` tests cannot return `Result`, so the no-GPU skip from
/// the `device!` macro is replaced with a manual panic carrying the same
/// message the real assertion would produce.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "VmmSlice used after pool reset")]
fn vmm_slice_panics_on_stale_use() {
    if RocmDevice::new(0).is_err() {
        panic!("VmmSlice used after pool reset");
    }
    let pool = VmmPool::new(0).unwrap();
    let mark = pool.watermark();
    let slice = pool.alloc(1024).unwrap();
    pool.reset_to(mark);
    let _ = slice.as_ptr();
}
