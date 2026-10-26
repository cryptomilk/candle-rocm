// SPDX-License-Identifier: MIT
//! GPU-level tests for the HIP graph capture/replay wrapper in
//! [`super::hip_graph`].

use super::{HipGraphCapture, RocmDevice};
use crate::Result;
use rocm_rs::hip::bindings;

/// Returns early when the machine has no `ROCm` GPU, like the other `ROCm`
/// test modules here.
macro_rules! device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

fn hip_check(status: bindings::hipError_t) -> Result<()> {
    if status == bindings::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(rocm_rs::hip::error::Error::new(status).into())
    }
}

/// A `hipMemsetAsync` recorded once and replayed N times must produce the
/// same result as issuing that same call N times without capture. The
/// whole point of graph replay is that it is observably equivalent to the
/// eager path, just without per-call dispatch overhead.
#[test]
fn capture_replay_memset_matches_uncaptured() -> Result<()> {
    const LEN: usize = 1024;
    const REPLAYS: usize = 4;
    const VALUE: i32 = 0x5a;

    let dev = device!();
    let stream = dev.stream();

    let mut reference = dev.alloc::<u8>(LEN)?;
    for _ in 0..REPLAYS {
        reference.memset(VALUE)?;
    }
    dev.synchronize()?;
    let reference_data = dev.clone_dtoh(&reference)?;

    let captured = dev.alloc::<u8>(LEN)?;
    let capture = HipGraphCapture::begin(stream)?;
    // SAFETY: `captured.as_ptr()` is valid for `LEN` bytes for the lifetime
    // of `captured`, and `stream` is in capture mode, so this call is
    // recorded into the graph rather than executed immediately.
    hip_check(unsafe { bindings::hipMemsetAsync(captured.as_ptr(), VALUE, LEN, stream.as_raw()) })?;
    let graph = capture.end()?.expect("capture recorded a hipMemsetAsync");
    assert!(!graph.is_empty()?);
    let exec = graph.instantiate()?;
    exec.upload(stream)?;

    for _ in 0..REPLAYS {
        exec.launch(stream)?;
    }
    dev.synchronize()?;
    let captured_data = dev.clone_dtoh(&captured)?;

    assert_eq!(reference_data, captured_data);
    Ok(())
}

/// Dropping a capture without calling `.end()` must leave the stream usable.
/// Otherwise an aborted capture (e.g. an error partway through recording)
/// would permanently wedge the device's one stream.
#[test]
fn drop_without_end_restores_the_stream() -> Result<()> {
    let dev = device!();
    let stream = dev.stream();

    {
        let _capture = HipGraphCapture::begin(stream)?;
        // Dropped here without `.end()`.
    }

    let mut buf = dev.alloc::<u8>(64)?;
    buf.memset(7)?;
    dev.synchronize()?;
    let data = dev.clone_dtoh(&buf)?;
    assert!(data.iter().all(|&b| b == 7));
    Ok(())
}

/// Updating an executable with the graph it was instantiated from is the
/// no-topology-change case: it must report success, not force a
/// re-instantiation.
#[test]
fn update_with_unchanged_topology_reports_success() -> Result<()> {
    let dev = device!();
    let stream = dev.stream();

    let buf = dev.alloc::<u8>(64)?;
    let capture = HipGraphCapture::begin(stream)?;
    // SAFETY: `buf.as_ptr()` is valid for 64 bytes for the lifetime of
    // `buf`, and `stream` is in capture mode.
    hip_check(unsafe { bindings::hipMemsetAsync(buf.as_ptr(), 0, 64, stream.as_raw()) })?;
    let graph = capture.end()?.expect("capture recorded a hipMemsetAsync");
    let mut exec = graph.instantiate()?;

    let result = exec.update(&graph)?;
    assert!(result.is_success());
    Ok(())
}

/// Updating an executable with a graph whose topology differs (a second
/// node added) must report that the topology changed, not silently patch
/// the mismatched graph in place.
#[test]
fn update_with_changed_topology_reports_topology_changed() -> Result<()> {
    let dev = device!();
    let stream = dev.stream();

    let buf = dev.alloc::<u8>(64)?;
    let capture = HipGraphCapture::begin(stream)?;
    // SAFETY: `buf.as_ptr()` is valid for 64 bytes for the lifetime of
    // `buf`, and `stream` is in capture mode.
    hip_check(unsafe { bindings::hipMemsetAsync(buf.as_ptr(), 0, 64, stream.as_raw()) })?;
    let graph = capture.end()?.expect("capture recorded a hipMemsetAsync");
    let mut exec = graph.instantiate()?;

    let capture2 = HipGraphCapture::begin(stream)?;
    // SAFETY: `buf.as_ptr()` is valid for 64 bytes for the lifetime of
    // `buf`, and `stream` is in capture mode. Two memsets instead of one
    // changes the node count, hence the topology.
    hip_check(unsafe { bindings::hipMemsetAsync(buf.as_ptr(), 0, 64, stream.as_raw()) })?;
    hip_check(unsafe { bindings::hipMemsetAsync(buf.as_ptr(), 1, 64, stream.as_raw()) })?;
    let graph2 = capture2
        .end()?
        .expect("capture recorded two hipMemsetAsync calls");

    let result = exec.update(&graph2)?;
    assert!(!result.is_success());
    Ok(())
}

/// A capture that enqueues no GPU operations comes back as `Some` on this
/// platform, unlike CUDA's `cudaStreamEndCapture`, which reports a null
/// graph. `HipGraph::is_empty` is what actually detects it.
#[test]
fn end_with_no_enqueued_ops_yields_an_empty_graph() -> Result<()> {
    let dev = device!();
    let stream = dev.stream();

    let capture = HipGraphCapture::begin(stream)?;
    let graph = capture
        .end()?
        .expect("HIP returns Some even for an empty capture");
    assert!(graph.is_empty()?);
    Ok(())
}

/// Two threads each running an independent capture on their own device/stream
/// at the same time must not corrupt each other's capture bookkeeping.
/// Regression test for `hipStreamCaptureModeRelaxed`, which *disables* the
/// HIP runtime's cross-thread capture-safety checks instead of scoping them,
/// so two concurrent `hipStreamBeginCapture`/`hipStreamEndCapture` pairs raced
/// inside the driver's shared capture state and segfaulted.
/// `hipStreamCaptureModeThreadLocal` scopes that tracking to each thread
/// instead.
#[test]
fn concurrent_independent_captures_on_separate_threads() -> Result<()> {
    use std::sync::{Arc, Barrier};
    use std::thread;

    const LEN: usize = 1024;
    const VALUE: i32 = 0x5a;
    // One shot rarely lands two threads inside hipStreamBeginCapture /
    // hipStreamEndCapture at the same instant; many threads times many
    // rounds widens the window until some pair of calls actually overlaps.
    const ROUNDS: usize = 200;
    const THREADS: usize = 8;

    fn run(barrier: &Barrier) -> Result<()> {
        let dev = device!();
        let stream = dev.stream();
        let buf = dev.alloc::<u8>(LEN)?;

        for _ in 0..ROUNDS {
            barrier.wait();
            let capture = HipGraphCapture::begin(stream)?;
            // SAFETY: `buf.as_ptr()` is valid for `LEN` bytes for the
            // lifetime of `buf`, and `stream` is in capture mode.
            hip_check(unsafe {
                bindings::hipMemsetAsync(buf.as_ptr(), VALUE, LEN, stream.as_raw())
            })?;
            let graph = capture.end()?.expect("capture recorded a hipMemsetAsync");
            let exec = graph.instantiate()?;
            exec.launch(stream)?;
            dev.synchronize()?;

            let data = dev.clone_dtoh(&buf)?;
            assert!(data.iter().all(|&b| b == VALUE as u8));
        }
        Ok(())
    }

    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (1..THREADS)
        .map(|_| {
            let b = barrier.clone();
            thread::spawn(move || run(&b))
        })
        .collect();
    let main_result = run(&barrier);
    for handle in handles {
        handle.join().expect("thread panicked")?;
    }
    main_result?;
    Ok(())
}

/// A host-device copy on one thread while another thread holds an unrelated
/// stream in capture must succeed, not fail or corrupt the capture.
/// Regression test for `copy_from_host`/`copy_to_host` issuing a blocking
/// `hipMemcpy`, which implicitly depends on HIP's legacy/default stream —
/// illegal while *any* stream in the process is capturing, even one
/// completely unrelated to the copy.
#[test]
fn host_copy_concurrent_with_unrelated_capture() -> Result<()> {
    use std::sync::{Arc, Barrier};
    use std::thread;

    const LEN: usize = 1024;
    const VALUE: i32 = 0x5a;

    // Two barriers pin the main thread's copy strictly inside the capture
    // window: the capturing thread cannot reach `end()` until the second
    // wait, so the copy is guaranteed to run while the stream is still
    // capturing rather than racing to land before or after it.
    let entered_capture = Arc::new(Barrier::new(2));
    let copy_done = Arc::new(Barrier::new(2));

    let capturing = {
        let entered_capture = entered_capture.clone();
        let copy_done = copy_done.clone();
        thread::spawn(move || -> Result<()> {
            let dev = device!();
            let stream = dev.stream();
            let buf = dev.alloc::<u8>(LEN)?;

            let capture = HipGraphCapture::begin(stream)?;
            entered_capture.wait();
            copy_done.wait();
            // SAFETY: `buf.as_ptr()` is valid for `LEN` bytes for the
            // lifetime of `buf`, and `stream` is in capture mode.
            hip_check(unsafe {
                bindings::hipMemsetAsync(buf.as_ptr(), VALUE, LEN, stream.as_raw())
            })?;
            capture.end()?;
            Ok(())
        })
    };

    let dev = device!();
    let mut buf = dev.alloc::<u8>(LEN)?;
    let host_data = vec![VALUE as u8; LEN];

    entered_capture.wait();
    buf.copy_from_host(&host_data)?;
    let mut roundtrip = vec![0u8; LEN];
    buf.copy_to_host(&mut roundtrip)?;
    copy_done.wait();
    assert_eq!(roundtrip, host_data);

    capturing.join().expect("thread panicked")?;
    Ok(())
}

/// Two threads each capturing a real multi-kernel rocBLAS matmul chain
/// through the production path (`RocmDevice::begin_graph_capture`'s
/// `VmmPool`-routed intermediate allocations, warmup passes, then
/// instantiate/upload/launch — the same shape as `GraphPlan::capture`) at
/// the same time must not corrupt each other's capture bookkeeping.
///
/// [`concurrent_independent_captures_on_separate_threads`] covers the same
/// claim with a single `hipMemsetAsync` and did not reproduce the original
/// segfault even at 8 threads x 200 rounds; this mirrors the heavier,
/// longer-duration, rocBLAS-and-VMM-pool-involving workload that did.
#[test]
fn concurrent_graph_plan_style_captures_do_not_corrupt_each_other() -> Result<()> {
    use super::VmmPool;
    use crate::{Device, Tensor};
    use std::sync::{Arc, Barrier};
    use std::thread;

    const ROUNDS: usize = 20;
    const THREADS: usize = 8;

    fn matmul_chain(a: &Tensor, w1: &Tensor, w2: &Tensor) -> Result<Tensor> {
        a.matmul(w1)?.matmul(w2)
    }

    fn run(barrier: &Barrier) -> Result<()> {
        let Ok(device) = Device::new_rocm(0) else {
            return Ok(());
        };
        let a = Tensor::randn(0f32, 1.0, (4, 64), &device)?;
        let w1 = Tensor::randn(0f32, 1.0, (64, 64), &device)?;
        let w2 = Tensor::randn(0f32, 1.0, (64, 32), &device)?;

        // Warm up kernel JIT/metadata outside capture, matching
        // `GraphPlan::capture`.
        for _ in 0..2 {
            let _ = matmul_chain(&a, &w1, &w2)?;
        }
        let rocm_device = device.as_rocm_device()?.clone();
        rocm_device.synchronize()?;

        for _ in 0..ROUNDS {
            let pool = Arc::new(VmmPool::new(0)?);
            barrier.wait();
            let session = rocm_device.begin_graph_capture(pool)?;
            let out = matmul_chain(&a, &w1, &w2);
            let graph = session.end();
            let out = out?;
            let graph = graph?.expect("capture recorded matmul kernels");
            assert!(!graph.is_empty()?);
            let exec = graph.instantiate()?;
            exec.upload(rocm_device.stream())?;
            exec.launch(rocm_device.stream())?;
            rocm_device.synchronize()?;
            let _ = out.flatten_all()?.to_vec1::<f32>()?;
        }
        Ok(())
    }

    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (1..THREADS)
        .map(|_| {
            let b = barrier.clone();
            thread::spawn(move || run(&b))
        })
        .collect();
    let main_result = run(&barrier);
    for handle in handles {
        handle.join().expect("thread panicked")?;
    }
    main_result?;
    Ok(())
}
