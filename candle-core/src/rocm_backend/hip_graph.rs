// SPDX-License-Identifier: MIT
//! Type-safe RAII wrappers over HIP's graph capture/replay API.
//!
//! Three types encode the graph lifecycle as a linear state machine:
//! [`HipGraphCapture`] (in-progress capture, move-only) -> [`HipGraph`]
//! (captured graph, owns `hipGraph_t`) -> [`HipGraphExec`] (instantiated
//! executable, owns `hipGraphExec_t`). Each transition consumes the source or
//! borrows it immutably, so invalid transitions (launching an
//! un-instantiated graph, ending an already-ended capture) are compile-time
//! errors.
//!
//! # Caller responsibilities
//!
//! The wrapper takes a [`Stream`] reference, not a `RocmDevice`. Two things
//! are the caller's responsibility, not this module's:
//!
//! - **hipBLAS mutex.** If the captured region includes hipBLAS calls, the
//!   caller must hold whatever lock guards hipBLAS handle lifecycle for the
//!   capture's duration. A concurrent handle teardown on another thread can
//!   corrupt the capture.
//! - **Address stability.** Replayed kernels target the device pointers
//!   recorded during capture. If a tensor's backing allocation moves between
//!   capture and replay, replay silently corrupts memory or reads stale
//!   data. The caller must ensure address stability for anything the
//!   captured region touches.

use rocm_rs::hip::bindings;
use rocm_rs::hip::error::Error as HipError;
use rocm_rs::hip::Stream;

/// Result of [`HipGraphExec::update`], wrapping `hipGraphExecUpdateResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExecUpdateResult {
    /// The update succeeded; the executable graph now reflects the new
    /// graph's kernel arguments. Replay via [`HipGraphExec::launch`] will use
    /// the updated values.
    Success,
    /// The graph topology changed (nodes added, removed, or reordered). The
    /// caller must re-instantiate via [`HipGraph::instantiate`].
    TopologyChanged,
    /// A variant this wrapper does not enumerate individually. The raw
    /// `hipGraphExecUpdateResult` value is preserved for diagnostics; every
    /// value other than `Success` means the caller must re-instantiate.
    Other(u32),
}

impl GraphExecUpdateResult {
    fn from_raw(raw: bindings::hipGraphExecUpdateResult) -> Self {
        match raw {
            bindings::hipGraphExecUpdateResult_hipGraphExecUpdateSuccess => Self::Success,
            bindings::hipGraphExecUpdateResult_hipGraphExecUpdateErrorTopologyChanged => {
                Self::TopologyChanged
            }
            other => Self::Other(other),
        }
    }

    /// Whether the update succeeded and the executable graph is ready to
    /// replay with the new parameters.
    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

/// An in-progress graph capture session.
///
/// Created by [`Self::begin`], which puts a stream into capture mode.
/// Consumes itself on [`Self::end`], which extracts the captured
/// [`HipGraph`]. If `.end()` is never called, `Drop` ends the capture and
/// discards the resulting graph. This prevents leaving the stream stuck in
/// capture mode, which would make every subsequent enqueue on that stream
/// fail.
///
/// Not `Clone`, not `Copy`. A capture session is a unique, linear
/// resource.
pub struct HipGraphCapture<'s> {
    /// The stream currently in capture mode. Borrowed rather than copied as
    /// a raw handle, so the borrow checker ties this capture's lifetime to
    /// the `Stream` (and the `RocmDevice` that owns it): the stream cannot
    /// be destroyed while a capture referencing it is still alive.
    stream: &'s Stream,
    /// Set by `.end()` so `Drop` knows the capture was already consumed
    /// normally and should not try to end it again.
    ended: bool,
}

impl<'s> HipGraphCapture<'s> {
    /// Begin capturing GPU operations enqueued on `stream`.
    ///
    /// Every GPU operation enqueued on `stream` after this call, and before
    /// the matching [`Self::end`], is recorded into a graph rather than
    /// executed immediately. `stream` must not already be in capture mode.
    ///
    /// Concurrent activity on other streams from other threads is tolerated
    /// while this capture is in progress, including another thread running
    /// its own independent capture at the same time. A multi-request server
    /// process can keep serving other streams normally, or capturing a graph
    /// of its own, while only this one stream is being captured here.
    ///
    /// # Errors
    ///
    /// Returns an error if `hipStreamBeginCapture` fails, e.g. because
    /// `stream` is already being captured.
    pub fn begin(stream: &'s Stream) -> Result<Self, HipError> {
        // SAFETY: `stream` is a valid, non-capturing stream handle borrowed
        // from a live `Stream` for the duration of this call.
        // `hipStreamCaptureModeThreadLocal` scopes the runtime's capture
        // legality tracking to this host thread, so another thread capturing
        // its own stream concurrently cannot corrupt this capture's
        // bookkeeping. `Relaxed` disables that tracking process-wide instead
        // of scoping it, which left two threads' concurrent
        // hipStreamBeginCapture/hipStreamEndCapture pairs free to race inside
        // the driver's shared capture state and segfault; `Global` is the one
        // mode that would actually rule out the legitimate case of another
        // thread using an unrelated stream while this capture is active.
        hip_check(unsafe {
            bindings::hipStreamBeginCapture(
                stream.as_raw(),
                bindings::hipStreamCaptureMode_hipStreamCaptureModeThreadLocal,
            )
        })?;
        Ok(Self {
            stream,
            ended: false,
        })
    }

    /// End the capture and return the recorded graph.
    ///
    /// Consumes `self`, so the capture handle cannot be used afterwards. The
    /// returned [`HipGraph`] can be instantiated into a [`HipGraphExec`] for
    /// replay. Returns `Ok(None)` if `hipStreamEndCapture` itself reports no
    /// graph. HIP's documented out-parameter contract allows this, mirroring
    /// `cudaStreamEndCapture`'s behavior on an empty capture.
    ///
    /// This has not been observed in practice on this platform: a capture
    /// that enqueues no GPU operations still comes back as `Some`, holding a
    /// valid, zero-node graph (CUDA's own behavior differs here). Callers
    /// that need to detect "nothing was captured" on HIP should check
    /// [`HipGraph::is_empty`] on the returned graph rather than relying on
    /// this method returning `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if `hipStreamEndCapture` fails. The stream is left
    /// in capture mode in that case, so `Drop` will retry the end-capture
    /// when `self` goes out of scope.
    pub fn end(mut self) -> Result<Option<HipGraph>, HipError> {
        let mut graph: bindings::hipGraph_t = std::ptr::null_mut();
        // SAFETY: `self.stream` was put into capture mode by `begin` and has
        // not been ended yet; `graph` is a valid out-pointer.
        hip_check(unsafe { bindings::hipStreamEndCapture(self.stream.as_raw(), &raw mut graph) })?;
        // Only reached once `hipStreamEndCapture` has actually succeeded, so
        // `Drop` never skips its own recovery attempt after a failed end.
        self.ended = true;
        if graph.is_null() {
            return Ok(None);
        }
        Ok(Some(HipGraph { graph }))
    }
}

impl Drop for HipGraphCapture<'_> {
    fn drop(&mut self) {
        if !self.ended {
            // The capture was abandoned (`.end()` was never called, or it
            // was called but `hipStreamEndCapture` failed). Retrying
            // `hipStreamEndCapture` pulls the stream back out of capture
            // mode; without this, the stream stays stuck in capture mode
            // and every subsequent enqueue on it fails. Errors here are
            // discarded. There is nothing more useful to do from `Drop`.
            let mut graph: bindings::hipGraph_t = std::ptr::null_mut();
            // SAFETY: `self.stream` is (or was) in capture mode from
            // `begin`, and `graph` is a valid out-pointer.
            unsafe {
                let _ = bindings::hipStreamEndCapture(self.stream.as_raw(), &raw mut graph);
                if !graph.is_null() {
                    let _ = bindings::hipGraphDestroy(graph);
                }
            }
        }
    }
}

// SAFETY: `HipGraphCapture` holds a `&Stream`, a borrow of a process-wide
// driver object with no thread affinity, plus a `bool`. Moving this value
// across threads is sound. The handle addresses driver state, not host
// thread-local memory. HIP rejects `hipStreamBeginCapture` on an
// already-capturing stream, so no two `HipGraphCapture`s can alias one
// stream.
unsafe impl Send for HipGraphCapture<'_> {}
// SAFETY: `HipGraphCapture` has no `&self` method that mutates driver state.
// `.end()` takes `self` by value, so it cannot run concurrently with
// anything else observing the same value.
unsafe impl Sync for HipGraphCapture<'_> {}

/// A captured HIP graph, ready to be instantiated into an executable.
///
/// Owns a `hipGraph_t` handle; `Drop` calls `hipGraphDestroy`. Not `Clone`,
/// not `Copy`.
pub struct HipGraph {
    graph: bindings::hipGraph_t,
}

impl HipGraph {
    /// Instantiate this graph into an executable that can be launched on a
    /// stream.
    ///
    /// The returned [`HipGraphExec`] encodes the graph's topology and kernel
    /// arguments as a single launchable unit. Re-instantiation is needed
    /// only when the topology changes; for parameter-only updates (e.g. a
    /// kernel's device pointer argument changed but the graph structure is
    /// identical), use [`HipGraphExec::update`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if `hipGraphInstantiate` fails.
    pub fn instantiate(&self) -> Result<HipGraphExec, HipError> {
        let mut exec: bindings::hipGraphExec_t = std::ptr::null_mut();
        // SAFETY: `self.graph` is a valid graph handle produced by a
        // successful `hipStreamEndCapture`. The error-node and log-buffer
        // out-parameters are null, which HIP documents as "not requested".
        hip_check(unsafe {
            bindings::hipGraphInstantiate(
                &raw mut exec,
                self.graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        })?;
        Ok(HipGraphExec { exec })
    }

    /// Number of nodes recorded in this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `hipGraphGetNodes` fails.
    pub fn node_count(&self) -> Result<usize, HipError> {
        let mut count: usize = 0;
        // SAFETY: `self.graph` is a valid graph handle. Passing a null
        // `nodes` pointer with a valid `numNodes` out-pointer is HIP's
        // documented way to query the node count without listing the nodes
        // themselves.
        hip_check(unsafe {
            bindings::hipGraphGetNodes(self.graph, std::ptr::null_mut(), &raw mut count)
        })?;
        Ok(count)
    }

    /// Whether this graph recorded no operations at all.
    ///
    /// On this platform, [`HipGraphCapture::end`] has been observed to
    /// return a valid, non-null graph handle even for a capture that
    /// enqueued no GPU work. This differs from CUDA's `cudaStreamEndCapture`,
    /// which returns a null graph for the same situation, so this is the
    /// reliable way to detect that case on HIP.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying node-count query fails.
    pub fn is_empty(&self) -> Result<bool, HipError> {
        Ok(self.node_count()? == 0)
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        // SAFETY: `self.graph` was produced by `hipStreamEndCapture` and has
        // not been destroyed yet; `HipGraph`'s only constructor guarantees
        // it is non-null, and `Drop::drop` runs at most once.
        unsafe {
            let _ = bindings::hipGraphDestroy(self.graph);
        }
    }
}

// SAFETY: a `hipGraph_t` is a process-wide driver object describing a
// computation graph; it is not bound to any stream or thread, so moving it
// across threads is sound.
unsafe impl Send for HipGraph {}
// Deliberately not `Sync`: unlike `hipGraphLaunch` (documented safe for
// concurrent submission from multiple threads, see `HipGraphExec`'s `Sync`
// impl below), there is no documented guarantee that `hipGraphInstantiate`
// or `hipGraphGetNodes` are safe to call concurrently on the same
// `hipGraph_t` from multiple host threads. No caller needs this yet; add it
// only once that guarantee is verified against ROCm's own docs/source.

/// An instantiated HIP graph executable, ready for replay.
///
/// Owns a `hipGraphExec_t` handle; `Drop` calls `hipGraphExecDestroy`. Not
/// `Clone`, not `Copy`.
pub struct HipGraphExec {
    exec: bindings::hipGraphExec_t,
}

impl HipGraphExec {
    /// Upload this executable graph's resources to `stream` ahead of the
    /// first [`Self::launch`].
    ///
    /// Pre-stages kernel metadata and parameters on the device so the first
    /// replay does not pay that setup cost inline. `launch` uploads
    /// implicitly on first use if this is never called, so this is purely a
    /// way to separate "prepare" from "replay" (e.g. warming up a freshly
    /// instantiated graph before it serves real traffic).
    ///
    /// # Errors
    ///
    /// Returns an error if `hipGraphUpload` fails.
    pub fn upload(&self, stream: &Stream) -> Result<(), HipError> {
        // SAFETY: `self.exec` is a valid executable graph handle from a
        // successful `hipGraphInstantiate`, and `stream` is a live stream
        // borrowed for the duration of this call.
        hip_check(unsafe { bindings::hipGraphUpload(self.exec, stream.as_raw()) })
    }

    /// Launch (replay) this executable graph on `stream`.
    ///
    /// All kernels recorded during capture are submitted as a single unit,
    /// eliminating per-kernel launch overhead. The launch is asynchronous:
    /// it returns as soon as the work is enqueued, not when it completes.
    ///
    /// # Errors
    ///
    /// Returns an error if `hipGraphLaunch` fails.
    pub fn launch(&self, stream: &Stream) -> Result<(), HipError> {
        // SAFETY: `self.exec` is a valid executable graph handle from a
        // successful `hipGraphInstantiate`, and `stream` is a live stream
        // borrowed for the duration of this call.
        hip_check(unsafe { bindings::hipGraphLaunch(self.exec, stream.as_raw()) })
    }

    /// Update this executable graph's parameters from `graph`.
    ///
    /// If `graph` has the same topology as the one this executable was
    /// instantiated from (same nodes in the same order, only kernel
    /// arguments or memory addresses changed), the update patches the
    /// executable in place. This is cheaper than re-instantiating. If the
    /// topology changed, the returned [`GraphExecUpdateResult`] reports
    /// that, and the caller must re-instantiate via [`HipGraph::instantiate`]
    /// instead.
    ///
    /// This cannot overlap with a concurrent [`Self::launch`] or another
    /// `update` call on the same executable from the host side. `&mut self`
    /// prevents that overlap at compile time. The caller must also ensure
    /// that any prior [`Self::launch`] on the same stream has completed
    /// (e.g. via stream synchronization) before calling `update`, since HIP
    /// does not allow updating an executable while its GPU work is in
    /// flight.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `hipGraphExecUpdate` call fails for
    /// a reason other than a topology change. A topology change is reported
    /// through the returned [`GraphExecUpdateResult`] instead, even though
    /// the HIP call itself returns a non-success status for it too.
    pub fn update(&mut self, graph: &HipGraph) -> Result<GraphExecUpdateResult, HipError> {
        // `error_node` is intentionally discarded: it is only valid until
        // the next graph mutation, and surfacing it would mean exposing
        // `hipGraphNode_t` in this module's public API. On a topology
        // change the caller must re-capture and re-instantiate regardless
        // of which node triggered it, so the extra diagnostic has no
        // actionable use here.
        let mut error_node: bindings::hipGraphNode_t = std::ptr::null_mut();
        let mut result: bindings::hipGraphExecUpdateResult = 0;
        // SAFETY: `self.exec` and `graph.graph` are valid handles; the
        // error-node and result out-pointers are valid for a single write.
        let status = unsafe {
            bindings::hipGraphExecUpdate(
                self.exec,
                graph.graph,
                &raw mut error_node,
                &raw mut result,
            )
        };
        // A topology change is reported two ways at once: `result` is set to
        // `hipGraphExecUpdateErrorTopologyChanged`, and the call itself
        // returns `hipErrorGraphExecUpdateFailure` rather than `hipSuccess`.
        // Treat that one status as informational, same as success, and let
        // `result` carry the detail; any other non-success status is a
        // genuine failure with no usable `result`.
        if status != bindings::hipError_t_hipSuccess
            && status != bindings::hipError_t_hipErrorGraphExecUpdateFailure
        {
            return Err(HipError::new(status));
        }
        Ok(GraphExecUpdateResult::from_raw(result))
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        // SAFETY: `self.exec` was produced by `hipGraphInstantiate` and has
        // not been destroyed yet; `HipGraphExec`'s only constructor
        // guarantees it is non-null, and `Drop::drop` runs at most once.
        unsafe {
            let _ = bindings::hipGraphExecDestroy(self.exec);
        }
    }
}

// SAFETY: a `hipGraphExec_t` is a process-wide driver object with no thread
// affinity. `launch` takes the target stream as an explicit argument, so
// the executable itself is not bound to any particular stream or thread.
unsafe impl Send for HipGraphExec {}
// SAFETY: `launch` takes `&self` and only submits work to the caller-given
// stream; HIP serializes submissions to one stream internally, so
// concurrent `launch` calls from different threads are sound. `update`
// takes `&mut self`. The borrow checker enforces that it cannot overlap
// with `launch` or another `update`; this is not a runtime convention.
unsafe impl Sync for HipGraphExec {}

fn hip_check(status: bindings::hipError_t) -> Result<(), HipError> {
    if status == bindings::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(HipError::new(status))
    }
}
