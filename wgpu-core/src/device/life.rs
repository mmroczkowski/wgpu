use alloc::{sync::Arc, vec::Vec};

use smallvec::SmallVec;
use thiserror::Error;

use crate::{
    device::{
        queue::{EncoderInFlight, SubmittedWorkDoneClosure, TempResource},
        DeviceError,
    },
    ray_tracing::BlasCompactReadyPendingClosure,
    resource::{Blas, Buffer, QuerySet, Texture, Trackable},
    snatch::SnatchGuard,
    SubmissionIndex,
};

/// A command submitted to the GPU for execution.
///
/// ## Keeping resources alive while the GPU is using them
///
/// [`wgpu_hal`] requires that, when a command is submitted to a queue, all the
/// resources it uses must remain alive until it has finished executing.
///
/// [`wgpu_hal`]: hal
struct ActiveSubmission {
    /// The index of the submission we track.
    ///
    /// When `Device::fence`'s value is greater than or equal to this, our queue
    /// submission has completed.
    index: SubmissionIndex,

    /// Buffers to be mapped once this submission has completed.
    mapped: Vec<Arc<Buffer>>,

    /// BLASes to have their compacted size read back once this submission has completed.
    compact_read_back: Vec<Arc<Blas>>,

    /// Command buffers used by this submission, and the encoder that owns them.
    ///
    /// [`wgpu_hal::Queue::submit`] requires the submitted command buffers to
    /// remain alive until the submission has completed execution. Command
    /// encoders double as allocation pools for command buffers, so holding them
    /// here and cleaning them up in [`LifetimeTracker::triage_submissions`]
    /// satisfies that requirement.
    ///
    /// Once this submission has completed, the command buffers are reset and
    /// the command encoder is recycled.
    ///
    /// [`wgpu_hal::Queue::submit`]: hal::Queue::submit
    encoders: Vec<EncoderInFlight>,

    /// List of queue "on_submitted_work_done" closures to be called once this
    /// submission has completed.
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,
}

impl ActiveSubmission {
    /// Returns true if this submission contains the given buffer.
    ///
    /// This only uses constant-time operations.
    pub fn contains_buffer(&self, buffer: &Buffer) -> bool {
        for encoder in &self.encoders {
            // The ownership location of buffers depends on where the command encoder
            // came from. If it is the staging command encoder on the queue, it is
            // in the pending buffer list. If it came from a user command encoder,
            // it is in the tracker.

            if encoder.trackers.buffers.contains(buffer) {
                return true;
            }

            if encoder
                .pending_buffers
                .contains_key(&buffer.tracker_index())
            {
                return true;
            }
        }

        false
    }

    /// Returns true if this submission contains the given texture.
    ///
    /// This only uses constant-time operations.
    pub fn contains_texture(&self, texture: &Texture) -> bool {
        for encoder in &self.encoders {
            // The ownership location of textures depends on where the command encoder
            // came from. If it is the staging command encoder on the queue, it is
            // in the pending buffer list. If it came from a user command encoder,
            // it is in the tracker.

            if encoder.trackers.textures.contains(texture) {
                return true;
            }

            if encoder
                .pending_textures
                .contains_key(&texture.tracker_index())
            {
                return true;
            }
        }

        false
    }

    /// Returns true if this submission contains the given blas.
    ///
    /// This only uses constant-time operations.
    pub fn contains_blas(&self, blas: &Blas) -> bool {
        for encoder in &self.encoders {
            if encoder.trackers.blas_s.contains(blas) {
                return true;
            }

            if encoder.pending_blas_s.contains_key(&blas.tracker_index()) {
                return true;
            }
        }

        false
    }

    pub fn contains_query_set(&self, query_set: &QuerySet) -> bool {
        for encoder in &self.encoders {
            if encoder.trackers.query_sets.contains(query_set) {
                return true;
            }
        }

        false
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum WaitIdleError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Tried to wait using a submission index ({0}) that has not been returned by a successful submission (last successful submission: {1})")]
    WrongSubmissionIndex(SubmissionIndex, SubmissionIndex),
    #[error("Timed out trying to wait for the given submission index.")]
    Timeout,
}

impl WaitIdleError {
    pub fn to_poll_error(&self) -> Option<wgt::PollError> {
        match self {
            WaitIdleError::Timeout => Some(wgt::PollError::Timeout),
            &WaitIdleError::WrongSubmissionIndex(a, b) => {
                Some(wgt::PollError::WrongSubmissionIndex(a, b))
            }
            _ => None,
        }
    }
}

/// Resource tracking for a device.
///
/// ## Host mapping buffers
///
/// A buffer cannot be mapped until all active queue submissions that use it
/// have completed. To that end:
///
/// -   Each buffer's `ResourceInfo::submission_index` records the index of the
///     most recent queue submission that uses that buffer.
///
/// -   When the device is polled, the following `LifetimeTracker` methods decide
///     what should happen next:
///
///     1)  `triage_submissions` moves entries in `self.active[i]` for completed
///         submissions to `self.ready_to_map`.  At this point, both
///         `self.active` and `self.ready_to_map` are up to date with the given
///         submission index.
///
///     2)  `handle_mapping` drains `self.ready_to_map` and actually maps the
///         buffers, collecting a list of notification closures to call.
///
/// Only calling `Global::buffer_map_async` clones a new `Arc` for the
/// buffer. This new `Arc` is only dropped by `handle_mapping`.
pub(crate) struct LifetimeTracker {
    /// Resources used by queue submissions still in flight. One entry per
    /// submission, with older submissions appearing before younger.
    ///
    /// Entries are added by `track_submission` and drained by
    /// `LifetimeTracker::triage_submissions`. Lots of methods contribute data
    /// to particular entries.
    active: Vec<ActiveSubmission>,

    /// Buffers the user has asked us to map, and which are not used by any
    /// queue submission still in flight.
    ready_to_map: Vec<Arc<Buffer>>,

    /// BLASes the user has asked us to prepare to compact, and which are not used by any
    /// queue submission still in flight.
    ready_to_compact: Vec<Arc<Blas>>,

    /// Queue "on_submitted_work_done" closures that were initiated for while there is no
    /// currently pending submissions. These cannot be immediately invoked as they
    /// must happen _after_ all mapped buffer callbacks are mapped, so we defer them
    /// here until the next time the device is maintained.
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,

    /// The thread releasing the encoders of finished submissions, started by
    /// the first triage that may leave their release to it.
    encoder_releaser: Option<EncoderReleaser>,
}

/// When [`LifetimeTracker::triage_submissions`] releases the encoders of the
/// submissions it finds finished, together with the resources they kept alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EncoderRelease {
    /// Before the triage returns, once every encoder left to the background
    /// earlier is released too. A caller that goes on to depend on the
    /// resources being free, as configuring a surface does, asks for this.
    AtOnce,

    /// On another thread where one can be started, as releasing the encoders
    /// of a frame costs the triaging thread a large part of a submission.
    InBackground,
}

impl LifetimeTracker {
    pub fn new() -> Self {
        Self {
            active: Vec::new(),
            ready_to_map: Vec::new(),
            ready_to_compact: Vec::new(),
            work_done_closures: SmallVec::new(),
            encoder_releaser: None,
        }
    }

    /// Return true if there are no queue submissions still in flight.
    pub fn queue_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Start tracking resources associated with a new queue submission.
    pub fn track_submission(&mut self, index: SubmissionIndex, encoders: Vec<EncoderInFlight>) {
        self.active.push(ActiveSubmission {
            index,
            mapped: Vec::new(),
            compact_read_back: Vec::new(),
            encoders,
            work_done_closures: SmallVec::new(),
        });
    }

    /// Schedule a buffer for mapping.
    ///
    /// The buffer will be added either to a pending submission, or to `self.ready_to_map`.
    /// If it is added to a pending submission, returns the index of that submission.
    pub(crate) fn map(&mut self, buffer: &Arc<Buffer>) -> Option<SubmissionIndex> {
        let submission = self
            .active
            .iter_mut()
            .rev()
            .find(|a| a.contains_buffer(buffer));

        let maybe_submission_index = submission.as_ref().map(|s| s.index);

        submission
            .map_or(&mut self.ready_to_map, |a| &mut a.mapped)
            .push(buffer.clone());

        maybe_submission_index
    }

    pub(crate) fn prepare_compact(&mut self, blas: &Arc<Blas>) -> Option<SubmissionIndex> {
        // Determine which BLASes are ready to map, and which must wait for the GPU.
        let submission = self.active.iter_mut().rev().find(|a| a.contains_blas(blas));

        let maybe_submission_index = submission.as_ref().map(|s| s.index);

        submission
            .map_or(&mut self.ready_to_compact, |a| &mut a.compact_read_back)
            .push(blas.clone());

        maybe_submission_index
    }

    /// Returns the submission index of the most recent submission that uses the
    /// given buffer.
    pub fn get_buffer_latest_submission_index(&self, buffer: &Buffer) -> Option<SubmissionIndex> {
        // We iterate in reverse order, so that we can bail out early as soon
        // as we find a hit.
        self.active.iter().rev().find_map(|submission| {
            if submission.contains_buffer(buffer) {
                Some(submission.index)
            } else {
                None
            }
        })
    }

    /// Returns the submission index of the most recent submission that uses the
    /// given texture.
    pub fn get_texture_latest_submission_index(
        &self,
        texture: &Texture,
    ) -> Option<SubmissionIndex> {
        // We iterate in reverse order, so that we can bail out early as soon
        // as we find a hit.
        self.active.iter().rev().find_map(|submission| {
            if submission.contains_texture(texture) {
                Some(submission.index)
            } else {
                None
            }
        })
    }

    /// Returns the submission index of the most recent submission that uses the
    /// given query set.
    pub fn get_query_set_latest_submission_index(
        &self,
        query_set: &QuerySet,
    ) -> Option<SubmissionIndex> {
        // We iterate in reverse order, so that we can bail out early as soon
        // as we find a hit.
        self.active.iter().rev().find_map(|submission| {
            if submission.contains_query_set(query_set) {
                Some(submission.index)
            } else {
                None
            }
        })
    }

    /// Sort out the consequences of completed submissions.
    ///
    /// Assume that all submissions up through `last_done` have completed.
    ///
    /// -   Buffers used by those submissions are now ready to map, if requested.
    ///     Add any buffers in the submission's [`mapped`] list to
    ///     [`self.ready_to_map`], where [`LifetimeTracker::handle_mapping`]
    ///     will find them.
    ///
    /// Return a list of [`SubmittedWorkDoneClosure`]s to run.
    ///
    /// [`mapped`]: ActiveSubmission::mapped
    /// [`self.ready_to_map`]: LifetimeTracker::ready_to_map
    /// [`SubmittedWorkDoneClosure`]: crate::device::queue::SubmittedWorkDoneClosure
    #[must_use]
    pub fn triage_submissions(
        &mut self,
        last_done: SubmissionIndex,
        encoder_release: EncoderRelease,
    ) -> SmallVec<[SubmittedWorkDoneClosure; 1]> {
        profiling::scope!("triage_submissions");

        debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self.active.partition_point(|a| a.index <= last_done);

        let mut work_done_closures: SmallVec<_> = self.work_done_closures.drain(..).collect();
        let mut finished_encoders = Vec::new();
        for a in self.active.drain(..done_count) {
            self.ready_to_map.extend(a.mapped);
            self.ready_to_compact.extend(a.compact_read_back);
            finished_encoders.extend(a.encoders);
            work_done_closures.extend(a.work_done_closures);
        }
        self.release_encoders(finished_encoders, encoder_release);
        work_done_closures
    }

    /// Release the encoders of finished submissions.
    ///
    /// This involves actually decrementing the ref count of all command buffer
    /// resources, so can be _very_ expensive.
    fn release_encoders(&mut self, encoders: Vec<EncoderInFlight>, release: EncoderRelease) {
        match release {
            EncoderRelease::InBackground => {
                if encoders.is_empty() {
                    return;
                }
                if self.encoder_releaser.is_none() {
                    self.encoder_releaser = EncoderReleaser::start();
                }
                match &self.encoder_releaser {
                    Some(releaser) => releaser.release(encoders),
                    None => drop(encoders),
                }
            }
            EncoderRelease::AtOnce => {
                if let Some(releaser) = &self.encoder_releaser {
                    releaser.wait_until_all_released();
                }
                profiling::scope!("drop command buffer trackers");
                drop(encoders);
            }
        }
    }

    pub fn schedule_resource_destruction(
        &mut self,
        temp_resource: TempResource,
        last_submit_index: SubmissionIndex,
    ) {
        let resources = self
            .active
            .iter_mut()
            .find(|a| a.index == last_submit_index)
            .map(|a| {
                // Because this resource's `last_submit_index` matches `a.index`,
                // we know that we must have done something with the resource,
                // so `a.encoders` should not be empty.
                &mut a.encoders.last_mut().unwrap().temp_resources
            });
        if let Some(resources) = resources {
            resources.push(temp_resource);
        }
    }

    pub fn add_work_done_closure(
        &mut self,
        closure: SubmittedWorkDoneClosure,
    ) -> Option<SubmissionIndex> {
        match self.active.last_mut() {
            Some(active) => {
                active.work_done_closures.push(closure);
                Some(active.index)
            }
            // We must defer the closure until all previously occurring map_async closures
            // have fired. This is required by the spec.
            None => {
                self.work_done_closures.push(closure);
                None
            }
        }
    }

    /// Map the buffers in `self.ready_to_map`.
    ///
    /// Return a list of mapping notifications to send.
    ///
    /// See the documentation for [`LifetimeTracker`] for details.
    #[must_use]
    pub(crate) fn handle_mapping(
        &mut self,
        snatch_guard: &SnatchGuard,
    ) -> Vec<super::BufferMapPendingClosure> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let mut pending_callbacks: Vec<super::BufferMapPendingClosure> =
            Vec::with_capacity(self.ready_to_map.len());

        for buffer in self.ready_to_map.drain(..) {
            match buffer.map(snatch_guard) {
                Some(cb) => pending_callbacks.push(cb),
                None => continue,
            }
        }
        pending_callbacks
    }
    /// Read back compact sizes from the BLASes in `self.ready_to_compact`.
    ///
    /// Return a list of mapping notifications to send.
    ///
    /// See the documentation for [`LifetimeTracker`] for details.
    #[must_use]
    pub(crate) fn handle_compact_read_back(&mut self) -> Vec<BlasCompactReadyPendingClosure> {
        if self.ready_to_compact.is_empty() {
            return Vec::new();
        }
        let mut pending_callbacks: Vec<BlasCompactReadyPendingClosure> =
            Vec::with_capacity(self.ready_to_compact.len());

        for blas in self.ready_to_compact.drain(..) {
            match blas.read_back_compact_size() {
                Some(cb) => pending_callbacks.push(cb),
                None => continue,
            }
        }
        pending_callbacks
    }
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
mod encoder_releaser {
    use alloc::{sync::Arc, vec::Vec};
    use std::sync::{mpsc, Condvar, Mutex, PoisonError};

    use crate::device::queue::EncoderInFlight;

    /// The batches of encoders handed over and not yet released.
    #[derive(Default)]
    struct UnreleasedBatches {
        count: Mutex<usize>,
        released: Condvar,
    }

    impl UnreleasedBatches {
        fn add_one(&self) {
            *self.count.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        }

        fn remove_one(&self) {
            *self.count.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
            self.released.notify_all();
        }
    }

    /// Counts a batch as released when dropped, so a release that panics
    /// leaves nobody waiting for it.
    struct ReleasedBatch<'a>(&'a UnreleasedBatches);

    impl Drop for ReleasedBatch<'_> {
        fn drop(&mut self) {
            self.0.remove_one();
        }
    }

    /// A thread that releases the encoders of finished submissions, which
    /// stops once its owner is dropped and every encoder handed over is released.
    pub(super) struct EncoderReleaser {
        encoders: mpsc::Sender<Vec<EncoderInFlight>>,
        unreleased: Arc<UnreleasedBatches>,
    }

    impl EncoderReleaser {
        /// Starts the thread, or returns `None` where the system gives none.
        pub(super) fn start() -> Option<Self> {
            let (encoders, handed_over) = mpsc::channel::<Vec<EncoderInFlight>>();
            let unreleased = Arc::new(UnreleasedBatches::default());
            let unreleased_by_thread = unreleased.clone();
            std::thread::Builder::new()
                .name("wgpu encoder release".into())
                .spawn(move || {
                    for encoders in handed_over {
                        let _released = ReleasedBatch(&unreleased_by_thread);
                        profiling::scope!("drop command buffer trackers");
                        drop(encoders);
                    }
                })
                .ok()?;
            Some(Self {
                encoders,
                unreleased,
            })
        }

        pub(super) fn release(&self, encoders: Vec<EncoderInFlight>) {
            self.unreleased.add_one();
            if let Err(mpsc::SendError(encoders)) = self.encoders.send(encoders) {
                let _released = ReleasedBatch(&self.unreleased);
                drop(encoders);
            }
        }

        pub(super) fn wait_until_all_released(&self) {
            let mut count = self
                .unreleased
                .count
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            while *count > 0 {
                count = self
                    .unreleased
                    .released
                    .wait(count)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
    }
}

#[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
mod encoder_releaser {
    use alloc::vec::Vec;

    use crate::device::queue::EncoderInFlight;

    /// Never started where there are no threads to release encoders on.
    pub(super) enum EncoderReleaser {}

    impl EncoderReleaser {
        pub(super) fn start() -> Option<Self> {
            None
        }

        pub(super) fn release(&self, _encoders: Vec<EncoderInFlight>) {
            match *self {}
        }

        pub(super) fn wait_until_all_released(&self) {
            match *self {}
        }
    }
}

use encoder_releaser::EncoderReleaser;
