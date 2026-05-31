// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`RestoreRunner`] — the shared bulk-load lifecycle owned by
//! both restore drivers (formal snapshot, perpetual store).
//!
//! The runner is intentionally narrow: it knows how to drive one
//! shard of objects through one pipeline, persist the partition's
//! progress in the `__restore` CF, and switch from
//! `InProgress` → `Complete` at the end. Parallelism strategy
//! (async tokio tasks fed by an object-store stream, sync
//! `thread::scope` over `ObjectID`-range shards, etc.) belongs to
//! the calling driver.
//!
//! # Lifecycle
//!
//! ```text
//! RestoreRunner::new(db, pipeline, schema, target_checkpoint)
//!     ↓
//! runner.begin()  →  Set { partitions already complete from a prior run }
//!     ↓
//! For each shard not in the skip set, in any order, possibly
//! in parallel:
//!     runner.process_shard(partition_id, objects)
//!         1. fold objects into Restore::Batch via Restore::restore
//!         2. drain into a Db::batch() via Restore::commit
//!         3. stage the partition-complete marker into the same batch
//!         4. Batch::commit() — atomic across pipeline data and marker
//!     ↓
//! runner.finish()  →  RestoreState::Complete { restored_at }
//! ```
//!
//! # Atomicity and resumability
//!
//! Each shard's pipeline writes and its partition-complete marker
//! land in a single [`rocksdb::WriteBatch`] commit. A crash before
//! the commit leaves the marker unwritten *and* the data writes
//! unwritten, so resume re-runs the shard from scratch with no
//! double-application of merges or duplicate puts.
//!
//! [`begin`](RestoreRunner::begin) inspects the existing
//! [`RestoreState`] for the pipeline:
//!
//! - `None`: writes a fresh `InProgress` entry for
//!   `target_checkpoint`.
//! - `InProgress { target_checkpoint: T, … }` with a matching `T`:
//!   returns the previously-completed partitions; subsequent
//!   [`process_shard`](RestoreRunner::process_shard) calls skip
//!   them via the [`already_complete`](RestoreRunner::already_complete)
//!   check.
//! - `InProgress` with a different target: refuses to proceed.
//!   Mid-restore at a different checkpoint indicates the caller
//!   is mixing two restore runs.
//! - `Complete`: refuses to begin again. The pipeline is already
//!   restored; tip indexing should resume.
//!
//! # Option toggles are not the runner's job
//!
//! The runner does not call
//! [`Db::set_restore_options_cf`](crate::Db::set_restore_options_cf)
//! or [`Db::set_tip_options_cf`](crate::Db::set_tip_options_cf).
//! That sequencing lives in the driver because the driver knows
//! which CFs across which pipelines should be toggled together
//! (and when to begin tip indexing for any pipeline).

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;

use parking_lot::Mutex;
use sui_consistent_store::Db;
use sui_consistent_store::FrameworkSchema;
use sui_consistent_store::PipelineTaskKey;
use sui_consistent_store::Restore;
use sui_consistent_store::RestoreState;
use sui_consistent_store::error::Error;
use sui_consistent_store::restore_state;
use sui_types::object::Object;
use tracing::debug;
use tracing::info;

/// Drives one pipeline's restore from a stream of objects.
///
/// `RestoreRunner` is constructed once per pipeline and shared
/// across worker threads or tasks via [`Arc`]. Per-shard work is
/// done by [`process_shard`](Self::process_shard), which is
/// `&self` and internally synchronizes updates to the persisted
/// `__restore` state.
///
/// The pipeline's [`Schema`](crate::Restore::Schema) is shared
/// across workers as `&Self::Schema`; the pipeline itself is wrapped
/// in [`Arc<P>`] so `&self` calls can spread across threads.
pub struct RestoreRunner<P: Restore> {
    db: Db,
    /// Cached owned [`FrameworkSchema`] for typed access to the
    /// `__restore` CF. Avoids the per-call construction cost that
    /// `db.framework()` would pay for staged writes.
    framework: FrameworkSchema,
    pipeline: Arc<P>,
    schema: Arc<P::Schema>,
    target_checkpoint: u64,
    /// Serializes read-modify-write of the persisted `__restore`
    /// state across concurrent shard processors. Updates are
    /// once-per-shard and small, so the contention is minimal.
    state_lock: Mutex<()>,
}

impl<P: Restore> RestoreRunner<P> {
    /// Create a new runner.
    pub fn new(
        db: Db,
        pipeline: Arc<P>,
        schema: Arc<P::Schema>,
        target_checkpoint: u64,
    ) -> Self {
        let framework = FrameworkSchema::new(db.clone());
        Self {
            db,
            framework,
            pipeline,
            schema,
            target_checkpoint,
            state_lock: Mutex::new(()),
        }
    }

    /// Read the current persisted [`RestoreState`] for this
    /// runner's pipeline.
    fn read_state(&self) -> Result<Option<RestoreState>, Error> {
        self.framework.restore.get(&PipelineTaskKey::new(P::NAME))
    }

    /// Write `state` for this runner's pipeline (commit immediately).
    fn write_state(&self, state: &RestoreState) -> Result<(), Error> {
        let mut batch = self.db.batch();
        batch.put(&self.framework.restore, &PipelineTaskKey::new(P::NAME), state)?;
        batch.commit()
    }

    /// Initialize the runner's `__restore` entry and return the
    /// set of partition IDs that were already ingested in a prior
    /// run. Callers use the returned set to skip those partitions.
    ///
    /// # Errors
    ///
    /// - The pipeline is already
    ///   [`Complete`](RestoreState::Complete) — refuse to restart a
    ///   completed restore.
    /// - The pipeline is `InProgress` with a different
    ///   `target_checkpoint` — refuse to mix two restore runs.
    pub fn begin(&self) -> anyhow::Result<BTreeSet<Vec<u8>>> {
        let _guard = self.state_lock.lock();
        match self.read_state()?.and_then(|s| s.state) {
            None => {
                self.write_state(&RestoreState::default().with_in_progress(
                    restore_state::InProgress {
                        target_checkpoint: self.target_checkpoint,
                        partitions_complete: vec![],
                    },
                ))?;
                info!(
                    pipeline = P::NAME,
                    target_checkpoint = self.target_checkpoint,
                    "Beginning restore",
                );
                Ok(BTreeSet::new())
            }
            Some(restore_state::State::InProgress(in_progress)) => {
                anyhow::ensure!(
                    in_progress.target_checkpoint == self.target_checkpoint,
                    "pipeline {} is mid-restore at checkpoint {}, but this runner targets {}",
                    P::NAME,
                    in_progress.target_checkpoint,
                    self.target_checkpoint,
                );
                let partitions_complete: BTreeSet<Vec<u8>> = in_progress
                    .partitions_complete
                    .iter()
                    .map(|p| p.as_ref().to_vec())
                    .collect();
                info!(
                    pipeline = P::NAME,
                    target_checkpoint = in_progress.target_checkpoint,
                    skip_count = partitions_complete.len(),
                    "Resuming restore",
                );
                Ok(partitions_complete)
            }
            Some(restore_state::State::Complete(complete)) => {
                anyhow::bail!(
                    "pipeline {} is already restored at checkpoint {}; refusing to restart",
                    P::NAME,
                    complete.restored_at,
                );
            }
        }
    }

    /// Returns `true` if `partition_id` was already ingested in a
    /// prior run.
    ///
    /// Equivalent to checking the set returned by
    /// [`begin`](Self::begin), but reads the persisted state on
    /// each call. Useful when the driver does not cache the
    /// returned set, or when shard processing happens concurrently
    /// with [`begin`](Self::begin) returning.
    pub fn already_complete(&self, partition_id: &[u8]) -> anyhow::Result<bool> {
        let _guard = self.state_lock.lock();
        match self.read_state()?.and_then(|s| s.state) {
            Some(restore_state::State::InProgress(in_progress)) => Ok(in_progress
                .partitions_complete
                .iter()
                .any(|p: &bytes::Bytes| p.as_ref() == partition_id)),
            Some(restore_state::State::Complete(_)) => Ok(true),
            None => Ok(false),
        }
    }

    /// Process one shard: fold its objects into the accumulator,
    /// drain into a typed [`Batch`](sui_consistent_store::Batch), stage the
    /// partition-complete marker into the same batch, and commit
    /// atomically.
    ///
    /// `partition_id` is the opaque driver-supplied identifier
    /// that distinguishes shards in the persisted progress set.
    ///
    /// `objects` is an iterator of fallible objects. Iterator
    /// errors abort the shard before the batch is committed, so
    /// partial state never lands in the database.
    pub fn process_shard<I>(&self, partition_id: &[u8], objects: I) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = anyhow::Result<Object>>,
    {
        if self.already_complete(partition_id)? {
            debug!(
                pipeline = P::NAME,
                partition = %hex_encode(partition_id),
                "Skipping already-complete partition",
            );
            return Ok(());
        }

        // 1. Stage per-object writes directly into the shard's
        // batch. Per-object errors abort the shard before the
        // batch is committed.
        let mut batch = self.db.batch();
        let mut object_count = 0usize;
        for object in objects {
            let object = object?;
            self.pipeline.restore(&self.schema, &object, &mut batch)?;
            object_count += 1;
        }

        // 2. Stage the partition-complete marker into the same
        // batch, then commit. The state-lock guard serializes the
        // read-modify-write of `partitions_complete` against
        // concurrent shard processors, and the WriteBatch's
        // atomicity ensures the marker and data writes either both
        // land or neither does.
        {
            let _guard = self.state_lock.lock();
            let next_state = self.next_partition_state(partition_id)?;
            batch.put(
                &self.framework.restore,
                &PipelineTaskKey::new(P::NAME),
                &next_state,
            )?;
            batch.commit()?;
        }

        debug!(
            pipeline = P::NAME,
            partition = %hex_encode(partition_id),
            objects = object_count,
            "Shard complete",
        );

        Ok(())
    }

    /// Mark the restore complete and switch
    /// `RestoreState::InProgress` → `RestoreState::Complete`.
    ///
    /// The caller is responsible for confirming every shard the
    /// driver intends to run has succeeded before calling this —
    /// the runner does not enumerate "all partitions" because the
    /// driver owns that list.
    pub fn finish(&self) -> anyhow::Result<()> {
        let _guard = self.state_lock.lock();
        self.write_state(
            &RestoreState::default().with_complete(restore_state::Complete {
                restored_at: self.target_checkpoint,
            }),
        )?;
        info!(
            pipeline = P::NAME,
            restored_at = self.target_checkpoint,
            "Restore complete",
        );
        Ok(())
    }

    /// Compute the next persisted [`RestoreState`] for this
    /// pipeline given that `partition_id` is about to be marked
    /// complete.
    ///
    /// Callers stage the returned state into the same
    /// [`Batch`](sui_consistent_store::Batch) that holds the shard's pipeline
    /// writes, so the marker lands atomically with the writes.
    /// Must be invoked under [`state_lock`](Self::state_lock) so
    /// concurrent shard processors do not race on the
    /// read-modify-write of `partitions_complete`.
    fn next_partition_state(&self, partition_id: &[u8]) -> Result<RestoreState, Error> {
        let mut in_progress = match self.read_state()?.and_then(|s| s.state) {
            Some(restore_state::State::InProgress(in_progress)) => {
                debug_assert_eq!(
                    in_progress.target_checkpoint, self.target_checkpoint,
                    "partition update after target_checkpoint mismatch — \
                     should have been caught at begin()",
                );
                in_progress
            }
            // Mid-restore should always be InProgress; if we see
            // something else, the state was changed under us, which
            // is a programmer error in the driver.
            _ => {
                return Err(Error::Internal(
                    "restore state changed unexpectedly during shard processing",
                ));
            }
        };
        in_progress
            .partitions_complete
            .push(bytes::Bytes::copy_from_slice(partition_id));
        Ok(RestoreState::default().with_in_progress(in_progress))
    }
}

/// Hex-encode a byte slice for use in human-readable identifiers
/// (log fields). The crate does not otherwise depend on `hex`, and
/// the implementation is small.
fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Buf;
    use bytes::BufMut;
    use sui_indexer_alt_framework::pipeline::Processor;
    use sui_indexer_alt_framework::types::full_checkpoint_content::Checkpoint;
    use sui_types::base_types::ObjectID;
    use sui_types::object::Object;
    use tempfile::TempDir;

    use sui_consistent_store::Batch;
    use sui_consistent_store::CfDescriptor;
    use sui_consistent_store::DbMap;
    use sui_consistent_store::DbOptions;
    use sui_consistent_store::Decode;
    use sui_consistent_store::Encode;
    use sui_consistent_store::Schema;
    use sui_consistent_store::error::DecodeError;
    use sui_consistent_store::error::EncodeError;
    use sui_consistent_store::error::OpenError;
    use sui_consistent_store::rocksdb;

    use super::*;

    /// Build an `InProgress` restore state with the supplied
    /// partition ids. Useful in test helpers that previously
    /// constructed `RestoreState::InProgress { ... }` directly.
    fn in_progress(
        target_checkpoint: u64,
        partitions: impl IntoIterator<Item = Vec<u8>>,
    ) -> RestoreState {
        RestoreState::default().with_in_progress(restore_state::InProgress {
            target_checkpoint,
            partitions_complete: partitions.into_iter().map(bytes::Bytes::from).collect(),
        })
    }

    /// Build a `Complete` restore state.
    fn complete(restored_at: u64) -> RestoreState {
        RestoreState::default().with_complete(restore_state::Complete { restored_at })
    }

    /// Extract the `InProgress` payload (panics otherwise). Mirrors
    /// the old `match Some(RestoreState::InProgress { .. })` pattern
    /// that tests used to write directly.
    fn expect_in_progress(state: Option<RestoreState>) -> restore_state::InProgress {
        match state.and_then(|s| s.state) {
            Some(restore_state::State::InProgress(in_progress)) => in_progress,
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    /// Extract the `Complete` payload (panics otherwise).
    fn expect_complete(state: Option<RestoreState>) -> restore_state::Complete {
        match state.and_then(|s| s.state) {
            Some(restore_state::State::Complete(complete)) => complete,
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    /// True if `partitions` contains a byte sequence equal to `id`.
    fn partitions_contain(partitions: &[bytes::Bytes], id: &[u8]) -> bool {
        partitions.iter().any(|p| p.as_ref() == id)
    }

    /// Test helper: read the persisted `RestoreState` for
    /// `pipeline`. Mirrors the production
    /// `runner.read_state()` but takes the pipeline name as an
    /// argument so tests can poke at arbitrary entries.
    fn read_restore_state(db: &Db, pipeline: &str) -> Option<RestoreState> {
        db.framework()
            .restore
            .get(&PipelineTaskKey::new(pipeline))
            .unwrap()
    }

    /// Test helper: write `state` for `pipeline` (commit
    /// immediately).
    fn write_restore_state(db: &Db, pipeline: &str, state: &RestoreState) {
        let fw = FrameworkSchema::new(db.clone());
        let mut batch = db.batch();
        batch
            .put(&fw.restore, &PipelineTaskKey::new(pipeline), state)
            .unwrap();
        batch.commit().unwrap();
    }

    /// Big-endian `ObjectID` key newtype.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ObjectIdKey([u8; ObjectID::LENGTH]);

    impl ObjectIdKey {
        fn new(id: ObjectID) -> Self {
            Self(id.into_bytes())
        }
    }

    impl Encode for ObjectIdKey {
        fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
            buf.put_slice(&self.0);
            Ok(())
        }
    }

    impl Decode for ObjectIdKey {
        fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
            if buf.remaining() != ObjectID::LENGTH {
                return Err(DecodeError::msg("unexpected ObjectIdKey length"));
            }
            let mut id = [0u8; ObjectID::LENGTH];
            buf.copy_to_slice(&mut id);
            Ok(Self(id))
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct U64Be(u64);

    impl Encode for U64Be {
        fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
            buf.put_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    impl Decode for U64Be {
        fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
            if buf.remaining() != 8 {
                return Err(DecodeError::msg("expected 8 bytes"));
            }
            Ok(Self(buf.get_u64()))
        }
    }

    /// Schema with one CF mapping object id → version.
    #[derive(Debug)]
    struct VersionsSchema {
        versions: DbMap<ObjectIdKey, U64Be>,
    }

    impl Schema for VersionsSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
            vec![CfDescriptor::new("versions", base_options.clone())]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

    /// Test pipeline: writes (object_id → version) per object.
    struct VersionsPipeline;

    #[async_trait]
    impl Processor for VersionsPipeline {
        const NAME: &'static str = "versions";
        type Value = ();

        async fn process(&self, _: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
            Ok(vec![])
        }
    }

    impl Restore for VersionsPipeline {
        type Schema = VersionsSchema;

        fn restore(
            &self,
            schema: &Self::Schema,
            object: &Object,
            batch: &mut Batch,
        ) -> anyhow::Result<()> {
            batch.put(
                &schema.versions,
                &ObjectIdKey::new(object.id()),
                &U64Be(object.version().value()),
            )?;
            Ok(())
        }
    }

    /// Schema with one merge-operator CF.
    #[derive(Debug)]
    struct CountersSchema {
        counters: DbMap<ObjectIdKey, U64Be>,
    }

    fn add_u64_merge_op(
        _key: &[u8],
        existing: Option<&[u8]>,
        operands: &rocksdb::MergeOperands,
    ) -> Option<Vec<u8>> {
        let mut total: u64 = existing
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .map(u64::from_be_bytes)
            .unwrap_or(0);
        for op in operands {
            if let Ok(arr) = <[u8; 8]>::try_from(op) {
                total = total.saturating_add(u64::from_be_bytes(arr));
            }
        }
        Some(total.to_be_bytes().to_vec())
    }

    impl Schema for CountersSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
            let mut opts = base_options.clone();
            opts.set_merge_operator_associative("u64-add", add_u64_merge_op);
            vec![CfDescriptor::new("counters", opts)]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                counters: DbMap::new(db.clone(), "counters")?,
            })
        }
    }

    /// Test pipeline: counts how many times each id was observed.
    /// Uses a merge operator so per-object emits (one merge per
    /// object) combine within and across shards through RocksDB's
    /// registered merge operator.
    struct CountersPipeline;

    #[async_trait]
    impl Processor for CountersPipeline {
        const NAME: &'static str = "counters";
        type Value = ();

        async fn process(&self, _: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
            Ok(vec![])
        }
    }

    impl Restore for CountersPipeline {
        type Schema = CountersSchema;

        fn restore(
            &self,
            schema: &Self::Schema,
            object: &Object,
            batch: &mut Batch,
        ) -> anyhow::Result<()> {
            batch.merge(&schema.counters, &ObjectIdKey::new(object.id()), &U64Be(1))?;
            Ok(())
        }
    }

    /// Build a runner against a fresh database. Returns the runner
    /// plus the lifetime-extending temp dir.
    fn setup_versions(
        target_checkpoint: u64,
    ) -> (
        TempDir,
        Db,
        Arc<VersionsSchema>,
        RestoreRunner<VersionsPipeline>,
    ) {
        let db_dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<VersionsSchema>(db_dir.path(), DbOptions::default()).unwrap();
        let schema = Arc::new(schema);
        let runner = RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            target_checkpoint,
        );
        (db_dir, db, schema, runner)
    }

    fn obj(id: u8) -> Object {
        Object::immutable_with_id_for_testing(ObjectID::from_single_byte(id))
    }

    #[test]
    fn begin_initializes_state_when_none() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        let skip = runner.begin().unwrap();
        assert!(skip.is_empty());
        let in_progress = expect_in_progress(read_restore_state(&db, "versions"));
        assert_eq!(in_progress.target_checkpoint, 100);
        assert!(in_progress.partitions_complete.is_empty());
    }

    #[test]
    fn begin_returns_completed_partitions_from_prior_run() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        // Simulate a prior partial run.
        let prior_ids = [vec![0xAAu8], vec![0xBBu8]];
        write_restore_state(&db, "versions", &in_progress(100, prior_ids.clone()));

        let skip = runner.begin().unwrap();
        let expected: BTreeSet<Vec<u8>> = prior_ids.into_iter().collect();
        assert_eq!(skip, expected);
    }

    #[test]
    fn begin_refuses_target_mismatch() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        write_restore_state(&db, "versions", &in_progress(200, []));
        let err = runner.begin().unwrap_err();
        assert!(format!("{err}").contains("mid-restore at checkpoint 200"));
    }

    #[test]
    fn begin_refuses_already_complete() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        write_restore_state(&db, "versions", &complete(50));
        let err = runner.begin().unwrap_err();
        assert!(format!("{err}").contains("already restored"));
    }

    #[test]
    fn process_shard_writes_data_and_marks_partition_complete() {
        let (_db_dir, db, schema, runner) = setup_versions(100);
        runner.begin().unwrap();

        let o1 = obj(1);
        let o2 = obj(2);
        let (v1, v2) = (o1.version().value(), o2.version().value());
        runner
            .process_shard(b"shard-0", vec![Ok(o1), Ok(o2)])
            .unwrap();

        // Data is visible.
        assert_eq!(
            schema
                .versions
                .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                .unwrap(),
            Some(U64Be(v1)),
        );
        assert_eq!(
            schema
                .versions
                .get(&ObjectIdKey::new(ObjectID::from_single_byte(2)))
                .unwrap(),
            Some(U64Be(v2)),
        );

        // Partition is recorded.
        let in_progress = expect_in_progress(read_restore_state(&db, "versions"));
        assert!(partitions_contain(&in_progress.partitions_complete, b"shard-0"));
    }

    #[test]
    fn process_shard_skips_already_complete_partitions() {
        let (_db_dir, db, schema, runner) = setup_versions(100);
        runner.begin().unwrap();

        // Pre-mark shard-0 as complete.
        write_restore_state(&db, "versions", &in_progress(100, [b"shard-0".to_vec()]));

        // Running shard-0 again must not write anything new.
        runner.process_shard(b"shard-0", vec![Ok(obj(1))]).unwrap();
        assert!(
            schema
                .versions
                .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                .unwrap()
                .is_none(),
            "skipped shard must not write",
        );
    }

    #[test]
    fn process_shard_with_empty_object_stream_still_marks_partition_complete() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();
        runner
            .process_shard(b"empty", Vec::<anyhow::Result<Object>>::new())
            .unwrap();
        let in_progress = expect_in_progress(read_restore_state(&db, "versions"));
        assert!(partitions_contain(&in_progress.partitions_complete, b"empty"));
    }

    #[test]
    fn process_shard_propagates_object_stream_error_and_does_not_mark_complete() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();

        let stream: Vec<anyhow::Result<Object>> = vec![
            Ok(obj(1)),
            Err(anyhow::anyhow!("source-side IO failed")),
            Ok(obj(2)),
        ];
        let err = runner.process_shard(b"bad", stream).unwrap_err();
        assert!(format!("{err}").contains("source-side IO failed"));

        let in_progress = expect_in_progress(read_restore_state(&db, "versions"));
        assert!(!partitions_contain(&in_progress.partitions_complete, b"bad"));
    }

    #[test]
    fn cross_shard_merges_combine_through_runner() {
        // End-to-end exercise of the design's restore story using
        // the merge-operator pipeline. Two shards each touch the
        // same object id; the runner emits one merge per shard's
        // commit, and the registered operator combines them on
        // read.
        let db_dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<CountersSchema>(db_dir.path(), DbOptions::default()).unwrap();
        let schema = Arc::new(schema);
        let runner = RestoreRunner::new(
            db.clone(),
            Arc::new(CountersPipeline),
            schema.clone(),
            42,
        );

        runner.begin().unwrap();

        // Shard A: id #1 appears 3 times in this shard. Folded to
        // one merge(+=3) by the accumulator.
        runner
            .process_shard(b"A", vec![Ok(obj(1)), Ok(obj(1)), Ok(obj(1))])
            .unwrap();
        // Shard B: id #1 appears 2 times here, id #2 once.
        runner
            .process_shard(b"B", vec![Ok(obj(1)), Ok(obj(1)), Ok(obj(2))])
            .unwrap();

        runner.finish().unwrap();

        // id #1: 3 + 2 = 5 via the cross-shard merge.
        assert_eq!(
            schema
                .counters
                .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                .unwrap(),
            Some(U64Be(5)),
        );
        // id #2: 1.
        assert_eq!(
            schema
                .counters
                .get(&ObjectIdKey::new(ObjectID::from_single_byte(2)))
                .unwrap(),
            Some(U64Be(1)),
        );

        // Restore is marked complete.
        let complete = expect_complete(read_restore_state(&db, "counters"));
        assert_eq!(complete.restored_at, 42);
    }

    #[test]
    fn finish_transitions_to_complete() {
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();
        runner.process_shard(b"only", vec![Ok(obj(1))]).unwrap();
        runner.finish().unwrap();
        let complete = expect_complete(read_restore_state(&db, "versions"));
        assert_eq!(complete.restored_at, 100);
    }

    #[test]
    fn already_complete_returns_true_after_finish() {
        let (_db_dir, _db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();
        runner.finish().unwrap();
        assert!(runner.already_complete(b"anything").unwrap());
    }

    #[test]
    fn concurrent_shard_processing_serializes_state_updates() {
        // Spawn many threads each processing one shard. The mutex on
        // RestoreState updates must serialize so every shard's
        // partition id ends up recorded.
        use std::thread;
        let (_db_dir, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();

        let runner = Arc::new(runner);
        let mut handles = Vec::new();
        for i in 0..16u8 {
            let runner = runner.clone();
            handles.push(thread::spawn(move || {
                let partition = vec![i];
                runner.process_shard(&partition, vec![Ok(obj(i))]).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let in_progress = expect_in_progress(read_restore_state(&db, "versions"));
        assert_eq!(in_progress.partitions_complete.len(), 16);
        for i in 0..16u8 {
            assert!(partitions_contain(&in_progress.partitions_complete, &[i]));
        }
    }

    #[test]
    fn hex_encode_roundtrips_byte_values() {
        assert_eq!(super::hex_encode(&[]), "");
        assert_eq!(super::hex_encode(&[0]), "00");
        assert_eq!(super::hex_encode(&[0xAB, 0xCD]), "abcd");
        assert_eq!(super::hex_encode(&[0xFF; 4]), "ffffffff");
    }
}
