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
//! RestoreRunner::new(db, pipeline, schema, target_checkpoint, …)
//!     ↓
//! runner.begin()  →  Set { partitions already complete from a prior run }
//!     ↓
//! For each shard not in the skip set, in any order, possibly
//! in parallel:
//!     runner.process_shard(partition_id, objects)
//!         1. fold objects into Pipeline::Batch via Pipeline::restore
//!         2. drain into a shard-backed Batch via Pipeline::commit
//!         3. finalize into a ShardFinalize (per-CF SSTs +
//!            merge-mode WriteBatch)
//!         4. stage the partition-complete marker into the WriteBatch
//!         5. ShardFinalize::commit() — ingest SSTs first, then
//!            commit the WriteBatch (atomic with the marker)
//!     ↓
//! runner.finish()  →  RestoreState::Complete { restored_at }
//! ```
//!
//! # Atomicity across the two write paths
//!
//! Step 5 ingests SSTs before committing the WriteBatch. A crash
//! between the two leaves the partition-complete marker unwritten,
//! so resume re-runs the shard from scratch:
//!
//! - Re-ingest of a freshly built SST against an existing one for
//!   the same key range is idempotent for puts (last-write-wins)
//!   and tombstones; the bottom-most level converges to the same
//!   observable state.
//! - The merge-mode WriteBatch never committed on the first run,
//!   so no merge operands landed — re-running the shard does not
//!   double-merge.
//!
//! # Resumability
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
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use sui_types::object::Object;
use tracing::debug;
use tracing::info;

use crate::Db;
use crate::Pipeline;
use crate::RestoreState;
use crate::error::Error;

/// Drives one pipeline's restore from a stream of objects.
///
/// `RestoreRunner` is constructed once per pipeline and shared
/// across worker threads or tasks via [`Arc`]. Per-shard work is
/// done by [`process_shard`](Self::process_shard), which is
/// `&self` and internally synchronizes updates to the persisted
/// `__restore` state.
///
/// The pipeline's [`Schema`](crate::Pipeline::Schema) is shared
/// across workers as `&Self::Schema`; the pipeline itself is wrapped
/// in [`Arc<P>`] so `&self` calls can spread across threads.
pub struct RestoreRunner<P: Pipeline> {
    db: Arc<Db>,
    pipeline: Arc<P>,
    schema: Arc<P::Schema>,
    target_checkpoint: u64,
    staging_dir: PathBuf,
    sst_options: rocksdb::Options,
    /// Serializes read-modify-write of the persisted `__restore`
    /// state across concurrent shard processors. Updates are
    /// once-per-shard and small, so the contention is minimal.
    state_lock: Mutex<()>,
}

impl<P: Pipeline> RestoreRunner<P> {
    /// Create a new runner.
    ///
    /// `staging_dir` should be a directory dedicated to this
    /// restore (typically created with [`tempfile::TempDir`] or
    /// otherwise unique per run). The runner creates per-shard
    /// subdirectories underneath it, and removes them on shard
    /// success. The parent directory is the caller's responsibility
    /// to clean up.
    ///
    /// `sst_options` is the [`rocksdb::Options`] used to construct
    /// per-shard [`SstWriter`](crate::SstWriter) instances. The
    /// default comparator (byte-wise) must match the target CFs;
    /// schemas using non-default comparators must supply matching
    /// options here.
    pub fn new(
        db: Arc<Db>,
        pipeline: Arc<P>,
        schema: Arc<P::Schema>,
        target_checkpoint: u64,
        staging_dir: PathBuf,
        sst_options: rocksdb::Options,
    ) -> Self {
        Self {
            db,
            pipeline,
            schema,
            target_checkpoint,
            staging_dir,
            sst_options,
            state_lock: Mutex::new(()),
        }
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
        match self.db.restore_state(P::NAME)? {
            None => {
                self.db.set_restore_state(
                    P::NAME,
                    &RestoreState::InProgress {
                        target_checkpoint: self.target_checkpoint,
                        partitions_complete: BTreeSet::new(),
                    },
                )?;
                info!(
                    pipeline = P::NAME,
                    target_checkpoint = self.target_checkpoint,
                    "Beginning restore",
                );
                Ok(BTreeSet::new())
            }
            Some(RestoreState::InProgress {
                target_checkpoint,
                partitions_complete,
            }) => {
                anyhow::ensure!(
                    target_checkpoint == self.target_checkpoint,
                    "pipeline {} is mid-restore at checkpoint {}, but this runner targets {}",
                    P::NAME,
                    target_checkpoint,
                    self.target_checkpoint,
                );
                info!(
                    pipeline = P::NAME,
                    target_checkpoint,
                    skip_count = partitions_complete.len(),
                    "Resuming restore",
                );
                Ok(partitions_complete)
            }
            Some(RestoreState::Complete { restored_at }) => {
                anyhow::bail!(
                    "pipeline {} is already restored at checkpoint {}; refusing to restart",
                    P::NAME,
                    restored_at,
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
        match self.db.restore_state(P::NAME)? {
            Some(RestoreState::InProgress {
                partitions_complete,
                ..
            }) => Ok(partitions_complete.contains(partition_id)),
            Some(RestoreState::Complete { .. }) => Ok(true),
            None => Ok(false),
        }
    }

    /// Process one shard: fold its objects into the accumulator,
    /// drain into a shard-backed batch, finalize into per-CF SSTs
    /// plus a merge-mode WriteBatch, and atomically commit (SSTs
    /// first, then WriteBatch carrying the partition-complete
    /// marker).
    ///
    /// `partition_id` is the opaque driver-supplied identifier
    /// that distinguishes shards in the persisted progress set;
    /// it is also hex-encoded into the staging subdirectory name
    /// so each shard's SST files live in their own scratch space.
    ///
    /// `objects` is an iterator of fallible objects. Iterator
    /// errors abort the shard before any SST is written, so partial
    /// state never lands in the database.
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

        // 1. Fold objects into the typed accumulator. Per-object
        // errors abort the shard.
        let mut acc = <P::Batch as Default>::default();
        let mut object_count = 0usize;
        for object in objects {
            let object = object?;
            self.pipeline.restore(&mut acc, &object)?;
            object_count += 1;
        }

        // 2. Drain the accumulator into a shard-backed Batch. The
        // batch routes per-CF: BulkIngest CFs buffer into per-CF
        // sorted maps (one op per key); MergeViaWriteBatch CFs
        // stream into an internal rocksdb::WriteBatch.
        let mut batch = self.db.shard_batch();
        let row_count = self.pipeline.commit(&self.schema, &acc, &mut batch)?;
        drop(acc);

        // 3. Finalize. Even an empty batch produces a ShardFinalize
        // (with zero SSTs and an empty WriteBatch) — we still need
        // to stage and commit the partition-complete marker.
        let shard_dir = self.shard_staging_dir(partition_id);
        std::fs::create_dir_all(&shard_dir)?;
        let mut finalize = batch.finalize_for_shard(&shard_dir, &self.sst_options)?;

        // 4. Stage the partition-complete marker into the same
        // WriteBatch. Done under `state_lock` so the
        // read-modify-write of `partitions_complete` is serialized
        // with concurrent shard processors.
        {
            let _guard = self.state_lock.lock();
            let next_state = self.next_partition_state(partition_id)?;
            self.db
                .stage_restore_state(finalize.write_batch_mut(), P::NAME, &next_state)?;
            // 5. Atomic commit: SSTs ingest first; if a crash
            // happens between ingest and write_batch commit, the
            // marker never lands, the shard re-runs on resume, and
            // (a) SST re-ingest is idempotent, (b) merge-mode ops
            // never committed so no double-merge.
            finalize.commit()?;
        }

        // 6. Best-effort: remove the (now-empty) shard staging dir.
        if let Err(e) = std::fs::remove_dir_all(&shard_dir) {
            debug!(
                pipeline = P::NAME,
                partition = %hex_encode(partition_id),
                error = %e,
                "Failed to remove shard staging dir (non-fatal)",
            );
        }

        debug!(
            pipeline = P::NAME,
            partition = %hex_encode(partition_id),
            objects = object_count,
            rows = row_count,
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
        self.db.set_restore_state(
            P::NAME,
            &RestoreState::Complete {
                restored_at: self.target_checkpoint,
            },
        )?;
        info!(
            pipeline = P::NAME,
            restored_at = self.target_checkpoint,
            "Restore complete",
        );
        Ok(())
    }

    /// The path used to stage SSTs for `partition_id`. Exposed so
    /// tests and drivers can introspect the layout if needed.
    fn shard_staging_dir(&self, partition_id: &[u8]) -> PathBuf {
        self.staging_dir
            .join(format!("{}_{}", P::NAME, hex_encode(partition_id)))
    }

    /// Compute the next persisted [`RestoreState`] for this
    /// pipeline given that `partition_id` is about to be marked
    /// complete.
    ///
    /// Callers stage the returned state into the same
    /// [`rocksdb::WriteBatch`] that holds the shard's merge-mode
    /// writes, so the marker lands atomically with the writes.
    /// Must be invoked under [`state_lock`](Self::state_lock) so
    /// concurrent shard processors do not race on the
    /// read-modify-write of `partitions_complete`.
    fn next_partition_state(&self, partition_id: &[u8]) -> Result<RestoreState, Error> {
        let current = self.db.restore_state(P::NAME)?;
        let mut partitions_complete = match current {
            Some(RestoreState::InProgress {
                target_checkpoint,
                partitions_complete,
            }) => {
                debug_assert_eq!(
                    target_checkpoint, self.target_checkpoint,
                    "partition update after target_checkpoint mismatch — \
                     should have been caught at begin()",
                );
                partitions_complete
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
        partitions_complete.insert(partition_id.to_vec());
        Ok(RestoreState::InProgress {
            target_checkpoint: self.target_checkpoint,
            partitions_complete,
        })
    }
}

/// Hex-encode a byte slice for use in human-readable identifiers
/// (log fields, staging-directory names). The crate does not
/// otherwise depend on `hex`, and the implementation is small.
fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Buf;
    use bytes::BufMut;
    use sui_types::base_types::ObjectID;
    use sui_types::object::Object;
    use tempfile::TempDir;

    use super::*;
    use crate::Batch;
    use crate::DbMap;
    use crate::DbOptions;
    use crate::Decode;
    use crate::Encode;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

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
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            vec![crate::CfDescriptor::new("versions", base_options.clone())]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

    /// Test pipeline: per-object key → version. The accumulator
    /// keeps the highest version observed per id.
    struct VersionsPipeline;

    impl Pipeline for VersionsPipeline {
        const NAME: &'static str = "versions";

        type Schema = VersionsSchema;
        type Value = (ObjectID, u64);
        type Batch = BTreeMap<ObjectID, u64>;

        fn restore(&self, accumulator: &mut Self::Batch, object: &Object) -> anyhow::Result<()> {
            accumulator
                .entry(object.id())
                .and_modify(|hi| {
                    if object.version().value() > *hi {
                        *hi = object.version().value();
                    }
                })
                .or_insert(object.version().value());
            Ok(())
        }

        fn process(
            &self,
            _checkpoint: &sui_types::full_checkpoint_content::CheckpointData,
        ) -> anyhow::Result<Vec<Self::Value>> {
            Ok(vec![])
        }

        fn batch(&self, _: &mut Self::Batch, _: std::vec::IntoIter<Self::Value>) {}

        fn commit(
            &self,
            schema: &Self::Schema,
            batch: &Self::Batch,
            write_batch: &mut Batch,
        ) -> anyhow::Result<usize> {
            for (id, version) in batch {
                write_batch.put(&schema.versions, &ObjectIdKey::new(*id), &U64Be(*version))?;
            }
            Ok(batch.len())
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
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            let mut opts = base_options.clone();
            opts.set_merge_operator_associative("u64-add", add_u64_merge_op);
            // Merge-operator CF — opt into the WriteBatch shard mode
            // so the pipeline's per-shard accumulator can emit many
            // operands per key without DuplicateShardOp firing.
            vec![
                crate::CfDescriptor::new("counters", opts)
                    .with_restore_mode(crate::RestoreMode::MergeViaWriteBatch),
            ]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                counters: DbMap::new(db.clone(), "counters")?,
            })
        }
    }

    /// Test pipeline: counts how many times each id was observed.
    /// Uses a merge operator so cross-shard merges combine.
    struct CountersPipeline;

    impl Pipeline for CountersPipeline {
        const NAME: &'static str = "counters";

        type Schema = CountersSchema;
        type Value = ObjectID;
        type Batch = BTreeMap<ObjectID, u64>;

        fn restore(&self, accumulator: &mut Self::Batch, object: &Object) -> anyhow::Result<()> {
            *accumulator.entry(object.id()).or_insert(0) += 1;
            Ok(())
        }

        fn process(
            &self,
            _checkpoint: &sui_types::full_checkpoint_content::CheckpointData,
        ) -> anyhow::Result<Vec<Self::Value>> {
            Ok(vec![])
        }

        fn batch(&self, _: &mut Self::Batch, _: std::vec::IntoIter<Self::Value>) {}

        fn commit(
            &self,
            schema: &Self::Schema,
            batch: &Self::Batch,
            write_batch: &mut Batch,
        ) -> anyhow::Result<usize> {
            for (id, count) in batch {
                write_batch.merge(&schema.counters, &ObjectIdKey::new(*id), &U64Be(*count))?;
            }
            Ok(batch.len())
        }
    }

    /// Build a runner against a fresh database and a fresh staging
    /// dir. Returns the runner plus the lifetime-extending temp dirs.
    fn setup_versions(
        target_checkpoint: u64,
    ) -> (
        TempDir,
        TempDir,
        Arc<Db>,
        Arc<VersionsSchema>,
        RestoreRunner<VersionsPipeline>,
    ) {
        let db_dir = TempDir::new().unwrap();
        let staging = TempDir::new().unwrap();
        let (db, schema) = Db::open::<VersionsSchema>(db_dir.path(), DbOptions::default()).unwrap();
        let schema = Arc::new(schema);
        let runner = RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            target_checkpoint,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        );
        (db_dir, staging, db, schema, runner)
    }

    fn obj(id: u8) -> Object {
        Object::immutable_with_id_for_testing(ObjectID::from_single_byte(id))
    }

    #[test]
    fn begin_initializes_state_when_none() {
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        let skip = runner.begin().unwrap();
        assert!(skip.is_empty());
        match db.restore_state("versions").unwrap() {
            Some(RestoreState::InProgress {
                target_checkpoint,
                partitions_complete,
            }) => {
                assert_eq!(target_checkpoint, 100);
                assert!(partitions_complete.is_empty());
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn begin_returns_completed_partitions_from_prior_run() {
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        // Simulate a prior partial run.
        let mut prior = BTreeSet::new();
        prior.insert(vec![0xAAu8]);
        prior.insert(vec![0xBBu8]);
        db.set_restore_state(
            "versions",
            &RestoreState::InProgress {
                target_checkpoint: 100,
                partitions_complete: prior.clone(),
            },
        )
        .unwrap();

        let skip = runner.begin().unwrap();
        assert_eq!(skip, prior);
    }

    #[test]
    fn begin_refuses_target_mismatch() {
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        db.set_restore_state(
            "versions",
            &RestoreState::InProgress {
                target_checkpoint: 200,
                partitions_complete: BTreeSet::new(),
            },
        )
        .unwrap();
        let err = runner.begin().unwrap_err();
        assert!(format!("{err}").contains("mid-restore at checkpoint 200"));
    }

    #[test]
    fn begin_refuses_already_complete() {
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        db.set_restore_state("versions", &RestoreState::Complete { restored_at: 50 })
            .unwrap();
        let err = runner.begin().unwrap_err();
        assert!(format!("{err}").contains("already restored"));
    }

    #[test]
    fn process_shard_writes_data_and_marks_partition_complete() {
        let (_db_dir, _staging, db, schema, runner) = setup_versions(100);
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
        match db.restore_state("versions").unwrap() {
            Some(RestoreState::InProgress {
                partitions_complete,
                ..
            }) => {
                assert!(partitions_complete.contains(b"shard-0".as_slice()));
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn process_shard_skips_already_complete_partitions() {
        let (_db_dir, _staging, db, schema, runner) = setup_versions(100);
        runner.begin().unwrap();

        // Pre-mark shard-0 as complete.
        let mut done = BTreeSet::new();
        done.insert(b"shard-0".to_vec());
        db.set_restore_state(
            "versions",
            &RestoreState::InProgress {
                target_checkpoint: 100,
                partitions_complete: done,
            },
        )
        .unwrap();

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
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();
        runner
            .process_shard(b"empty", Vec::<anyhow::Result<Object>>::new())
            .unwrap();
        match db.restore_state("versions").unwrap() {
            Some(RestoreState::InProgress {
                partitions_complete,
                ..
            }) => {
                assert!(partitions_complete.contains(b"empty".as_slice()));
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn process_shard_propagates_object_stream_error_and_does_not_mark_complete() {
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();

        let stream: Vec<anyhow::Result<Object>> = vec![
            Ok(obj(1)),
            Err(anyhow::anyhow!("source-side IO failed")),
            Ok(obj(2)),
        ];
        let err = runner.process_shard(b"bad", stream).unwrap_err();
        assert!(format!("{err}").contains("source-side IO failed"));

        match db.restore_state("versions").unwrap() {
            Some(RestoreState::InProgress {
                partitions_complete,
                ..
            }) => {
                assert!(!partitions_complete.contains(b"bad".as_slice()));
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn cross_shard_merges_combine_through_runner() {
        // End-to-end exercise of the design's restore story using
        // the merge-operator pipeline. Two shards each touch the
        // same object id; the runner emits one merge per shard's
        // SST, and the registered operator combines them on read.
        let db_dir = TempDir::new().unwrap();
        let staging = TempDir::new().unwrap();
        let (db, schema) = Db::open::<CountersSchema>(db_dir.path(), DbOptions::default()).unwrap();
        let schema = Arc::new(schema);
        let runner = RestoreRunner::new(
            db.clone(),
            Arc::new(CountersPipeline),
            schema.clone(),
            42,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
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
        match db.restore_state("counters").unwrap() {
            Some(RestoreState::Complete { restored_at }) => assert_eq!(restored_at, 42),
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn finish_transitions_to_complete() {
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
        runner.begin().unwrap();
        runner.process_shard(b"only", vec![Ok(obj(1))]).unwrap();
        runner.finish().unwrap();
        match db.restore_state("versions").unwrap() {
            Some(RestoreState::Complete { restored_at }) => assert_eq!(restored_at, 100),
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn already_complete_returns_true_after_finish() {
        let (_db_dir, _staging, _db, _schema, runner) = setup_versions(100);
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
        let (_db_dir, _staging, db, _schema, runner) = setup_versions(100);
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

        match db.restore_state("versions").unwrap() {
            Some(RestoreState::InProgress {
                partitions_complete,
                ..
            }) => {
                assert_eq!(partitions_complete.len(), 16);
                for i in 0..16u8 {
                    assert!(partitions_complete.contains(&vec![i]));
                }
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn hex_encode_roundtrips_byte_values() {
        assert_eq!(super::hex_encode(&[]), "");
        assert_eq!(super::hex_encode(&[0]), "00");
        assert_eq!(super::hex_encode(&[0xAB, 0xCD]), "abcd");
        assert_eq!(super::hex_encode(&[0xFF; 4]), "ffffffff");
    }

    /// Tests for the hybrid restore mode: pipelines that write to a
    /// mix of [`crate::RestoreMode::BulkIngest`] and
    /// [`crate::RestoreMode::MergeViaWriteBatch`] CFs in the same
    /// shard.
    mod hybrid {
        use super::*;

        /// Schema with two CFs: `versions` (plain put, BulkIngest)
        /// and `counters` (merge operator, MergeViaWriteBatch).
        #[derive(Debug)]
        struct HybridSchema {
            versions: DbMap<ObjectIdKey, U64Be>,
            counters: DbMap<ObjectIdKey, U64Be>,
        }

        impl Schema for HybridSchema {
            fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
                let mut counter_opts = base_options.clone();
                counter_opts.set_merge_operator_associative("u64-add", add_u64_merge_op);
                vec![
                    crate::CfDescriptor::new("versions", base_options.clone()),
                    crate::CfDescriptor::new("counters", counter_opts)
                        .with_restore_mode(crate::RestoreMode::MergeViaWriteBatch),
                ]
            }

            fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
                Ok(Self {
                    versions: DbMap::new(db.clone(), "versions")?,
                    counters: DbMap::new(db.clone(), "counters")?,
                })
            }
        }

        /// Pipeline that writes both CFs: per object emit one
        /// `versions` put (folded by id) plus one `counters` merge
        /// operand per observation. With repeated observations in a
        /// shard the pipeline emits *many* merges per id —
        /// previously rejected by DuplicateShardOp; now allowed by
        /// the WriteBatch backing for the `counters` CF.
        struct HybridPipeline;

        impl Pipeline for HybridPipeline {
            const NAME: &'static str = "hybrid";
            type Schema = HybridSchema;
            type Value = (ObjectID, u64);
            // Two accumulators: versions (fold by id), counters
            // (preserve per-observation deltas as a Vec).
            type Batch = HybridBatch;

            fn restore(
                &self,
                accumulator: &mut Self::Batch,
                object: &Object,
            ) -> anyhow::Result<()> {
                let id = object.id();
                let v = object.version().value();
                accumulator
                    .versions
                    .entry(id)
                    .and_modify(|hi| {
                        if v > *hi {
                            *hi = v;
                        }
                    })
                    .or_insert(v);
                accumulator.counters.push((id, 1));
                Ok(())
            }

            fn process(
                &self,
                _: &sui_types::full_checkpoint_content::CheckpointData,
            ) -> anyhow::Result<Vec<Self::Value>> {
                Ok(vec![])
            }

            fn batch(&self, _: &mut Self::Batch, _: std::vec::IntoIter<Self::Value>) {}

            fn commit(
                &self,
                schema: &Self::Schema,
                batch: &Self::Batch,
                write_batch: &mut Batch,
            ) -> anyhow::Result<usize> {
                for (id, version) in &batch.versions {
                    write_batch.put(&schema.versions, &ObjectIdKey::new(*id), &U64Be(*version))?;
                }
                // Many merges per id allowed because `counters` is
                // MergeViaWriteBatch.
                for (id, delta) in &batch.counters {
                    write_batch.merge(&schema.counters, &ObjectIdKey::new(*id), &U64Be(*delta))?;
                }
                Ok(batch.versions.len() + batch.counters.len())
            }
        }

        #[derive(Default)]
        struct HybridBatch {
            versions: BTreeMap<ObjectID, u64>,
            counters: Vec<(ObjectID, u64)>,
        }

        fn setup_hybrid(
            target_checkpoint: u64,
        ) -> (
            TempDir,
            TempDir,
            Arc<Db>,
            Arc<HybridSchema>,
            RestoreRunner<HybridPipeline>,
        ) {
            let db_dir = TempDir::new().unwrap();
            let staging = TempDir::new().unwrap();
            let (db, schema) =
                Db::open::<HybridSchema>(db_dir.path(), DbOptions::default()).unwrap();
            let schema = Arc::new(schema);
            let runner = RestoreRunner::new(
                db.clone(),
                Arc::new(HybridPipeline),
                schema.clone(),
                target_checkpoint,
                staging.path().to_path_buf(),
                rocksdb::Options::default(),
            );
            (db_dir, staging, db, schema, runner)
        }

        #[test]
        fn shard_with_mixed_cfs_writes_both_correctly() {
            // One shard with three objects (two distinct ids).
            // Versions: 2 puts. Counters: 3 merges (with two for
            // the same id — only legal because counters is
            // MergeViaWriteBatch).
            let (_db_dir, _staging, db, schema, runner) = setup_hybrid(7);
            runner.begin().unwrap();
            runner
                .process_shard(b"shard-0", vec![Ok(obj(1)), Ok(obj(1)), Ok(obj(2))])
                .unwrap();

            // Versions: highest observed per id.
            assert_eq!(
                schema
                    .versions
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap(),
                Some(U64Be(obj(1).version().value())),
            );
            assert_eq!(
                schema
                    .versions
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(2)))
                    .unwrap(),
                Some(U64Be(obj(2).version().value())),
            );

            // Counters: two merges for id 1 (sum=2), one for id 2.
            assert_eq!(
                schema
                    .counters
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap(),
                Some(U64Be(2)),
            );
            assert_eq!(
                schema
                    .counters
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(2)))
                    .unwrap(),
                Some(U64Be(1)),
            );

            // Partition marked complete atomically with the counters
            // writes.
            match db.restore_state("hybrid").unwrap() {
                Some(RestoreState::InProgress {
                    partitions_complete,
                    ..
                }) => assert!(partitions_complete.contains(b"shard-0".as_slice())),
                other => panic!("expected InProgress, got {other:?}"),
            }
        }

        #[test]
        fn cross_shard_merges_combine_via_operator() {
            // Two shards each contribute multiple merges for the
            // same id. WriteBatch commits combine within a shard
            // (operator-summed) and across shards (operator-summed
            // again at read).
            let (_db_dir, _staging, _db, schema, runner) = setup_hybrid(7);
            runner.begin().unwrap();
            runner
                .process_shard(b"shard-A", vec![Ok(obj(1)), Ok(obj(1)), Ok(obj(1))])
                .unwrap();
            runner
                .process_shard(b"shard-B", vec![Ok(obj(1)), Ok(obj(1))])
                .unwrap();
            runner.finish().unwrap();

            // 3 + 2 = 5 merges for id 1, each operand = 1.
            assert_eq!(
                schema
                    .counters
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap(),
                Some(U64Be(5)),
            );
        }

        #[test]
        fn partition_marker_is_atomic_with_merge_writes() {
            // Manually drive a single shard through the same surface
            // the runner uses, but commit the WriteBatch carrying
            // the marker only at the very end. Until then, no merges
            // are visible.
            let (_db_dir, staging, db, schema, _runner) = setup_hybrid(7);

            // Mid-restore state: we'd normally have begun via the
            // runner, but the marker we stage below replaces any
            // existing entry, so this is fine.
            let pipeline = HybridPipeline;
            let mut acc = HybridBatch::default();
            pipeline.restore(&mut acc, &obj(1)).unwrap();
            pipeline.restore(&mut acc, &obj(1)).unwrap();

            let mut wb_batch = db.shard_batch();
            pipeline.commit(&*schema, &acc, &mut wb_batch).unwrap();

            let shard_dir = staging.path().join("manual");
            std::fs::create_dir_all(&shard_dir).unwrap();
            let mut finalize = wb_batch
                .finalize_for_shard(&shard_dir, &rocksdb::Options::default())
                .unwrap();

            // Stage marker but do not commit yet.
            let marker = RestoreState::InProgress {
                target_checkpoint: 7,
                partitions_complete: {
                    let mut s = BTreeSet::new();
                    s.insert(b"manual".to_vec());
                    s
                },
            };
            db.stage_restore_state(finalize.write_batch_mut(), HybridPipeline::NAME, &marker)
                .unwrap();

            // Pre-commit: marker is absent and counter is absent.
            assert!(db.restore_state(HybridPipeline::NAME).unwrap().is_none());
            assert!(
                schema
                    .counters
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap()
                    .is_none(),
            );

            finalize.commit().unwrap();

            // Post-commit: both visible.
            assert_eq!(
                db.restore_state(HybridPipeline::NAME).unwrap(),
                Some(marker)
            );
            assert_eq!(
                schema
                    .counters
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap(),
                Some(U64Be(2)),
            );
        }

        #[test]
        fn resume_does_not_double_merge_after_simulated_crash() {
            // Simulate the crash-between-ingest-and-marker scenario.
            // We do this in two ways the runner could be interrupted
            // and verify the resume doesn't double-apply merges.
            //
            // Approach: drive the first shard halfway — finalize the
            // batch, ingest SSTs, but DROP the ShardFinalize without
            // committing its WriteBatch (so merges never land and
            // the marker never lands). Then drive again from scratch
            // through the runner; the partition is not complete, so
            // it re-runs end-to-end. Verify the final merge result
            // equals exactly one run, not two.
            let (_db_dir, staging, db, schema, runner) = setup_hybrid(7);
            runner.begin().unwrap();

            // First "interrupted" run: finalize but don't commit.
            {
                let pipeline = HybridPipeline;
                let mut acc = HybridBatch::default();
                for _ in 0..3 {
                    pipeline.restore(&mut acc, &obj(1)).unwrap();
                }
                let mut wb_batch = db.shard_batch();
                pipeline.commit(&*schema, &acc, &mut wb_batch).unwrap();
                let shard_dir = staging.path().join("interrupted_shard-X");
                std::fs::create_dir_all(&shard_dir).unwrap();
                let mut finalize = wb_batch
                    .finalize_for_shard(&shard_dir, &rocksdb::Options::default())
                    .unwrap();
                // Ingest the SSTs only (simulates a crash *between*
                // SST ingest and WriteBatch commit).
                let ssts = finalize.take_ssts();
                for (cf, path) in ssts {
                    db.ingest_files_cf(&cf, vec![path]).unwrap();
                }
                // Drop `finalize` (with its WriteBatch) without
                // committing. The marker never lands and the
                // `counters` merges never apply.
                drop(finalize);
            }

            // The partition is still not marked complete, so resume
            // re-runs the shard. counters should reflect exactly one
            // successful run (3 merges = 3), not two (6).
            runner
                .process_shard(b"shard-X", vec![Ok(obj(1)), Ok(obj(1)), Ok(obj(1))])
                .unwrap();
            runner.finish().unwrap();

            assert_eq!(
                schema
                    .counters
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap(),
                Some(U64Be(3)),
                "merges from the failed first run must not double-apply on resume",
            );
            // Versions (BulkIngest) reflect the successful run.
            assert_eq!(
                schema
                    .versions
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap(),
                Some(U64Be(obj(1).version().value())),
            );
        }

        #[test]
        fn duplicate_put_in_bulk_ingest_cf_still_errors() {
            // Hybrid mode preserves the SST one-op-per-key invariant
            // for BulkIngest CFs. The pipeline can't emit two puts
            // for the same id in a single shard's `versions` write.
            let (_db_dir, _staging, db, schema, _runner) = setup_hybrid(7);
            let mut batch = db.shard_batch();
            batch
                .put(
                    &schema.versions,
                    &ObjectIdKey::new(ObjectID::from_single_byte(1)),
                    &U64Be(1),
                )
                .unwrap();
            let err = batch
                .put(
                    &schema.versions,
                    &ObjectIdKey::new(ObjectID::from_single_byte(1)),
                    &U64Be(2),
                )
                .unwrap_err();
            assert!(matches!(err, crate::error::Error::DuplicateShardOp { .. }));
        }

        #[test]
        fn many_merges_per_key_in_merge_mode_cf_succeed() {
            // Counterpart to the above: many merges per key per
            // shard succeed in MergeViaWriteBatch CFs.
            let (_db_dir, _staging, db, schema, _runner) = setup_hybrid(7);
            let mut batch = db.shard_batch();
            for _ in 0..10 {
                batch
                    .merge(
                        &schema.counters,
                        &ObjectIdKey::new(ObjectID::from_single_byte(1)),
                        &U64Be(1),
                    )
                    .unwrap();
            }
            assert_eq!(batch.len(), 10);
        }
    }
}
