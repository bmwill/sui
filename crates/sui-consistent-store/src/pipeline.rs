// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Pipeline`] trait — a unified shape for a single indexing
//! pipeline that drives both restore and tip-of-chain flows.
//!
//! A schema author implements `Pipeline` once for a logical unit of
//! indexing (an owner index, a balance accumulator, a coin lookup),
//! and four shells consume the impl:
//!
//! 1. A restore driver fed by a formal-snapshot live-object stream.
//! 2. A restore driver fed by the validator's perpetual-store
//!    live-object iterator.
//! 3. A tip driver inside `sui-indexer-alt-framework` (provided by
//!    the future `sui-consistent-store-framework` crate).
//! 4. A tip driver inside the validator's checkpoint executor.
//!
//! The split mirrors `sui-indexer-alt-framework`'s
//! `Processor` + `sequential::Handler`, with [`restore`](Pipeline::restore)
//! added on top so the same impl serves the bulk-load path.
//!
//! # Method roles
//!
//! Both restore and tip funnel through the same accumulator type
//! ([`Pipeline::Batch`]) and the same write path
//! ([`commit`](Pipeline::commit)). The only thing that differs is
//! how the accumulator is populated:
//!
//! - [`restore`](Pipeline::restore) folds updates derived from a
//!   single live object into a per-shard accumulator. Drivers
//!   parallelize across input objects up to
//!   [`RESTORE_FANOUT`](Pipeline::RESTORE_FANOUT). Each worker owns
//!   its own accumulator; the driver feeds it to
//!   [`commit`](Pipeline::commit) when the shard is done.
//! - [`process`](Pipeline::process) is called once per checkpoint
//!   to extract typed rows from a [`CheckpointData`]. Pure
//!   function; called from worker threads under the framework's
//!   processor stage and from the validator's checkpoint executor.
//! - [`batch`](Pipeline::batch) folds rows from one or more
//!   consecutive checkpoints into the pipeline's accumulator. The
//!   driver decides how many checkpoints to fold per commit, up to
//!   [`MAX_BATCH_CHECKPOINTS`](Pipeline::MAX_BATCH_CHECKPOINTS).
//! - [`commit`](Pipeline::commit) applies the folded accumulator to
//!   a [`Batch`]. The driver commits the batch atomically alongside
//!   any other state it owns (watermarks, restore markers).
//!
//! # Why `anyhow::Error` rather than [`crate::error::Error`]
//!
//! Existing indexer-alt pipelines return [`anyhow::Error`] from
//! `Processor::process`. Aligning here keeps migrations a
//! single-line change. Pipeline authors who only call
//! [`Batch`] methods (which return [`crate::error::Error`]) get
//! free conversion via `?` because [`crate::error::Error`]
//! implements [`std::error::Error`].

use sui_types::full_checkpoint_content::CheckpointData;
use sui_types::object::Object;

use crate::Batch;

/// A single indexing pipeline.
///
/// Implementations are typically unit structs whose trait methods
/// are essentially functions; `&self` is supplied for symmetry with
/// `sui-indexer-alt-framework`'s `Processor` trait, so per-instance
/// configuration (a config struct loaded from CLI, a logger handle)
/// can live on the implementing type.
pub trait Pipeline: Send + Sync + 'static {
    /// Identifies the pipeline in logs, metrics, and persisted
    /// per-pipeline progress markers (the `__restore` column
    /// family added in a follow-up commit).
    ///
    /// Must be unique among the pipelines registered against a
    /// single database; drivers use it as a primary key.
    const NAME: &'static str;

    /// The portion of the schema this pipeline reads from and
    /// writes to.
    ///
    /// Typically a borrow of the full schema or a sub-projection
    /// (a struct of [`DbMap`](crate::DbMap) fields scoped to the
    /// CFs this pipeline owns). The driver supplies an
    /// `&Self::Schema` on every call.
    type Schema: Send + Sync;

    /// The type produced by [`process`](Self::process) and
    /// consumed by [`batch`](Self::batch). Equivalent to
    /// `Processor::Value` in the framework.
    type Value: Send + Sync + 'static;

    /// The accumulator type into which [`batch`](Self::batch) folds
    /// values across one or more checkpoints. The driver decides
    /// how much to fold (per-checkpoint or many consecutive
    /// checkpoints).
    type Batch: Default + Send + Sync + 'static;

    /// Restore-side parallelism the driver should use for this
    /// pipeline. Both restore drivers honor it (number of worker
    /// threads per partition or per shard).
    const RESTORE_FANOUT: usize = 10;

    /// Maximum number of consecutive checkpoints a tip driver may
    /// fold into a single commit. The framework driver honors this
    /// to batch sequential commits; the validator driver always
    /// commits one checkpoint at a time and ignores the value.
    const MAX_BATCH_CHECKPOINTS: usize = 5 * 60;

    /// Fold updates derived from a single live object into the
    /// per-shard accumulator.
    ///
    /// Called from worker threads under both restore drivers, up
    /// to [`RESTORE_FANOUT`](Self::RESTORE_FANOUT) in parallel per
    /// pipeline. Each worker owns its own `accumulator`; objects
    /// may arrive in any order within a worker's slice. The
    /// pipeline is free to fold (combine deltas, dedup by key)
    /// before [`commit`](Self::commit) emits writes — this avoids
    /// staging redundant operations in the shard's
    /// [`Batch`](crate::Batch).
    ///
    /// Cross-shard collisions are RocksDB's concern: two shards
    /// that both touch the same key each emit their own ops, and
    /// the registered merge operator (for merge-CFs) or last-write
    /// semantics (for put-CFs) reconcile them.
    fn restore(&self, accumulator: &mut Self::Batch, object: &Object) -> anyhow::Result<()>;

    /// Extract typed values from a checkpoint.
    ///
    /// Pure function; called from worker threads. Errors are
    /// surfaced via [`anyhow::Error`]; framework callers may retry
    /// transient failures. Permanently un-processable input
    /// (malformed objects, protocol-violating data) should panic so
    /// the indexer halts and an operator is alerted.
    fn process(&self, checkpoint: &CheckpointData) -> anyhow::Result<Vec<Self::Value>>;

    /// Fold values from one or more checkpoints into the running
    /// accumulator.
    ///
    /// Values are presented in checkpoint order. The same `batch`
    /// receives values from many consecutive checkpoints before
    /// the driver calls [`commit`](Self::commit), up to
    /// [`MAX_BATCH_CHECKPOINTS`](Self::MAX_BATCH_CHECKPOINTS).
    fn batch(&self, batch: &mut Self::Batch, values: std::vec::IntoIter<Self::Value>);

    /// Apply the folded accumulator to `write_batch`.
    ///
    /// The pipeline encodes its rows into `write_batch` via the
    /// [`Batch`] API (typed `put`, `delete`, `merge` against
    /// [`DbMap`](crate::DbMap) handles in `schema`). The driver
    /// commits `write_batch` atomically — typically alongside a
    /// watermark write of its own.
    ///
    /// Returns the number of rows applied, for metrics.
    fn commit(
        &self,
        schema: &Self::Schema,
        batch: &Self::Batch,
        write_batch: &mut Batch,
    ) -> anyhow::Result<usize>;
}

#[cfg(test)]
mod tests {
    //! End-to-end exercise of all four [`Pipeline`] methods against
    //! an in-memory `Db`. No restore drivers yet — the test calls
    //! [`Pipeline::restore`] directly. This verifies the trait
    //! shape compiles and the four methods are wired up correctly.

    use std::collections::BTreeMap;

    use bytes::Buf;
    use bytes::BufMut;
    use sui_types::base_types::ObjectID;
    use sui_types::object::Object;
    use sui_types::test_checkpoint_data_builder::TestCheckpointBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::Db;
    use crate::DbMap;
    use crate::DbOptions;
    use crate::Decode;
    use crate::Encode;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

    /// Big-endian `ObjectID` newtype, suitable as a typed key.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

    /// Big-endian `u64` value, suitable as both a typed value and a
    /// per-object count.
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

    /// Schema with a single CF mapping object id → version.
    #[derive(Debug)]
    struct ObjectVersionSchema {
        versions: DbMap<ObjectIdKey, U64Be>,
    }

    impl Schema for ObjectVersionSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            vec![crate::CfDescriptor::new("versions", base_options.clone())]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

    /// Test pipeline: tracks the latest version observed for each
    /// object. `restore` writes one entry per live object;
    /// `process` extracts (id, version) pairs from a checkpoint's
    /// output objects; `batch` folds by id, keeping the highest
    /// version; `commit` writes the folded entries.
    struct ObjectVersionPipeline;

    #[derive(Debug, Clone, Copy)]
    struct VersionRow {
        id: ObjectID,
        version: u64,
    }

    impl Pipeline for ObjectVersionPipeline {
        const NAME: &'static str = "object_version";

        type Schema = ObjectVersionSchema;
        type Value = VersionRow;
        type Batch = BTreeMap<ObjectID, u64>;

        fn restore(&self, accumulator: &mut Self::Batch, object: &Object) -> anyhow::Result<()> {
            // One entry per object — restore folds the same way
            // `batch` would: keep the highest version observed.
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

        fn process(&self, checkpoint: &CheckpointData) -> anyhow::Result<Vec<Self::Value>> {
            let mut rows = vec![];
            for tx in &checkpoint.transactions {
                for object in &tx.output_objects {
                    rows.push(VersionRow {
                        id: object.id(),
                        version: object.version().value(),
                    });
                }
            }
            Ok(rows)
        }

        fn batch(&self, batch: &mut Self::Batch, values: std::vec::IntoIter<Self::Value>) {
            for v in values {
                batch
                    .entry(v.id)
                    .and_modify(|hi| {
                        if v.version > *hi {
                            *hi = v.version
                        }
                    })
                    .or_insert(v.version);
            }
        }

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

    /// Open a fresh database with `ObjectVersionSchema`.
    fn open() -> (TempDir, Db, ObjectVersionSchema) {
        let dir = TempDir::new().unwrap();
        let (db, schema) =
            Db::open::<ObjectVersionSchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db, schema)
    }

    #[test]
    fn restore_folds_into_accumulator_then_commit_writes() {
        let (_dir, db, schema) = open();
        let pipeline = ObjectVersionPipeline;

        let o1 = Object::immutable_with_id_for_testing(ObjectID::from_single_byte(1));
        let o2 = Object::immutable_with_id_for_testing(ObjectID::from_single_byte(2));

        // Restore folds into the typed accumulator — no Db writes
        // happen here. The runner would do many of these, then
        // call `commit` once per shard.
        let mut acc = <ObjectVersionPipeline as Pipeline>::Batch::default();
        pipeline.restore(&mut acc, &o1).unwrap();
        pipeline.restore(&mut acc, &o2).unwrap();
        assert_eq!(acc.len(), 2);

        // Commit writes the accumulator to a `Batch`, which the
        // driver commits atomically. The pipeline does not call
        // `Batch::commit` itself.
        let mut write_batch = db.batch();
        let n = pipeline.commit(&schema, &acc, &mut write_batch).unwrap();
        write_batch.commit().unwrap();
        assert_eq!(n, 2);

        assert_eq!(
            schema.versions.get(&ObjectIdKey::new(o1.id())).unwrap(),
            Some(U64Be(o1.version().value())),
        );
        assert_eq!(
            schema.versions.get(&ObjectIdKey::new(o2.id())).unwrap(),
            Some(U64Be(o2.version().value())),
        );
    }

    #[test]
    fn restore_dedups_repeated_objects_into_one_accumulator_entry() {
        // Two restore calls for the same object id (same shard)
        // fold into the accumulator: one entry, the highest
        // version wins.
        let pipeline = ObjectVersionPipeline;
        let id = ObjectID::from_single_byte(7);
        let obj = Object::immutable_with_id_for_testing(id);
        let version = obj.version().value();

        let mut acc = <ObjectVersionPipeline as Pipeline>::Batch::default();
        pipeline.restore(&mut acc, &obj).unwrap();
        pipeline.restore(&mut acc, &obj).unwrap();
        assert_eq!(acc.len(), 1);
        assert_eq!(acc.get(&id), Some(&version));
    }

    #[test]
    fn process_batch_commit_writes_latest_version_per_object() {
        let (_dir, db, schema) = open();
        let pipeline = ObjectVersionPipeline;

        // One builder, two checkpoints: cp1 creates objects #1 and
        // #2, cp2 mutates #1 (bumping its version).
        // `TestCheckpointBuilder` retains created objects across
        // `build_checkpoint` calls, so mutations in cp2 see them.
        // The transaction helpers consume `self`; chain through a
        // single rebound binding to preserve state across checkpoints.
        let mut builder = TestCheckpointBuilder::new(1)
            .start_transaction(0)
            .create_owned_object(1)
            .create_owned_object(2)
            .finish_transaction();
        let cp1: CheckpointData = builder.build_checkpoint().into();
        builder = builder
            .start_transaction(0)
            .mutate_owned_object(1)
            .finish_transaction();
        let cp2: CheckpointData = builder.build_checkpoint().into();

        let values1 = pipeline.process(&cp1).unwrap();
        let values2 = pipeline.process(&cp2).unwrap();
        assert!(!values1.is_empty(), "cp1 produced no output objects");
        assert!(!values2.is_empty(), "cp2 produced no output objects");

        // The mutated object appears in cp2's output objects with a
        // higher version than in cp1. Snapshot the pre-fold per-id
        // maxes from the raw values to drive the post-fold assertion.
        let mut expected_by_id: BTreeMap<ObjectID, u64> = BTreeMap::new();
        for v in values1.iter().chain(values2.iter()) {
            expected_by_id
                .entry(v.id)
                .and_modify(|hi| {
                    if v.version > *hi {
                        *hi = v.version;
                    }
                })
                .or_insert(v.version);
        }

        // Fold both checkpoints' values into one accumulator.
        let mut acc = <ObjectVersionPipeline as Pipeline>::Batch::default();
        pipeline.batch(&mut acc, values1.into_iter());
        pipeline.batch(&mut acc, values2.into_iter());
        assert_eq!(acc, expected_by_id);

        let mut write_batch = db.batch();
        let n = pipeline.commit(&schema, &acc, &mut write_batch).unwrap();
        write_batch.commit().unwrap();
        assert_eq!(n, acc.len());

        // Iterate the persisted state and confirm each row matches
        // the folded accumulator.
        let mut persisted_count = 0usize;
        for entry in schema.versions.iter(..).unwrap() {
            let (k, v) = entry.unwrap();
            let stored_id = ObjectID::new(k.0);
            let expected = *acc.get(&stored_id).unwrap();
            assert_eq!(v.0, expected, "id={stored_id}");
            persisted_count += 1;
        }
        assert_eq!(persisted_count, acc.len());
    }

    #[test]
    fn empty_batch_commit_writes_nothing() {
        let (_dir, db, schema) = open();
        let pipeline = ObjectVersionPipeline;
        let acc = <ObjectVersionPipeline as Pipeline>::Batch::default();
        let mut write_batch = db.batch();
        let n = pipeline.commit(&schema, &acc, &mut write_batch).unwrap();
        write_batch.commit().unwrap();
        assert_eq!(n, 0);
        // No rows persisted.
        let rows = schema.versions.iter(..).unwrap().count();
        assert_eq!(rows, 0);
    }

    #[test]
    fn batch_folds_to_highest_version_per_id() {
        let pipeline = ObjectVersionPipeline;
        let mut acc = <ObjectVersionPipeline as Pipeline>::Batch::default();
        let id = ObjectID::from_single_byte(7);
        let values = vec![
            VersionRow { id, version: 1 },
            VersionRow { id, version: 5 },
            VersionRow { id, version: 3 },
        ];
        pipeline.batch(&mut acc, values.into_iter());
        assert_eq!(acc.get(&id), Some(&5));
    }
}
