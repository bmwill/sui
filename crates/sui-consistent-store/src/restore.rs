// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Restore`] trait — bulk-load shape for an indexing
//! pipeline driven from a stream of live objects (formal snapshot,
//! perpetual store).
//!
//! `Restore` is independent of the tip-of-chain shape (the
//! indexer-alt framework's `Processor` + `sequential::Handler`).
//! A pipeline that needs both bulk-load and tip indexing
//! implements both: the two traits typically share the same
//! `Batch` accumulator type, but the framework does not require
//! that.
//!
//! # Method roles
//!
//! - [`restore`](Restore::restore) folds updates derived from a
//!   single live object into a per-shard accumulator. Restore
//!   drivers parallelize across input objects however they see fit
//!   (e.g. one tokio task per snapshot partition, one thread per
//!   `ObjectID` range). Each worker owns its own accumulator; the
//!   driver feeds it to [`commit`](Restore::commit) when the shard
//!   is done.
//! - [`commit`](Restore::commit) applies the folded accumulator to
//!   a [`Batch`]. The driver commits the batch atomically alongside
//!   any other state it owns (e.g. a partition-complete marker in
//!   the `__restore` CF).
//!
//! # Why `anyhow::Error`
//!
//! Matches the indexer-alt framework's `Processor`/`Handler`
//! signatures, so a single pipeline can return one error type from
//! both its restore and tip methods. Pipeline authors who only
//! call [`Batch`] methods (which return [`crate::error::Error`])
//! get free conversion via `?`.

use sui_types::object::Object;

use crate::Batch;

/// A pipeline that can be populated from a stream of live objects.
///
/// Implementations are typically unit structs whose trait methods
/// are essentially functions; `&self` is supplied so per-instance
/// configuration (a config struct loaded from CLI, a logger handle)
/// can live on the implementing type.
pub trait Restore: Send + Sync + 'static {
    /// Identifies the pipeline in logs, metrics, and persisted
    /// per-pipeline progress markers (the `__restore` column
    /// family).
    ///
    /// Must be unique among the pipelines registered against a
    /// single database; drivers use it as a primary key.
    const NAME: &'static str;

    /// The portion of the schema this pipeline reads from and
    /// writes to.
    ///
    /// Typically a struct of [`DbMap`](crate::DbMap) fields scoped
    /// to the CFs this pipeline owns. The driver supplies an
    /// `&Self::Schema` on every [`commit`](Self::commit) call.
    type Schema: Send + Sync;

    /// The accumulator type into which [`restore`](Self::restore)
    /// folds per-object updates. Each shard worker owns one; the
    /// driver hands it to [`commit`](Self::commit) once the shard
    /// is done.
    type Batch: Default + Send + Sync + 'static;

    /// Fold updates derived from a single live object into the
    /// per-shard accumulator.
    ///
    /// Each worker owns its own `accumulator`; objects may arrive
    /// in any order within a worker's slice. The pipeline is free
    /// to fold (combine deltas, dedup by key) before
    /// [`commit`](Self::commit) emits writes — this avoids staging
    /// redundant operations in the shard's [`Batch`].
    ///
    /// Cross-shard collisions are RocksDB's concern: two shards
    /// that both touch the same key each emit their own ops, and
    /// the registered merge operator (for merge-CFs) or
    /// last-write semantics (for put-CFs) reconcile them.
    fn restore(&self, accumulator: &mut Self::Batch, object: &Object) -> anyhow::Result<()>;

    /// Apply the folded accumulator to `write_batch`.
    ///
    /// The pipeline encodes its rows into `write_batch` via the
    /// [`Batch`] API (typed `put`, `delete`, `merge` against
    /// [`DbMap`](crate::DbMap) handles in `schema`). The driver
    /// commits `write_batch` atomically — typically alongside a
    /// partition-complete marker of its own.
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
    //! End-to-end exercise of both [`Restore`] methods against an
    //! in-memory `Db`. No restore drivers here — the test calls
    //! [`Restore::restore`] / [`Restore::commit`] directly to
    //! verify the trait shape compiles and the two methods are
    //! wired up correctly.

    use std::collections::BTreeMap;

    use bytes::Buf;
    use bytes::BufMut;
    use sui_types::base_types::ObjectID;
    use sui_types::object::Object;
    use tempfile::TempDir;

    use super::*;
    use crate::CfDescriptor;
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

    #[derive(Debug)]
    struct ObjectVersionSchema {
        versions: DbMap<ObjectIdKey, U64Be>,
    }

    impl Schema for ObjectVersionSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
            vec![CfDescriptor::new("versions", base_options.clone())]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

    /// Test pipeline: tracks the latest version observed for each
    /// object id. `restore` folds the per-shard accumulator by id,
    /// keeping the highest version; `commit` writes the folded
    /// entries.
    struct ObjectVersionPipeline;

    impl Restore for ObjectVersionPipeline {
        const NAME: &'static str = "object_version";

        type Schema = ObjectVersionSchema;
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

        let mut acc = <ObjectVersionPipeline as Restore>::Batch::default();
        pipeline.restore(&mut acc, &o1).unwrap();
        pipeline.restore(&mut acc, &o2).unwrap();
        assert_eq!(acc.len(), 2);

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
        let pipeline = ObjectVersionPipeline;
        let id = ObjectID::from_single_byte(7);
        let obj = Object::immutable_with_id_for_testing(id);
        let version = obj.version().value();

        let mut acc = <ObjectVersionPipeline as Restore>::Batch::default();
        pipeline.restore(&mut acc, &obj).unwrap();
        pipeline.restore(&mut acc, &obj).unwrap();
        assert_eq!(acc.len(), 1);
        assert_eq!(acc.get(&id), Some(&version));
    }

    #[test]
    fn empty_batch_commit_writes_nothing() {
        let (_dir, db, schema) = open();
        let pipeline = ObjectVersionPipeline;
        let acc = <ObjectVersionPipeline as Restore>::Batch::default();
        let mut write_batch = db.batch();
        let n = pipeline.commit(&schema, &acc, &mut write_batch).unwrap();
        write_batch.commit().unwrap();
        assert_eq!(n, 0);
        let rows = schema.versions.iter(..).unwrap().count();
        assert_eq!(rows, 0);
    }
}
