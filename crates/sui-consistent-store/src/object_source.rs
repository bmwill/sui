// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`LiveObjectSource`] — the trait that abstracts "yield objects
//! for an `ObjectID` range" — plus
//! [`restore_pipeline_from_object_source`], the parallel driver
//! that uses it to drive a [`RestoreRunner`].
//!
//! Use this when the restore input is a single object iterator
//! shardable by `ObjectID` prefix (the validator's perpetual
//! store is the canonical example). The trait stays in this
//! crate; the `AuthorityStore` impl lives in `sui-core` where
//! that dep tree is already paid for.
//!
//! The companion restore source — fetched-per-partition formal
//! snapshots — uses its own driver in [`formal_snapshot`](crate::formal_snapshot),
//! because partitions there come from the producer's bucket /
//! partition numbering rather than a key-range slicing.
//!
//! # Partitioning
//!
//! The driver slices the [`ObjectID`] space into `1 << shard_bits`
//! shards via the top bits of byte 0 of the address. Each shard
//! gets one worker thread under [`std::thread::scope`]. The
//! per-shard `partition_id` persisted in
//! [`RestoreState::InProgress`](crate::RestoreState::InProgress)
//! is `[shard_bits, shard_index]`, two bytes — sufficient to
//! distinguish shards across different `shard_bits` settings if a
//! restore is somehow restarted with a different choice.
//!
//! `shard_bits` is capped at `8` (the bits of byte 0). Beyond that
//! the partitioning would need bytes 1+, which the driver does not
//! support today and likely never needs to — `1 << 8 = 256`
//! parallel shards already saturate any reasonable disk.

use std::sync::Arc;

use anyhow::ensure;
use sui_types::base_types::ObjectID;
use sui_types::object::Object;
use tracing::info;

use crate::Pipeline;
use crate::RestoreRunner;

/// A source of live objects, queryable by `ObjectID` range.
///
/// The driver issues one [`range`](LiveObjectSource::range) call
/// per shard, in parallel. Implementations are typically thin
/// wrappers around a backing store's range-iterator API.
///
/// The returned iterator is boxed so the trait stays object-safe
/// in shape and so different impls can return different concrete
/// iterator types without leaking the type through the public
/// surface. Per-object boxing cost is one vtable indirection per
/// `next()` call, which is negligible relative to the SST building
/// work per object on the consumer side.
///
/// The trait requires `Send + Sync` on the source itself so the
/// parallel driver can share `&self` across worker threads. The
/// returned iterator is not required to be `Send` — it is
/// created and consumed inside the same worker thread, so its
/// thread-affinity is unobserved by the driver.
pub trait LiveObjectSource: Send + Sync {
    /// Yield objects whose `ObjectID` falls in `[start, end]`
    /// inclusive, in `ObjectID` order. Wrapped or otherwise
    /// non-live entries are filtered before being yielded.
    ///
    /// Per-object iteration errors are surfaced via the
    /// `anyhow::Result` item type; the driver aborts the shard on
    /// the first error.
    fn range<'a>(
        &'a self,
        start: ObjectID,
        end: ObjectID,
    ) -> Box<dyn Iterator<Item = anyhow::Result<Object>> + 'a>;
}

/// Maximum allowed `shard_bits` value. `1 << 8 = 256` shards is
/// already enough to saturate any reasonable disk, and the
/// partitioning fits entirely in byte 0 of the `ObjectID`.
pub const MAX_SHARD_BITS: u8 = 8;

/// Drive a single-pipeline restore against `source` in parallel,
/// using `1 << shard_bits` worker threads.
///
/// Calls [`RestoreRunner::begin`] up front (which returns the set
/// of partitions already complete from a prior run; the driver
/// skips them naturally via
/// [`RestoreRunner::process_shard`]'s built-in already-complete
/// check). Spawns one worker per shard under
/// [`std::thread::scope`]; each worker calls
/// [`source.range`](LiveObjectSource::range) for its byte-0
/// sub-range and hands the iterator to
/// [`RestoreRunner::process_shard`]. On success, calls
/// [`RestoreRunner::finish`].
///
/// `shard_bits` must satisfy `1 <= shard_bits <= MAX_SHARD_BITS`.
pub fn restore_pipeline_from_object_source<P, S>(
    runner: Arc<RestoreRunner<P>>,
    source: &S,
    shard_bits: u8,
) -> anyhow::Result<()>
where
    P: Pipeline,
    S: LiveObjectSource,
{
    ensure!(
        (1..=MAX_SHARD_BITS).contains(&shard_bits),
        "shard_bits must be in 1..={MAX_SHARD_BITS}; got {shard_bits}",
    );

    runner.begin()?;

    let total_shards: u32 = 1 << shard_bits;
    info!(
        pipeline = P::NAME,
        shard_bits, total_shards, "Beginning perpetual-store restore",
    );

    std::thread::scope(|scope| -> anyhow::Result<()> {
        let mut handles = Vec::with_capacity(total_shards as usize);
        for index in 0..total_shards {
            let runner = runner.clone();
            handles.push(scope.spawn(move || -> anyhow::Result<()> {
                let (start, end) = shard_range(index as u8, shard_bits);
                let partition_id = partition_id(index as u8, shard_bits);
                let iter = source.range(start, end);
                runner.process_shard(&partition_id, iter)?;
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("restore worker panicked"))??;
        }
        Ok(())
    })?;

    runner.finish()?;
    Ok(())
}

/// Inclusive `(start, end)` `ObjectID` range covered by shard
/// `index` when the space is split into `1 << shard_bits` pieces.
///
/// With `shard_bits = 5` and `index = 1`, this returns
/// `(ObjectID([0x08, 0, 0, …]), ObjectID([0x0F, 0xFF, …, 0xFF]))`.
pub fn shard_range(index: u8, shard_bits: u8) -> (ObjectID, ObjectID) {
    debug_assert!(
        (1..=MAX_SHARD_BITS).contains(&shard_bits),
        "shard_bits out of range",
    );
    let shift = 8 - shard_bits;

    let mut start = [0u8; ObjectID::LENGTH];
    start[0] = index << shift;

    let mut end = [0u8; ObjectID::LENGTH];
    end[0] = (index << shift) | ((1u8 << shift) - 1);
    for byte in &mut end[1..] {
        *byte = u8::MAX;
    }

    (ObjectID::new(start), ObjectID::new(end))
}

/// `[shard_bits, shard_index]` — two bytes. See module
/// documentation for the rationale.
pub fn partition_id(index: u8, shard_bits: u8) -> [u8; 2] {
    [shard_bits, index]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use bytes::Buf;
    use bytes::BufMut;
    use tempfile::TempDir;

    use super::*;
    use crate::Batch;
    use crate::Db;
    use crate::DbMap;
    use crate::DbOptions;
    use crate::Decode;
    use crate::Encode;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

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
                return Err(DecodeError::msg("unexpected length"));
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
    struct VersionsSchema {
        versions: DbMap<ObjectIdKey, U64Be>,
    }

    impl Schema for VersionsSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            vec![crate::CfDescriptor::new("versions", base_options.clone())]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

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
            for (id, v) in batch {
                write_batch.put(&schema.versions, &ObjectIdKey::new(*id), &U64Be(*v))?;
            }
            Ok(batch.len())
        }
    }

    /// In-memory mock backed by a `BTreeMap`. Useful for testing
    /// the parallel driver without an `AuthorityStore`.
    struct MockSource {
        objects: BTreeMap<ObjectID, Object>,
        /// Records which `(start, end)` ranges were requested, so
        /// tests can assert the driver partitioned correctly.
        ranges_called: Mutex<Vec<(ObjectID, ObjectID)>>,
    }

    impl MockSource {
        fn new(objects: Vec<Object>) -> Self {
            let mut map = BTreeMap::new();
            for o in objects {
                map.insert(o.id(), o);
            }
            Self {
                objects: map,
                ranges_called: Mutex::new(Vec::new()),
            }
        }

        fn ranges_called(&self) -> Vec<(ObjectID, ObjectID)> {
            self.ranges_called.lock().unwrap().clone()
        }
    }

    impl LiveObjectSource for MockSource {
        fn range<'a>(
            &'a self,
            start: ObjectID,
            end: ObjectID,
        ) -> Box<dyn Iterator<Item = anyhow::Result<Object>> + 'a> {
            self.ranges_called.lock().unwrap().push((start, end));
            Box::new(self.objects.range(start..=end).map(|(_, o)| Ok(o.clone())))
        }
    }

    fn obj_with_first_byte(b: u8) -> Object {
        let mut bytes = [0u8; ObjectID::LENGTH];
        bytes[0] = b;
        // Add a per-byte tail so different objects with the same
        // first byte still have distinct ids.
        bytes[ObjectID::LENGTH - 1] = b;
        Object::immutable_with_id_for_testing(ObjectID::new(bytes))
    }

    fn open_db() -> (TempDir, Db, Arc<VersionsSchema>) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<VersionsSchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db, Arc::new(schema))
    }

    #[test]
    fn shard_range_partitions_byte_0_evenly() {
        let (a_start, a_end) = shard_range(0, 5);
        let (b_start, b_end) = shard_range(1, 5);
        // 5 bits = 32 shards = 8 bytes wide per shard.
        assert_eq!(a_start.into_bytes()[0], 0x00);
        assert_eq!(a_end.into_bytes()[0], 0x07);
        assert_eq!(b_start.into_bytes()[0], 0x08);
        assert_eq!(b_end.into_bytes()[0], 0x0F);
    }

    #[test]
    fn shard_range_covers_full_id_space() {
        let (first, _) = shard_range(0, 5);
        let (_, last) = shard_range(31, 5);
        assert_eq!(first.into_bytes()[0], 0x00);
        assert_eq!(last.into_bytes()[0], 0xFF);
        assert_eq!(last.into_bytes()[ObjectID::LENGTH - 1], 0xFF);
    }

    #[test]
    fn restore_drives_all_shards_and_writes_every_object() {
        let (_dir, db, schema) = open_db();

        // 64 objects spread across the byte-0 space.
        let objects: Vec<Object> = (0..=63u8)
            .map(|i| obj_with_first_byte(i.wrapping_mul(4)))
            .collect();
        let expected_ids: BTreeSet<ObjectID> = objects.iter().map(|o| o.id()).collect();
        let source = MockSource::new(objects);

        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            42,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        ));

        restore_pipeline_from_object_source(runner.clone(), &source, 5).unwrap();

        // Every object landed in the CF.
        let got_ids: BTreeSet<ObjectID> = schema
            .versions
            .iter(..)
            .unwrap()
            .map(|r| {
                let (k, _) = r.unwrap();
                ObjectID::new(k.0)
            })
            .collect();
        assert_eq!(got_ids, expected_ids);

        // The driver issued one `range` call per shard.
        let ranges = source.ranges_called();
        assert_eq!(ranges.len(), 32);

        // Restore state is Complete.
        match db.restore_state("versions").unwrap() {
            Some(crate::RestoreState::Complete { restored_at }) => {
                assert_eq!(restored_at, 42);
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn restore_with_one_shard_uses_full_id_range() {
        let (_dir, db, schema) = open_db();
        let objects = vec![obj_with_first_byte(0), obj_with_first_byte(255)];
        let source = MockSource::new(objects);

        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            1,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        ));

        // shard_bits = 1 → 2 shards covering [0x00..0x7F] and
        // [0x80..0xFF].
        restore_pipeline_from_object_source(runner, &source, 1).unwrap();
        let count = schema.versions.iter(..).unwrap().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn restore_with_empty_source_still_marks_complete() {
        let (_dir, db, schema) = open_db();
        let source = MockSource::new(vec![]);
        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            7,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        ));

        restore_pipeline_from_object_source(runner, &source, 3).unwrap();
        match db.restore_state("versions").unwrap() {
            Some(crate::RestoreState::Complete { restored_at }) => assert_eq!(restored_at, 7),
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn restore_rejects_shard_bits_zero() {
        let (_dir, db, schema) = open_db();
        let source = MockSource::new(vec![]);
        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            1,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        ));
        let err = restore_pipeline_from_object_source(runner, &source, 0).unwrap_err();
        assert!(format!("{err:#}").contains("shard_bits"));
    }

    #[test]
    fn restore_rejects_shard_bits_too_large() {
        let (_dir, db, schema) = open_db();
        let source = MockSource::new(vec![]);
        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            1,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        ));
        let err = restore_pipeline_from_object_source(runner, &source, 9).unwrap_err();
        assert!(format!("{err:#}").contains("shard_bits"));
    }

    #[test]
    fn partition_id_layout() {
        assert_eq!(partition_id(0, 5), [5, 0]);
        assert_eq!(partition_id(31, 5), [5, 31]);
        assert_eq!(partition_id(7, 3), [3, 7]);
    }

    #[test]
    fn resume_skips_already_completed_partitions() {
        let (_dir, db, schema) = open_db();
        let objects = vec![obj_with_first_byte(0x00), obj_with_first_byte(0xFF)];
        let source = MockSource::new(objects);

        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            42,
            staging.path().to_path_buf(),
            rocksdb::Options::default(),
        ));

        // Pre-mark partition id [1, 0] (shard_bits=1, index=0) as
        // complete: simulates a crash after the first shard
        // ingested.
        let mut done = BTreeSet::new();
        done.insert(partition_id(0, 1).to_vec());
        db.set_restore_state(
            "versions",
            &crate::RestoreState::InProgress {
                target_checkpoint: 42,
                partitions_complete: done,
            },
        )
        .unwrap();

        restore_pipeline_from_object_source(runner, &source, 1).unwrap();

        // Only the high-half object (0xFF) is in the CF — the
        // low-half (0x00) was marked complete so its shard was
        // skipped.
        let count = schema.versions.iter(..).unwrap().count();
        assert_eq!(count, 1);
    }
}
