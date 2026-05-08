// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Consistent reads against a captured snapshot.
//!
//! [`SnapshotHandle`] is a cheap-to-clone token returned by
//! [`Db::at_snapshot`](crate::Db::at_snapshot) (or
//! [`Db::latest_snapshot`](crate::Db::latest_snapshot)). It is the
//! borrow point used to construct snapshot-bound reads via
//! [`DbMap::at`](crate::DbMap::at) (single CF) or
//! [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at) (whole
//! schema). The handle itself does not expose read methods — reads
//! always go through a [`DbMap<_, _, Snapshot<'_>>`](crate::DbMap)
//! produced by re-binding.
//!
//! # Lifetimes and ownership
//!
//! The handle co-owns an [`Arc<Db>`] and an [`Arc<SnapshotEntry>`]
//! ([`SnapshotEntry`](crate::db::SnapshotEntry) is private). As long
//! as a handle (or any clone of it) exists, the underlying snapshot
//! is kept alive even if [`Db::drop_snapshot`](crate::Db::drop_snapshot)
//! has been called for that checkpoint or the snapshot has been
//! evicted from the buffer by capacity pressure.
//!
//! Snapshot-bound [`DbMap`](crate::DbMap)s constructed from a handle
//! borrow from it; the handle must outlive any such re-binding (and
//! the iterators those re-bindings produce).
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//!
//! use sui_consistent_store::Db;
//! use sui_consistent_store::DbMap;
//! use sui_consistent_store::DbOptions;
//! use sui_consistent_store::Decode;
//! use sui_consistent_store::Encode;
//! use sui_consistent_store::Live;
//! use sui_consistent_store::Reader;
//! use sui_consistent_store::Schema;
//! use sui_consistent_store::error::DecodeError;
//! use sui_consistent_store::error::EncodeError;
//! use sui_consistent_store::error::OpenError;
//!
//! #[derive(Debug, PartialEq, Eq)]
//! struct U64Be(u64);
//!
//! impl Encode for U64Be {
//!     fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
//!         buf.extend_from_slice(&self.0.to_be_bytes());
//!         Ok(())
//!     }
//! }
//!
//! impl Decode for U64Be {
//!     fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
//!         let arr: [u8; 8] = bytes
//!             .try_into()
//!             .map_err(|_| DecodeError::msg("expected 8 bytes"))?;
//!         Ok(Self(u64::from_be_bytes(arr)))
//!     }
//! }
//!
//! struct MySchema<R: Reader = Live> {
//!     items: DbMap<U64Be, U64Be, R>,
//! }
//!
//! impl Schema for MySchema<Live> {
//!     fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
//!         vec![("items", base_options.clone())]
//!     }
//!
//!     fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
//!         Ok(Self {
//!             items: DbMap::new(db.clone(), "items")?,
//!         })
//!     }
//! }
//!
//! let dir = tempfile::tempdir().unwrap();
//! let (db, schema) = Db::open::<MySchema>(dir.path(), DbOptions::default()).unwrap();
//!
//! // Write the initial state, then take a snapshot at checkpoint 1.
//! let mut batch = db.batch();
//! batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
//! batch.commit().unwrap();
//! db.take_snapshot(1);
//!
//! // Mutate after the snapshot.
//! let mut batch = db.batch();
//! batch.put(&schema.items, &U64Be(1), &U64Be(999)).unwrap();
//! batch.commit().unwrap();
//!
//! // Re-bind the items map at the snapshot.
//! let snap = db.at_snapshot(1).unwrap();
//! let items_at_snap = schema.items.at(&snap);
//! assert_eq!(items_at_snap.get(&U64Be(1)).unwrap(), Some(U64Be(100)));
//! // The live binding sees the new value.
//! assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(999)));
//! ```

use std::fmt;
use std::sync::Arc;

use crate::db::Db;
use crate::db::SnapshotEntry;

/// A cheap-to-clone token referencing a single snapshot of the
/// database.
///
/// Returned by [`Db::at_snapshot`](crate::Db::at_snapshot) and
/// [`Db::latest_snapshot`](crate::Db::latest_snapshot). Pass a
/// reference to a [`SnapshotHandle`] to
/// [`DbMap::at`](crate::DbMap::at) or
/// [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at) to obtain
/// snapshot-bound read handles. Clones share the same underlying
/// snapshot; cloning is an `Arc` increment and a small struct copy.
pub struct SnapshotHandle {
    // Field declaration order is load-bearing: `entry` must drop
    // before `_db` so that the contained `rocksdb::Snapshot`
    // releases its borrow on `Db::inner` before the `Arc<Db>` ref
    // is decremented.
    entry: Arc<SnapshotEntry>,
    _db: Arc<Db>,
    checkpoint: u64,
}

impl SnapshotHandle {
    pub(crate) fn new(db: Arc<Db>, entry: Arc<SnapshotEntry>, checkpoint: u64) -> Self {
        Self {
            entry,
            _db: db,
            checkpoint,
        }
    }

    /// The checkpoint number this snapshot was taken at.
    pub fn checkpoint(&self) -> u64 {
        self.checkpoint
    }

    /// The shared database handle this snapshot is taken against.
    /// Used by the [`Reader`](crate::Reader) implementation for
    /// [`Snapshot`](crate::Snapshot) to look up column-family
    /// handles.
    pub(crate) fn db(&self) -> &Arc<Db> {
        &self._db
    }

    /// The retained [`SnapshotEntry`] backing this handle. Used by
    /// the [`Reader`](crate::Reader) implementation for
    /// [`Snapshot`](crate::Snapshot) to install the snapshot pointer
    /// on a fresh [`ReadOptions`](rocksdb::ReadOptions).
    pub(crate) fn entry(&self) -> &Arc<SnapshotEntry> {
        &self.entry
    }
}

impl Clone for SnapshotHandle {
    fn clone(&self) -> Self {
        Self {
            entry: self.entry.clone(),
            _db: self._db.clone(),
            checkpoint: self.checkpoint,
        }
    }
}

impl fmt::Debug for SnapshotHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SnapshotHandle")
            .field("checkpoint", &self.checkpoint)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::DbOptions;
    use crate::Live;
    use crate::Reader;
    use crate::Schema;
    use crate::SchemaAtSnapshot;
    use crate::Snapshot;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;
    use crate::map::DbMap;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct U64Be(u64);

    impl crate::Encode for U64Be {
        fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
            buf.extend_from_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    impl crate::Decode for U64Be {
        fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| DecodeError::msg("expected 8 bytes"))?;
            Ok(Self(u64::from_be_bytes(arr)))
        }
    }

    #[derive(Debug)]
    struct TestSchema<R: Reader = Live> {
        items: DbMap<U64Be, U64Be, R>,
    }

    impl Schema for TestSchema<Live> {
        fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
            vec![("items", base_options.clone())]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                items: DbMap::new(db.clone(), "items")?,
            })
        }
    }

    impl SchemaAtSnapshot for TestSchema<Live> {
        type At<'s> = TestSchema<Snapshot<'s>>;
        fn at<'s>(&'s self, snap: &'s SnapshotHandle) -> Self::At<'s> {
            TestSchema {
                items: self.items.at(snap),
            }
        }
    }

    fn open_with_capacity(capacity: usize) -> (TempDir, Arc<Db>, TestSchema) {
        let dir = TempDir::new().unwrap();
        let opts = DbOptions {
            snapshot_capacity: capacity,
            ..DbOptions::default()
        };
        let (db, schema) = Db::open::<TestSchema>(dir.path(), opts).unwrap();
        (dir, db, schema)
    }

    fn open() -> (TempDir, Arc<Db>, TestSchema) {
        open_with_capacity(32)
    }

    fn put(db: &Arc<Db>, schema: &TestSchema, key: u64, value: u64) {
        let mut batch = db.batch();
        batch
            .put(&schema.items, &U64Be(key), &U64Be(value))
            .unwrap();
        batch.commit().unwrap();
    }

    #[test]
    fn at_snapshot_returns_none_when_no_snapshot_taken() {
        let (_dir, db, _schema) = open();
        assert!(db.at_snapshot(0).is_none());
    }

    #[test]
    fn latest_snapshot_is_none_when_empty() {
        let (_dir, db, _schema) = open();
        assert!(db.latest_snapshot().is_none());
    }

    #[test]
    fn latest_snapshot_returns_highest_checkpoint() {
        let (_dir, db, _schema) = open();
        db.take_snapshot(3);
        db.take_snapshot(10);
        db.take_snapshot(5);
        let latest = db.latest_snapshot().expect("latest should exist");
        assert_eq!(latest.checkpoint(), 10);
    }

    #[test]
    fn latest_snapshot_after_eviction_reflects_remaining() {
        let (_dir, db, _schema) = open_with_capacity(2);
        db.take_snapshot(1);
        db.take_snapshot(2);
        db.take_snapshot(3);
        // Capacity 2 evicts checkpoint 1; latest is now 3.
        let latest = db.latest_snapshot().expect("latest should exist");
        assert_eq!(latest.checkpoint(), 3);
    }

    #[test]
    fn latest_snapshot_reads_pre_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        put(&db, &schema, 1, 999);
        let latest = db.latest_snapshot().unwrap();
        assert_eq!(
            schema.items.at(&latest).get(&U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
    }

    #[test]
    fn snapshot_range_is_none_when_empty() {
        let (_dir, db, _schema) = open();
        assert!(db.snapshot_range().is_none());
    }

    #[test]
    fn take_then_at_returns_handle() {
        let (_dir, db, _schema) = open();
        db.take_snapshot(7);
        let handle = db.at_snapshot(7).expect("handle should exist");
        assert_eq!(handle.checkpoint(), 7);
    }

    #[test]
    fn snapshot_range_reflects_taken_snapshots() {
        let (_dir, db, _schema) = open();
        db.take_snapshot(3);
        db.take_snapshot(10);
        db.take_snapshot(5);
        assert_eq!(db.snapshot_range(), Some(3..=10));
    }

    #[test]
    fn snapshot_capacity_evicts_oldest() {
        let (_dir, db, _schema) = open_with_capacity(2);
        db.take_snapshot(1);
        db.take_snapshot(2);
        db.take_snapshot(3);
        assert!(db.at_snapshot(1).is_none());
        assert!(db.at_snapshot(2).is_some());
        assert!(db.at_snapshot(3).is_some());
        assert_eq!(db.snapshot_range(), Some(2..=3));
    }

    #[test]
    fn drop_snapshot_removes_from_buffer() {
        let (_dir, db, _schema) = open();
        db.take_snapshot(5);
        assert!(db.drop_snapshot(5));
        assert!(db.at_snapshot(5).is_none());
        // Dropping a missing snapshot returns false.
        assert!(!db.drop_snapshot(5));
    }

    #[test]
    fn snapshot_sees_state_at_take_time() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        // Mutate after taking the snapshot.
        put(&db, &schema, 1, 999);

        let snap = db.at_snapshot(1).unwrap();
        assert_eq!(
            schema.items.at(&snap).get(&U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
        assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(999)));
    }

    #[test]
    fn snapshot_does_not_see_keys_inserted_after_take() {
        let (_dir, db, schema) = open();
        db.take_snapshot(0);
        put(&db, &schema, 1, 100);
        let snap = db.at_snapshot(0).unwrap();
        assert!(schema.items.at(&snap).get(&U64Be(1)).unwrap().is_none());
    }

    #[test]
    fn snapshot_get_raw_against_pre_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        put(&db, &schema, 1, 999);
        let snap = db.at_snapshot(1).unwrap();
        let bytes = schema
            .items
            .at(&snap)
            .get_raw(&U64Be(1))
            .unwrap()
            .expect("value should exist in snapshot");
        assert_eq!(&bytes[..], &100u64.to_be_bytes());
    }

    #[test]
    fn snapshot_contains_key_reflects_pre_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        // Mutation after snapshot must not affect snapshot's view.
        let mut batch = db.batch();
        batch.delete(&schema.items, &U64Be(1)).unwrap();
        batch.commit().unwrap();
        let snap = db.at_snapshot(1).unwrap();
        assert!(schema.items.at(&snap).contains_key(&U64Be(1)).unwrap());
        assert!(!schema.items.contains_key(&U64Be(1)).unwrap());
    }

    #[test]
    fn snapshot_multi_get_against_pre_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 3, 30);
        db.take_snapshot(1);
        // After-snapshot writes should not affect snapshot reads.
        put(&db, &schema, 2, 20);
        put(&db, &schema, 1, 999);

        let snap = db.at_snapshot(1).unwrap();
        let keys = [U64Be(1), U64Be(2), U64Be(3)];
        let results = schema.items.at(&snap).multi_get(keys.iter()).unwrap();
        assert_eq!(results[0].as_ref().unwrap(), &Some(U64Be(10)));
        assert_eq!(results[1].as_ref().unwrap(), &None);
        assert_eq!(results[2].as_ref().unwrap(), &Some(U64Be(30)));
    }

    #[test]
    fn snapshot_iter_yields_pre_snapshot_state_in_order() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 3, 30);
        db.take_snapshot(1);
        put(&db, &schema, 2, 20);

        let snap = db.at_snapshot(1).unwrap();
        let snap_items = schema.items.at(&snap);
        let collected: Vec<_> = snap_items.iter(..).unwrap().map(Result::unwrap).collect();
        assert_eq!(
            collected,
            vec![(U64Be(1), U64Be(10)), (U64Be(3), U64Be(30))],
        );
    }

    #[test]
    fn snapshot_iter_rev_yields_pre_snapshot_state_reversed() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 2, 20);
        db.take_snapshot(1);

        let snap = db.at_snapshot(1).unwrap();
        let snap_items = schema.items.at(&snap);
        let collected: Vec<_> = snap_items
            .iter_rev(..)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            collected,
            vec![(U64Be(2), U64Be(20)), (U64Be(1), U64Be(10))],
        );
    }

    #[test]
    fn snapshot_iter_prefix_filters_against_pre_snapshot_state() {
        let (_dir, db, schema) = open();
        // Use the same encoding as U64Be for prefix; an 8-byte
        // prefix matches an exact key, demonstrating prefix
        // iteration on the snapshot path.
        put(&db, &schema, 1, 10);
        put(&db, &schema, 2, 20);
        db.take_snapshot(1);
        put(&db, &schema, 1, 999);

        let snap = db.at_snapshot(1).unwrap();
        let snap_items = schema.items.at(&snap);
        let collected: Vec<_> = snap_items
            .iter_prefix(&U64Be(1))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(collected, vec![(U64Be(1), U64Be(10))]);
    }

    #[test]
    fn snapshot_handle_survives_drop_snapshot() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        let snap = db.at_snapshot(1).unwrap();
        // Remove the snapshot from the buffer; the handle still works.
        assert!(db.drop_snapshot(1));
        assert!(db.at_snapshot(1).is_none());
        assert_eq!(
            schema.items.at(&snap).get(&U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
    }

    #[test]
    fn snapshot_handle_clones_share_underlying_snapshot() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        let snap_a = db.at_snapshot(1).unwrap();
        let snap_b = snap_a.clone();
        // Drop the buffer ref; both clones still see the same state.
        db.drop_snapshot(1);
        put(&db, &schema, 1, 999);
        assert_eq!(
            schema.items.at(&snap_a).get(&U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
        assert_eq!(
            schema.items.at(&snap_b).get(&U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
    }

    #[test]
    fn snapshot_handle_outlives_schema() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        let snap = db.at_snapshot(1).unwrap();
        // Schema (and its DbMap) drops; the snapshot handle still
        // co-owns Arc<Db>, so the underlying database is alive.
        // We re-open a temporary DbMap pointed at the same CF and
        // re-bind it at the snapshot to exercise reads.
        let items: DbMap<U64Be, U64Be> = DbMap::new(db.clone(), "items").unwrap();
        drop(schema);
        assert_eq!(items.at(&snap).get(&U64Be(1)).unwrap(), Some(U64Be(100)));
    }

    #[test]
    fn taking_snapshot_at_existing_checkpoint_replaces() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        put(&db, &schema, 1, 200);
        // Re-take at the same checkpoint; the new snapshot reflects
        // the updated state.
        db.take_snapshot(1);
        let snap = db.at_snapshot(1).unwrap();
        assert_eq!(
            schema.items.at(&snap).get(&U64Be(1)).unwrap(),
            Some(U64Be(200)),
        );
    }

    /// Demonstrates the whole-schema re-binding pattern via
    /// `SchemaAtSnapshot::at`. The projected schema's reads see the
    /// captured snapshot state for every CF.
    #[test]
    fn schema_at_snapshot_projects_all_fields() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        put(&db, &schema, 1, 999);

        let snap = db.at_snapshot(1).unwrap();
        let snap_schema = schema.at(&snap);
        assert_eq!(
            snap_schema.items.get(&U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
    }
}
