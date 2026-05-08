// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Consistent reads against a captured snapshot.
//!
//! [`SnapshotHandle`] is a cheap-to-clone handle returned by
//! [`Db::at_snapshot`](crate::Db::at_snapshot) that exposes the same
//! read and iteration surface as [`DbMap`] but routes every call
//! through a [`rocksdb::Snapshot`] captured at a specific checkpoint.
//! Reads through the same handle (and through clones of it) all see
//! the database state at the moment [`Db::take_snapshot`] was called,
//! regardless of any writes that occur afterwards.
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
//! Iteration methods on a handle return iterators whose lifetime is
//! tied to both the handle and the [`DbMap`] they iterate over,
//! since the iterator's underlying [`rocksdb::ReadOptions`] holds a
//! raw pointer to the snapshot. The handle must outlive any iterator
//! it produces.
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
//! struct MySchema {
//!     items: DbMap<U64Be, U64Be>,
//! }
//!
//! impl Schema for MySchema {
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
//! // The snapshot still sees the pre-mutation value.
//! let snap = db.at_snapshot(1).unwrap();
//! assert_eq!(
//!     snap.get(&schema.items, &U64Be(1)).unwrap(),
//!     Some(U64Be(100))
//! );
//! // The current view sees the new value.
//! assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(999)));
//! ```

use std::fmt;
use std::ops::RangeBounds;
use std::sync::Arc;

use bytes::Bytes;
use rocksdb::ReadOptions;

use crate::Decode;
use crate::Encode;
use crate::db::Db;
use crate::db::SnapshotEntry;
use crate::error::Error;
use crate::iter::Iter;
use crate::iter::RevIter;
use crate::iter::prefix_to_byte_bounds;
use crate::iter::range_to_byte_bounds;
use crate::map::DbMap;

/// A cheap-to-clone handle to a single snapshot of the database.
///
/// Returned by [`Db::at_snapshot`](crate::Db::at_snapshot). Clones
/// share the same underlying snapshot; cloning is an `Arc`
/// increment and a small struct copy.
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

    fn read_options(&self) -> ReadOptions {
        let mut opts = ReadOptions::default();
        opts.set_snapshot(self.entry.as_snapshot());
        opts
    }

    /// Read and decode the value for `key` against this snapshot.
    pub fn get<K, V>(&self, map: &DbMap<K, V>, key: &K) -> Result<Option<V>, Error>
    where
        K: Encode,
        V: Decode,
    {
        map.get_with_opts(key, &self.read_options())
    }

    /// Read the raw bytes for `key` against this snapshot.
    pub fn get_raw<K, V>(&self, map: &DbMap<K, V>, key: &K) -> Result<Option<Bytes>, Error>
    where
        K: Encode,
    {
        map.get_raw_with_opts(key, &self.read_options())
    }

    /// Batched read and decode against this snapshot.
    pub fn multi_get<'k, K, V, I>(
        &self,
        map: &DbMap<K, V>,
        keys: I,
    ) -> Result<Vec<Result<Option<V>, Error>>, Error>
    where
        K: Encode + 'k,
        V: Decode,
        I: IntoIterator<Item = &'k K>,
    {
        map.multi_get_with_opts(keys, &self.read_options())
    }

    /// Batched raw-bytes read against this snapshot.
    pub fn multi_get_raw<'k, K, V, I>(
        &self,
        map: &DbMap<K, V>,
        keys: I,
    ) -> Result<Vec<Result<Option<Bytes>, Error>>, Error>
    where
        K: Encode + 'k,
        I: IntoIterator<Item = &'k K>,
    {
        map.multi_get_raw_with_opts(keys, &self.read_options())
    }

    /// Test whether `key` is present in `map` at this snapshot.
    pub fn contains_key<K, V>(&self, map: &DbMap<K, V>, key: &K) -> Result<bool, Error>
    where
        K: Encode,
    {
        map.contains_key_with_opts(key, &self.read_options())
    }

    /// Batched counterpart to [`contains_key`](Self::contains_key).
    pub fn multi_contains_keys<'k, K, V, I>(
        &self,
        map: &DbMap<K, V>,
        keys: I,
    ) -> Result<Vec<bool>, Error>
    where
        K: Encode + 'k,
        I: IntoIterator<Item = &'k K>,
    {
        map.multi_contains_keys_with_opts(keys, &self.read_options())
    }

    /// Forward iteration against this snapshot, bounded by `range`.
    ///
    /// The returned iterator borrows from `self`, so the snapshot
    /// handle must outlive the iterator. Cloning the handle before
    /// iterating gives a separately-owned alias if the caller needs
    /// to keep handles around for later use.
    pub fn iter<'s, K, V>(
        &'s self,
        map: &'s DbMap<K, V>,
        range: impl RangeBounds<K>,
    ) -> Result<Iter<'s, K, V>, Error>
    where
        K: Encode + Decode,
        V: Decode,
    {
        let (lower, upper) = range_to_byte_bounds(&range)?;
        map.iter_forward(lower, upper, self.read_options())
    }

    /// Forward iteration against this snapshot, restricted to keys
    /// whose encoding begins with `prefix`'s encoding. See
    /// [`DbMap::iter_prefix`] for the prefix contract.
    pub fn iter_prefix<'s, K, V>(
        &'s self,
        map: &'s DbMap<K, V>,
        prefix: &impl Encode,
    ) -> Result<Iter<'s, K, V>, Error>
    where
        K: Encode + Decode,
        V: Decode,
    {
        let (lower, upper) = prefix_to_byte_bounds(prefix)?;
        map.iter_forward(lower, upper, self.read_options())
    }

    /// Reverse iteration against this snapshot, bounded by `range`.
    pub fn iter_rev<'s, K, V>(
        &'s self,
        map: &'s DbMap<K, V>,
        range: impl RangeBounds<K>,
    ) -> Result<RevIter<'s, K, V>, Error>
    where
        K: Encode + Decode,
        V: Decode,
    {
        let (lower, upper) = range_to_byte_bounds(&range)?;
        map.iter_reverse(lower, upper, self.read_options())
    }

    /// Reverse iteration against this snapshot, restricted to keys
    /// whose encoding begins with `prefix`'s encoding.
    pub fn iter_rev_prefix<'s, K, V>(
        &'s self,
        map: &'s DbMap<K, V>,
        prefix: &impl Encode,
    ) -> Result<RevIter<'s, K, V>, Error>
    where
        K: Encode + Decode,
        V: Decode,
    {
        let (lower, upper) = prefix_to_byte_bounds(prefix)?;
        map.iter_reverse(lower, upper, self.read_options())
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
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct U64Be(u64);

    impl Encode for U64Be {
        fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
            buf.extend_from_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    impl Decode for U64Be {
        fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| DecodeError::msg("expected 8 bytes"))?;
            Ok(Self(u64::from_be_bytes(arr)))
        }
    }

    #[derive(Debug)]
    struct TestSchema {
        items: DbMap<U64Be, U64Be>,
    }

    impl Schema for TestSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
            vec![("items", base_options.clone())]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                items: DbMap::new(db.clone(), "items")?,
            })
        }
    }

    use crate::error::OpenError;

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
            latest.get(&schema.items, &U64Be(1)).unwrap(),
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
            snap.get(&schema.items, &U64Be(1)).unwrap(),
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
        assert!(snap.get(&schema.items, &U64Be(1)).unwrap().is_none());
    }

    #[test]
    fn snapshot_get_raw_against_pre_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        put(&db, &schema, 1, 999);
        let snap = db.at_snapshot(1).unwrap();
        let bytes = snap
            .get_raw(&schema.items, &U64Be(1))
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
        assert!(snap.contains_key(&schema.items, &U64Be(1)).unwrap());
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
        let results = snap.multi_get(&schema.items, keys.iter()).unwrap();
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
        let collected: Vec<_> = snap
            .iter(&schema.items, ..)
            .unwrap()
            .map(Result::unwrap)
            .collect();
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
        let collected: Vec<_> = snap
            .iter_rev(&schema.items, ..)
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
        let collected: Vec<_> = snap
            .iter_prefix(&schema.items, &U64Be(1))
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
            snap.get(&schema.items, &U64Be(1)).unwrap(),
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
            snap_a.get(&schema.items, &U64Be(1)).unwrap(),
            Some(U64Be(100)),
        );
        assert_eq!(
            snap_b.get(&schema.items, &U64Be(1)).unwrap(),
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
        // We re-open a temporary DbMap pointed at the same CF to
        // exercise reads against the live Db.
        let items: DbMap<U64Be, U64Be> = DbMap::new(db.clone(), "items").unwrap();
        drop(schema);
        assert_eq!(snap.get(&items, &U64Be(1)).unwrap(), Some(U64Be(100)));
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
            snap.get(&schema.items, &U64Be(1)).unwrap(),
            Some(U64Be(200)),
        );
    }
}
