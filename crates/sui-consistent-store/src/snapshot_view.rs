// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A typed view that pairs a [`SnapshotHandle`] with a single
//! [`DbMap`] so consecutive snapshot-bound reads against the same
//! column family don't have to repeat the map argument.
//!
//! [`SnapshotView`] is constructed via [`SnapshotHandle::view`] or
//! [`DbMap::at`]. It holds two references and is `Copy`, so it is
//! free to construct, free to clone, and never allocates. All
//! methods delegate to the corresponding method on
//! [`SnapshotHandle`], passing the bound map through transparently.
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
//! db.take_snapshot(1);
//!
//! let snap = db.latest_snapshot().expect("snapshot taken");
//! let items = snap.view(&schema.items);
//!
//! // All reads below are pinned to the same snapshot; the map
//! // argument doesn't need to repeat.
//! assert!(items.get(&U64Be(1)).unwrap().is_none());
//! assert!(items.get(&U64Be(2)).unwrap().is_none());
//! for entry in items.iter(..).unwrap() {
//!     let (_k, _v) = entry.unwrap();
//! }
//! ```

use std::fmt;
use std::ops::RangeBounds;

use bytes::Bytes;

use crate::Decode;
use crate::Encode;
use crate::error::Error;
use crate::iter::Iter;
use crate::iter::RevIter;
use crate::map::DbMap;
use crate::snapshot::SnapshotHandle;

/// A snapshot-bound view of a single typed column family.
///
/// Holds a reference to a [`SnapshotHandle`] and to a [`DbMap`].
/// Both methods on [`SnapshotHandle`] for snapshot-bound reads have
/// a counterpart here that drops the `&DbMap` argument. Use
/// [`SnapshotHandle::view`] (or [`DbMap::at`]) to construct one.
///
/// `SnapshotView` is `Copy` (it's two references); cloning is free.
/// The view's lifetime is the shorter of the snapshot handle's and
/// the map's borrow.
pub struct SnapshotView<'s, K, V> {
    snapshot: &'s SnapshotHandle,
    map: &'s DbMap<K, V>,
}

impl<'s, K, V> SnapshotView<'s, K, V> {
    pub(crate) fn new(snapshot: &'s SnapshotHandle, map: &'s DbMap<K, V>) -> Self {
        Self { snapshot, map }
    }

    /// The checkpoint number this view's snapshot was taken at.
    pub fn checkpoint(&self) -> u64 {
        self.snapshot.checkpoint()
    }

    /// The underlying [`SnapshotHandle`] this view is bound to.
    pub fn snapshot(&self) -> &'s SnapshotHandle {
        self.snapshot
    }

    /// The underlying [`DbMap`] this view is bound to.
    pub fn map(&self) -> &'s DbMap<K, V> {
        self.map
    }
}

impl<'s, K, V> SnapshotView<'s, K, V>
where
    K: Encode,
    V: Decode,
{
    /// Read and decode the value for `key` against the bound
    /// snapshot.
    pub fn get(&self, key: &K) -> Result<Option<V>, Error> {
        self.snapshot.get(self.map, key)
    }

    /// Batched counterpart to [`get`](Self::get).
    pub fn multi_get<'k, I>(&self, keys: I) -> Result<Vec<Result<Option<V>, Error>>, Error>
    where
        I: IntoIterator<Item = &'k K>,
        K: 'k,
    {
        self.snapshot.multi_get(self.map, keys)
    }
}

impl<'s, K, V> SnapshotView<'s, K, V>
where
    K: Encode,
{
    /// Read the raw bytes for `key` against the bound snapshot.
    pub fn get_raw(&self, key: &K) -> Result<Option<Bytes>, Error> {
        self.snapshot.get_raw(self.map, key)
    }

    /// Batched counterpart to [`get_raw`](Self::get_raw).
    pub fn multi_get_raw<'k, I>(&self, keys: I) -> Result<Vec<Result<Option<Bytes>, Error>>, Error>
    where
        I: IntoIterator<Item = &'k K>,
        K: 'k,
    {
        self.snapshot.multi_get_raw(self.map, keys)
    }

    /// Test whether `key` is present in the bound map at the bound
    /// snapshot.
    pub fn contains_key(&self, key: &K) -> Result<bool, Error> {
        self.snapshot.contains_key(self.map, key)
    }

    /// Batched counterpart to [`contains_key`](Self::contains_key).
    pub fn multi_contains_keys<'k, I>(&self, keys: I) -> Result<Vec<bool>, Error>
    where
        I: IntoIterator<Item = &'k K>,
        K: 'k,
    {
        self.snapshot.multi_contains_keys(self.map, keys)
    }
}

impl<'s, K, V> SnapshotView<'s, K, V>
where
    K: Encode + Decode,
    V: Decode,
{
    /// Forward iteration over the bound map at the bound snapshot,
    /// restricted to keys within `range`.
    pub fn iter(&self, range: impl RangeBounds<K>) -> Result<Iter<'s, K, V>, Error> {
        self.snapshot.iter(self.map, range)
    }

    /// Forward iteration restricted to keys whose encoding begins
    /// with `prefix`'s encoding. See [`DbMap::iter_prefix`] for the
    /// prefix contract.
    pub fn iter_prefix(&self, prefix: &impl Encode) -> Result<Iter<'s, K, V>, Error> {
        self.snapshot.iter_prefix(self.map, prefix)
    }

    /// Reverse iteration over the bound map at the bound snapshot,
    /// restricted to keys within `range`.
    pub fn iter_rev(&self, range: impl RangeBounds<K>) -> Result<RevIter<'s, K, V>, Error> {
        self.snapshot.iter_rev(self.map, range)
    }

    /// Reverse iteration restricted to keys whose encoding begins
    /// with `prefix`'s encoding.
    pub fn iter_rev_prefix(&self, prefix: &impl Encode) -> Result<RevIter<'s, K, V>, Error> {
        self.snapshot.iter_rev_prefix(self.map, prefix)
    }
}

impl<K, V> Clone for SnapshotView<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, V> Copy for SnapshotView<'_, K, V> {}

impl<K, V> fmt::Debug for SnapshotView<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SnapshotView")
            .field("checkpoint", &self.snapshot.checkpoint())
            .field("cf_name", &self.map.cf_name())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::Db;
    use crate::DbOptions;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

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

    fn open() -> (TempDir, Arc<Db>, TestSchema) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db, schema)
    }

    fn put(db: &Arc<Db>, schema: &TestSchema, key: u64, value: u64) {
        let mut batch = db.batch();
        batch
            .put(&schema.items, &U64Be(key), &U64Be(value))
            .unwrap();
        batch.commit().unwrap();
    }

    #[test]
    fn view_get_returns_value_at_snapshot() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        put(&db, &schema, 1, 999);

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        assert_eq!(items.checkpoint(), 1);
        assert_eq!(items.get(&U64Be(1)).unwrap(), Some(U64Be(100)));
    }

    #[test]
    fn view_get_raw_returns_pre_snapshot_bytes() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 7, 700);
        db.take_snapshot(1);
        put(&db, &schema, 7, 0);

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        let bytes = items.get_raw(&U64Be(7)).unwrap().unwrap();
        assert_eq!(&bytes[..], &700u64.to_be_bytes());
    }

    #[test]
    fn view_multi_get_reflects_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 3, 30);
        db.take_snapshot(1);
        put(&db, &schema, 2, 20);

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        let keys = [U64Be(1), U64Be(2), U64Be(3)];
        let results = items.multi_get(keys.iter()).unwrap();
        assert_eq!(results[0].as_ref().unwrap(), &Some(U64Be(10)));
        assert_eq!(results[1].as_ref().unwrap(), &None);
        assert_eq!(results[2].as_ref().unwrap(), &Some(U64Be(30)));
    }

    #[test]
    fn view_contains_key_against_snapshot() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        db.take_snapshot(1);
        let mut batch = db.batch();
        batch.delete(&schema.items, &U64Be(1)).unwrap();
        batch.commit().unwrap();

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        assert!(items.contains_key(&U64Be(1)).unwrap());
    }

    #[test]
    fn view_iter_walks_snapshot_state() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 2, 20);
        db.take_snapshot(1);
        put(&db, &schema, 3, 30);

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        let collected: Vec<_> = items.iter(..).unwrap().map(Result::unwrap).collect();
        assert_eq!(
            collected,
            vec![(U64Be(1), U64Be(10)), (U64Be(2), U64Be(20))],
        );
    }

    #[test]
    fn view_iter_prefix_filters() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 2, 20);
        db.take_snapshot(1);

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        let collected: Vec<_> = items
            .iter_prefix(&U64Be(1))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(collected, vec![(U64Be(1), U64Be(10))]);
    }

    #[test]
    fn view_iter_rev_walks_snapshot_state_reversed() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 10);
        put(&db, &schema, 2, 20);
        db.take_snapshot(1);

        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        let collected: Vec<_> = items.iter_rev(..).unwrap().map(Result::unwrap).collect();
        assert_eq!(
            collected,
            vec![(U64Be(2), U64Be(20)), (U64Be(1), U64Be(10))],
        );
    }

    #[test]
    fn dbmap_at_returns_equivalent_view() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);

        let snap = db.at_snapshot(1).unwrap();
        let view_a = snap.view(&schema.items);
        let view_b = schema.items.at(&snap);
        assert_eq!(
            view_a.get(&U64Be(1)).unwrap(),
            view_b.get(&U64Be(1)).unwrap()
        );
        assert_eq!(view_a.checkpoint(), view_b.checkpoint());
    }

    #[test]
    fn view_is_copy() {
        let (_dir, db, schema) = open();
        put(&db, &schema, 1, 100);
        db.take_snapshot(1);
        let snap = db.at_snapshot(1).unwrap();
        let items = snap.view(&schema.items);
        // Copy via assignment to a fresh binding.
        let copy = items;
        assert_eq!(items.get(&U64Be(1)).unwrap(), copy.get(&U64Be(1)).unwrap());
    }
}
