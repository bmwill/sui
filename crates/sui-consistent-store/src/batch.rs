// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Atomic write batches.
//!
//! [`Batch`] accumulates put, delete, and merge operations across one
//! or more column families and applies them atomically when
//! [`Batch::commit`] is called. RocksDB guarantees that the
//! operations in a single batch either all become visible to readers
//! or none do.
//!
//! Batches are constructed from a [`Db`] via [`Db::batch`]. Each
//! operation takes a [`DbMap`] handle whose key and value types are
//! encoded into bytes using the crate's encoding traits before being
//! handed to the underlying [`rocksdb::WriteBatch`].
//!
//! # Merge operations
//!
//! [`Batch::merge`] stages a merge operand against a key. The
//! merge operator that combines the operand with any existing value
//! is configured by the schema author on the column family's
//! [`rocksdb::Options`] (returned from
//! [`Schema::cfs`](crate::Schema::cfs)) at open time. RocksDB
//! applies the operator lazily at read or compaction time; this
//! crate simply forwards the bytes.
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
//! let mut batch = db.batch();
//! batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
//! batch.put(&schema.items, &U64Be(2), &U64Be(200)).unwrap();
//! batch.commit().unwrap();
//!
//! assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(100)));
//! ```

use std::fmt;
use std::sync::Arc;

use crate::Encode;
use crate::Live;
use crate::db::Db;
use crate::encode_buf::with_encode_buf;
use crate::error::Error;
use crate::map::DbMap;

/// An accumulating, atomic write batch.
///
/// Construct one with [`Db::batch`], stage [`put`](Self::put) and
/// [`delete`](Self::delete) operations against typed column-family
/// handles, then call [`commit`](Self::commit) to apply them
/// atomically.
///
/// All staged operations either become visible together or not at
/// all. Encoding failures during staging propagate as
/// [`Error::Encode`](crate::error::Error::Encode); the underlying
/// write to the database can fail with
/// [`Error::Rocksdb`](crate::error::Error::Rocksdb) at commit time.
pub struct Batch {
    db: Arc<Db>,
    inner: rocksdb::WriteBatch,
}

impl Batch {
    pub(crate) fn new(db: Arc<Db>) -> Self {
        Self {
            db,
            inner: rocksdb::WriteBatch::default(),
        }
    }

    /// Stage a put on the column family backing `map`.
    ///
    /// The key and value are encoded into a thread-local scratch
    /// buffer once per call; RocksDB copies the bytes into the
    /// batch's internal representation synchronously, so the
    /// scratch buffer can be reused for the next operation
    /// immediately.
    ///
    /// `map` is constrained to a [`Live`]-bound handle: writes always
    /// go to the live tip, and snapshot-bound projections are
    /// statically read-only.
    pub fn put<K, V>(
        &mut self,
        map: &DbMap<K, V, Live>,
        key: &K,
        value: &V,
    ) -> Result<&mut Self, Error>
    where
        K: Encode,
        V: Encode,
    {
        // The CF handle borrows from `map.db()`, which is an
        // `Arc<Db>`. The borrow lives only for this method call.
        let cf = map
            .db()
            .cf_handle(map.cf_name())
            .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;

        with_encode_buf(|buf| -> Result<(), Error> {
            key.encode_into(buf)?;
            let k_end = buf.len();
            value.encode_into(buf)?;
            let bytes = buf.as_slice();
            self.inner.put_cf(&cf, &bytes[..k_end], &bytes[k_end..]);
            Ok(())
        })?;
        Ok(self)
    }

    /// Stage a delete on the column family backing `map`.
    ///
    /// The key is encoded into a thread-local scratch buffer; RocksDB
    /// copies the bytes into the batch's internal representation
    /// before this method returns.
    ///
    /// `map` is constrained to a [`Live`]-bound handle.
    pub fn delete<K, V>(&mut self, map: &DbMap<K, V, Live>, key: &K) -> Result<&mut Self, Error>
    where
        K: Encode,
    {
        let cf = map
            .db()
            .cf_handle(map.cf_name())
            .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;

        with_encode_buf(|buf| -> Result<(), Error> {
            key.encode_into(buf)?;
            self.inner.delete_cf(&cf, buf.as_slice());
            Ok(())
        })?;
        Ok(self)
    }

    /// Stage a merge operand on the column family backing `map`.
    ///
    /// The encoded `operand` bytes are passed to the merge operator
    /// the schema registered on this column family's
    /// [`rocksdb::Options`] (via
    /// [`set_merge_operator_associative`](rocksdb::Options::set_merge_operator_associative)
    /// or
    /// [`set_merge_operator`](rocksdb::Options::set_merge_operator))
    /// at open time. The operator combines the operand with any
    /// existing value at `key` lazily, at the next read or during
    /// compaction; this method only stages the operand.
    ///
    /// `operand` is constrained to the column family's value type
    /// `V`. Schemas whose merge semantics expect a different operand
    /// type than the stored value should encode the operand into a
    /// wrapper that round-trips through the same `Encode`
    /// implementation (or split the column family).
    ///
    /// If the column family has no merge operator configured,
    /// RocksDB rejects the batch at [`commit`](Self::commit) time
    /// with a [`Error::Rocksdb`](crate::error::Error::Rocksdb).
    ///
    /// `map` is constrained to a [`Live`]-bound handle.
    pub fn merge<K, V>(
        &mut self,
        map: &DbMap<K, V, Live>,
        key: &K,
        operand: &V,
    ) -> Result<&mut Self, Error>
    where
        K: Encode,
        V: Encode,
    {
        let cf = map
            .db()
            .cf_handle(map.cf_name())
            .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;

        with_encode_buf(|buf| -> Result<(), Error> {
            key.encode_into(buf)?;
            let k_end = buf.len();
            operand.encode_into(buf)?;
            let bytes = buf.as_slice();
            self.inner.merge_cf(&cf, &bytes[..k_end], &bytes[k_end..]);
            Ok(())
        })?;
        Ok(self)
    }

    /// Commit the staged operations atomically.
    ///
    /// Consumes `self`. On success, all staged operations are visible
    /// to subsequent reads. On failure, the database is left in the
    /// state it was in before the commit was attempted.
    pub fn commit(self) -> Result<(), Error> {
        self.db.rocksdb().write(self.inner)?;
        Ok(())
    }

    /// Commit the staged operations atomically, with caller-supplied
    /// [`rocksdb::WriteOptions`].
    ///
    /// Useful for tuning write durability and WAL behavior on a
    /// per-batch basis (for example, disabling the WAL during a
    /// bulk load, or forcing an `fsync` on a critical commit).
    /// Defaults are appropriate for routine writes; consult the
    /// RocksDB docs for trade-offs.
    pub fn commit_opt(self, opts: rocksdb::WriteOptions) -> Result<(), Error> {
        self.db.rocksdb().write_opt(self.inner, &opts)?;
        Ok(())
    }

    /// Returns whether the batch has no staged operations.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of staged operations.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the size in bytes of the batch's serialized form.
    ///
    /// Useful for choosing when to flush a long-running batch
    /// rather than risk an oversized commit.
    pub fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes()
    }
}

impl fmt::Debug for Batch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `rocksdb::WriteBatch` does not implement Debug, so
        // summarize. The number of staged operations is reported as
        // a rough indication of batch state.
        f.debug_struct("Batch")
            .field("ops", &self.inner.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::DbOptions;
    use crate::Decode;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

    /// Hand-rolled big-endian `u64` for tests.
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

    /// Type whose encode always fails. Used to assert that batch
    /// staging propagates encoding errors out of the closure that
    /// holds the thread-local scratch buffer.
    #[derive(Debug)]
    struct AlwaysFails;

    impl Encode for AlwaysFails {
        fn encode_into(&self, _: &mut Vec<u8>) -> Result<(), EncodeError> {
            Err(EncodeError::msg("always fails"))
        }
    }

    impl Decode for AlwaysFails {
        fn decode(_: &[u8]) -> Result<Self, DecodeError> {
            Err(DecodeError::msg("never reached"))
        }
    }

    #[derive(Debug)]
    struct TestSchema {
        items: DbMap<U64Be, U64Be>,
        other: DbMap<U64Be, U64Be>,
    }

    impl Schema for TestSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
            vec![
                ("items", base_options.clone()),
                ("other", base_options.clone()),
            ]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                items: DbMap::new(db.clone(), "items")?,
                other: DbMap::new(db.clone(), "other")?,
            })
        }
    }

    fn open() -> (TempDir, Arc<Db>, TestSchema) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db, schema)
    }

    #[test]
    fn put_then_get_round_trips() {
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(100)));
    }

    #[test]
    fn delete_removes_existing_key() {
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
        batch.commit().unwrap();
        let mut batch = db.batch();
        batch.delete(&schema.items, &U64Be(1)).unwrap();
        batch.commit().unwrap();
        assert!(schema.items.get(&U64Be(1)).unwrap().is_none());
    }

    #[test]
    fn put_then_delete_in_same_batch_results_in_absent() {
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
        batch.delete(&schema.items, &U64Be(1)).unwrap();
        batch.commit().unwrap();
        assert!(schema.items.get(&U64Be(1)).unwrap().is_none());
    }

    #[test]
    fn empty_batch_commits_without_error() {
        let (_dir, db, _schema) = open();
        db.batch().commit().unwrap();
    }

    #[test]
    fn empty_batch_observability() {
        let (_dir, db, _schema) = open();
        let batch = db.batch();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        // The WriteBatch carries a small fixed header even when
        // empty; what matters here is that adding operations grows
        // the size. See `populated_batch_observability`.
    }

    #[test]
    fn populated_batch_observability() {
        let (_dir, db, schema) = open();
        let empty_size = db.batch().size_in_bytes();
        let mut batch = db.batch();
        batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
        batch.put(&schema.items, &U64Be(2), &U64Be(20)).unwrap();
        batch.delete(&schema.items, &U64Be(3)).unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 3);
        assert!(batch.size_in_bytes() > empty_size);
    }

    #[test]
    fn commit_opt_respects_disable_wal_flag() {
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
        let mut wopts = rocksdb::WriteOptions::default();
        wopts.disable_wal(true);
        batch.commit_opt(wopts).unwrap();
        assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
    }

    #[test]
    fn batch_spans_multiple_cfs_atomically() {
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
        batch.put(&schema.other, &U64Be(2), &U64Be(200)).unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(100)));
        assert_eq!(schema.other.get(&U64Be(2)).unwrap(), Some(U64Be(200)));
    }

    #[test]
    fn put_chains_via_mut_self() {
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch
            .put(&schema.items, &U64Be(1), &U64Be(10))
            .unwrap()
            .put(&schema.items, &U64Be(2), &U64Be(20))
            .unwrap()
            .delete(&schema.items, &U64Be(2))
            .unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
        assert!(schema.items.get(&U64Be(2)).unwrap().is_none());
    }

    #[test]
    fn put_propagates_encode_error_for_key() {
        let (_dir, db, _schema) = open();
        // A throwaway DbMap whose key type always fails to encode.
        let bad: DbMap<AlwaysFails, U64Be> = DbMap::new(db.clone(), "items").unwrap();
        let mut batch = db.batch();
        let err = batch.put(&bad, &AlwaysFails, &U64Be(1)).unwrap_err();
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn put_propagates_encode_error_for_value() {
        let (_dir, db, _schema) = open();
        let bad: DbMap<U64Be, AlwaysFails> = DbMap::new(db.clone(), "items").unwrap();
        let mut batch = db.batch();
        let err = batch.put(&bad, &U64Be(1), &AlwaysFails).unwrap_err();
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn delete_propagates_encode_error_for_key() {
        let (_dir, db, _schema) = open();
        let bad: DbMap<AlwaysFails, U64Be> = DbMap::new(db.clone(), "items").unwrap();
        let mut batch = db.batch();
        let err = batch.delete(&bad, &AlwaysFails).unwrap_err();
        assert!(matches!(err, Error::Encode(_)));
    }

    /// Associative merge operator that interprets each operand and
    /// the existing value (if any) as eight big-endian bytes, sums
    /// them with saturation, and writes the result back in the same
    /// format. Operands and missing values that aren't exactly
    /// eight bytes are skipped.
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

    #[derive(Debug)]
    struct MergeSchema {
        counters: DbMap<U64Be, U64Be>,
    }

    impl Schema for MergeSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
            let mut counter_opts = base_options.clone();
            counter_opts.set_merge_operator_associative("u64-add", add_u64_merge_op);
            vec![("counters", counter_opts)]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                counters: DbMap::new(db.clone(), "counters")?,
            })
        }
    }

    fn open_merge() -> (TempDir, Arc<Db>, MergeSchema) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<MergeSchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db, schema)
    }

    #[test]
    fn merge_aggregates_via_registered_operator() {
        let (_dir, db, schema) = open_merge();
        let mut batch = db.batch();
        batch
            .merge(&schema.counters, &U64Be(1), &U64Be(10))
            .unwrap();
        batch
            .merge(&schema.counters, &U64Be(1), &U64Be(20))
            .unwrap();
        batch.merge(&schema.counters, &U64Be(1), &U64Be(7)).unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(37)));
    }

    #[test]
    fn merge_into_empty_key_starts_from_zero() {
        let (_dir, db, schema) = open_merge();
        let mut batch = db.batch();
        batch
            .merge(&schema.counters, &U64Be(1), &U64Be(42))
            .unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(42)));
    }

    #[test]
    fn merge_combines_with_prior_put() {
        let (_dir, db, schema) = open_merge();
        let mut batch = db.batch();
        batch.put(&schema.counters, &U64Be(1), &U64Be(100)).unwrap();
        batch.commit().unwrap();
        let mut batch = db.batch();
        batch
            .merge(&schema.counters, &U64Be(1), &U64Be(50))
            .unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(150)));
    }

    #[test]
    fn merge_independent_keys_do_not_combine() {
        let (_dir, db, schema) = open_merge();
        let mut batch = db.batch();
        batch
            .merge(&schema.counters, &U64Be(1), &U64Be(10))
            .unwrap();
        batch
            .merge(&schema.counters, &U64Be(2), &U64Be(20))
            .unwrap();
        batch.commit().unwrap();
        assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
        assert_eq!(schema.counters.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
    }

    #[test]
    fn merge_propagates_encode_error_for_key() {
        let (_dir, db, _schema) = open_merge();
        let bad: DbMap<AlwaysFails, U64Be> = DbMap::new(db.clone(), "counters").unwrap();
        let mut batch = db.batch();
        let err = batch.merge(&bad, &AlwaysFails, &U64Be(1)).unwrap_err();
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn merge_propagates_encode_error_for_operand() {
        let (_dir, db, _schema) = open_merge();
        let bad: DbMap<U64Be, AlwaysFails> = DbMap::new(db.clone(), "counters").unwrap();
        let mut batch = db.batch();
        let err = batch.merge(&bad, &U64Be(1), &AlwaysFails).unwrap_err();
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn merge_without_operator_errors_at_commit() {
        // The default `items` CF in `TestSchema` does not have a
        // merge operator. RocksDB rejects merges on it at write
        // time.
        let (_dir, db, schema) = open();
        let mut batch = db.batch();
        batch.merge(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
        let err = batch.commit().unwrap_err();
        assert!(matches!(err, Error::Rocksdb(_)));
    }
}
