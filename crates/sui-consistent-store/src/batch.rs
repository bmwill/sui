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
//! use bytes::Buf;
//! use bytes::BufMut;
//!
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
//!     fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
//!         buf.put_slice(&self.0.to_be_bytes());
//!         Ok(())
//!     }
//! }
//!
//! impl Decode for U64Be {
//!     fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
//!         if buf.remaining() != 8 {
//!             return Err(DecodeError::msg("expected 8 bytes"));
//!         }
//!         Ok(Self(buf.get_u64()))
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

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use crate::Encode;
use crate::Live;
use crate::SstWriter;
use crate::db::Db;
use crate::encode_buf::with_encode_buf;
use crate::error::Error;
use crate::map::DbMap;

/// An accumulating, typed write batch.
///
/// `Batch` has two backings, chosen at construction time:
///
/// - The **write backing** (default; via [`Db::batch`]) wraps
///   [`rocksdb::WriteBatch`] and commits to the live database via
///   [`commit`](Self::commit) — the tip-of-chain write path.
/// - The **shard backing** (via [`Db::shard_batch`]) records writes
///   into per-CF sorted in-memory buffers. The driver finalizes the
///   buffers into SST files via
///   [`finalize_into_ssts`](Self::finalize_into_ssts) and atomically
///   ingests them via
///   [`Db::ingest_files_cf`](crate::Db::ingest_files_cf) — the
///   bulk-load restore path.
///
/// Both backings expose the same typed [`put`](Self::put),
/// [`merge`](Self::merge), and [`delete`](Self::delete) API against
/// [`DbMap`] handles, so a pipeline's
/// [`commit`](crate::Pipeline::commit) writes the same code against
/// either.
///
/// Shard-backed batches enforce at most one operation per key per
/// column family (the invariant SST ingestion requires). A second
/// write to the same key in the same CF returns
/// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp).
/// Pipelines should fold by key in their
/// [`Self::Batch`](crate::Pipeline::Batch) accumulator before
/// emitting writes.
///
/// All staged operations on a write-backed batch either become
/// visible together or not at all. Encoding failures during staging
/// propagate as [`Error::Encode`](crate::error::Error::Encode); the
/// underlying RocksDB write can fail with
/// [`Error::Rocksdb`](crate::error::Error::Rocksdb) at commit time.
pub struct Batch {
    db: Arc<Db>,
    backing: BatchBacking,
}

/// One of the two storage backings used by [`Batch`].
///
/// See [`Batch`] for the semantic differences between them. The
/// type is internal so the only way to choose a backing is via the
/// [`Db::batch`] (write) and [`Db::shard_batch`] (shard) entry
/// points.
enum BatchBacking {
    Write(rocksdb::WriteBatch),
    Shard(ShardBuffer),
}

/// Per-CF sorted in-memory buffer used by the shard backing of
/// [`Batch`]. Per the SST-ingestion invariant, at most one
/// operation per key is allowed; duplicates are rejected at write
/// time.
struct ShardBuffer {
    per_cf: BTreeMap<String, BTreeMap<Vec<u8>, ShardOp>>,
}

/// One staged operation in a [`ShardBuffer`].
enum ShardOp {
    Put(Vec<u8>),
    Merge(Vec<u8>),
    Delete,
}

impl ShardBuffer {
    fn new() -> Self {
        Self {
            per_cf: BTreeMap::new(),
        }
    }

    /// Insert `op` at `key` in `cf_name`. Errors if a different op
    /// already exists at the same key in the same CF.
    fn insert(&mut self, cf_name: &str, key: Vec<u8>, op: ShardOp) -> Result<(), Error> {
        let per_key = self.per_cf.entry(cf_name.to_string()).or_default();
        let key_len = key.len();
        match per_key.entry(key) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(op);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(_) => Err(Error::DuplicateShardOp {
                cf: cf_name.to_string(),
                key_len,
            }),
        }
    }

    /// Total number of staged operations across all CFs.
    fn len(&self) -> usize {
        self.per_cf.values().map(BTreeMap::len).sum()
    }

    /// Rough size in bytes: sum of encoded key + value bytes plus a
    /// one-byte op tag per entry. Useful for the `Batch::size_in_bytes`
    /// surface but not exact (excludes per-CF map overhead).
    fn size_in_bytes(&self) -> usize {
        self.per_cf
            .values()
            .flat_map(|m| m.iter())
            .map(|(k, op)| {
                k.len()
                    + 1
                    + match op {
                        ShardOp::Put(v) | ShardOp::Merge(v) => v.len(),
                        ShardOp::Delete => 0,
                    }
            })
            .sum()
    }
}

impl Batch {
    pub(crate) fn new(db: Arc<Db>) -> Self {
        Self {
            db,
            backing: BatchBacking::Write(rocksdb::WriteBatch::default()),
        }
    }

    pub(crate) fn new_shard(db: Arc<Db>) -> Self {
        Self {
            db,
            backing: BatchBacking::Shard(ShardBuffer::new()),
        }
    }

    /// Returns whether this batch is shard-backed.
    ///
    /// Useful for drivers that need to dispatch on the backing,
    /// but most callers can simply call [`commit`](Self::commit) or
    /// [`finalize_into_ssts`](Self::finalize_into_ssts) — each errors
    /// when called on the wrong backing.
    pub fn is_shard(&self) -> bool {
        matches!(self.backing, BatchBacking::Shard(_))
    }

    /// Stage a put on the column family backing `map`.
    ///
    /// The key and value are encoded into a thread-local scratch
    /// buffer once per call. For a write-backed batch RocksDB copies
    /// the bytes into the batch's internal representation
    /// synchronously; for a shard-backed batch the bytes are owned
    /// by an internal per-CF buffer.
    ///
    /// `map` is constrained to a [`Live`]-bound handle: writes always
    /// go to the live tip, and snapshot-bound projections are
    /// statically read-only.
    ///
    /// Shard-backed batches return
    /// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp)
    /// if `key` already has a staged operation in the same CF.
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
        match &mut self.backing {
            BatchBacking::Write(wb) => {
                let cf = map
                    .db()
                    .cf_handle(map.cf_name())
                    .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;
                with_encode_buf(|buf| -> Result<(), Error> {
                    key.encode_into(buf)?;
                    let k_end = buf.len();
                    value.encode_into(buf)?;
                    let bytes = buf.as_slice();
                    wb.put_cf(&cf, &bytes[..k_end], &bytes[k_end..]);
                    Ok(())
                })?;
            }
            BatchBacking::Shard(sb) => {
                let (key_bytes, value_bytes) = encode_key_and_value(key, value)?;
                sb.insert(map.cf_name(), key_bytes, ShardOp::Put(value_bytes))?;
            }
        }
        Ok(self)
    }

    /// Stage a delete on the column family backing `map`.
    ///
    /// `map` is constrained to a [`Live`]-bound handle.
    ///
    /// Shard-backed batches return
    /// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp)
    /// if `key` already has a staged operation in the same CF.
    pub fn delete<K, V>(&mut self, map: &DbMap<K, V, Live>, key: &K) -> Result<&mut Self, Error>
    where
        K: Encode,
    {
        match &mut self.backing {
            BatchBacking::Write(wb) => {
                let cf = map
                    .db()
                    .cf_handle(map.cf_name())
                    .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;
                with_encode_buf(|buf| -> Result<(), Error> {
                    key.encode_into(buf)?;
                    wb.delete_cf(&cf, buf.as_slice());
                    Ok(())
                })?;
            }
            BatchBacking::Shard(sb) => {
                let key_bytes = encode_key(key)?;
                sb.insert(map.cf_name(), key_bytes, ShardOp::Delete)?;
            }
        }
        Ok(self)
    }

    /// Stage a merge operand on the column family backing `map`.
    ///
    /// For a write-backed batch, the encoded `operand` bytes are
    /// passed to the merge operator the schema registered on this
    /// column family's [`rocksdb::Options`] at open time. The
    /// operator combines the operand with any existing value at
    /// `key` lazily, at the next read or during compaction.
    ///
    /// For a shard-backed batch, the operand is staged into the
    /// per-CF buffer and, on
    /// [`finalize_into_ssts`](Self::finalize_into_ssts), written as
    /// a merge entry into the produced SST file. RocksDB applies
    /// the operator at read or compaction time after the SST is
    /// ingested. Cross-shard collisions (the same key emitted by
    /// two different shards' SSTs) combine via the operator as
    /// usual.
    ///
    /// `operand` is constrained to the column family's value type
    /// `V`. Schemas whose merge semantics expect a different operand
    /// type than the stored value should encode the operand into a
    /// wrapper that round-trips through the same `Encode`
    /// implementation (or split the column family).
    ///
    /// If the column family has no merge operator configured,
    /// RocksDB rejects the batch at [`commit`](Self::commit) time
    /// (for the write backing) or at read time after ingest (for
    /// the shard backing).
    ///
    /// `map` is constrained to a [`Live`]-bound handle.
    ///
    /// Shard-backed batches return
    /// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp)
    /// if `key` already has a staged operation in the same CF.
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
        match &mut self.backing {
            BatchBacking::Write(wb) => {
                let cf = map
                    .db()
                    .cf_handle(map.cf_name())
                    .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;
                with_encode_buf(|buf| -> Result<(), Error> {
                    key.encode_into(buf)?;
                    let k_end = buf.len();
                    operand.encode_into(buf)?;
                    let bytes = buf.as_slice();
                    wb.merge_cf(&cf, &bytes[..k_end], &bytes[k_end..]);
                    Ok(())
                })?;
            }
            BatchBacking::Shard(sb) => {
                let (key_bytes, operand_bytes) = encode_key_and_value(key, operand)?;
                sb.insert(map.cf_name(), key_bytes, ShardOp::Merge(operand_bytes))?;
            }
        }
        Ok(self)
    }

    /// Commit the staged operations atomically.
    ///
    /// Consumes `self`. On success, all staged operations are visible
    /// to subsequent reads. On failure, the database is left in the
    /// state it was in before the commit was attempted.
    ///
    /// Only valid for a write-backed batch. Shard-backed batches
    /// return [`Error::Internal`](crate::error::Error::Internal) —
    /// the correct termination for those is
    /// [`finalize_into_ssts`](Self::finalize_into_ssts).
    pub fn commit(self) -> Result<(), Error> {
        match self.backing {
            BatchBacking::Write(wb) => {
                self.db.rocksdb().write(wb)?;
                Ok(())
            }
            BatchBacking::Shard(_) => Err(Error::Internal(
                "Batch::commit called on a shard-backed batch; use finalize_into_ssts",
            )),
        }
    }

    /// Commit the staged operations atomically, with caller-supplied
    /// [`rocksdb::WriteOptions`].
    ///
    /// Useful for tuning write durability and WAL behavior on a
    /// per-batch basis (for example, disabling the WAL during a
    /// bulk load, or forcing an `fsync` on a critical commit).
    /// Defaults are appropriate for routine writes; consult the
    /// RocksDB docs for trade-offs.
    ///
    /// Only valid for a write-backed batch; see [`commit`](Self::commit).
    pub fn commit_opt(self, opts: rocksdb::WriteOptions) -> Result<(), Error> {
        match self.backing {
            BatchBacking::Write(wb) => {
                self.db.rocksdb().write_opt(wb, &opts)?;
                Ok(())
            }
            BatchBacking::Shard(_) => Err(Error::Internal(
                "Batch::commit_opt called on a shard-backed batch; use finalize_into_ssts",
            )),
        }
    }

    /// Drain a shard-backed batch into per-CF SST files in
    /// `staging_dir`, ready for atomic ingest via
    /// [`Db::ingest_files_cf`](crate::Db::ingest_files_cf).
    ///
    /// Produces one SST per CF that has staged operations. CFs with
    /// no operations are skipped. The returned `Vec` holds
    /// `(cf_name, path)` pairs in CF-name order; the caller is
    /// responsible for ingesting them — typically one call per CF,
    /// so each set of SSTs lands at the bottommost level on a fresh
    /// CF.
    ///
    /// `sst_options` configures the produced SST files. For the
    /// SSTs to ingest cleanly the comparator must match the target
    /// CF's; default options work for schemas that use the default
    /// byte comparator.
    ///
    /// Consumes `self`. Only valid for a shard-backed batch;
    /// write-backed batches return
    /// [`Error::Internal`](crate::error::Error::Internal).
    pub fn finalize_into_ssts(
        self,
        staging_dir: &Path,
        sst_options: &rocksdb::Options,
    ) -> Result<Vec<(String, PathBuf)>, Error> {
        let buf = match self.backing {
            BatchBacking::Shard(b) => b,
            BatchBacking::Write(_) => {
                return Err(Error::Internal(
                    "Batch::finalize_into_ssts called on a write-backed batch; use commit",
                ));
            }
        };

        let mut out = Vec::with_capacity(buf.per_cf.len());
        for (cf_name, ops) in buf.per_cf {
            if ops.is_empty() {
                continue;
            }
            let filename = format!("{cf_name}.sst");
            let path = staging_dir.join(filename);
            // The BTreeMap iterator yields keys in byte order, which
            // is the order `SstWriter` requires.
            let mut writer: SstWriter<RawBytes, RawBytes> =
                SstWriter::create(&path, sst_options.clone())?;
            for (key, op) in ops {
                let key_raw = RawBytes(key);
                match op {
                    ShardOp::Put(value) => writer.put(&key_raw, &RawBytes(value))?,
                    ShardOp::Merge(operand) => writer.merge(&key_raw, &RawBytes(operand))?,
                    ShardOp::Delete => writer.delete(&key_raw)?,
                }
            }
            let path = writer.finish()?;
            out.push((cf_name, path));
        }
        Ok(out)
    }

    /// Returns whether the batch has no staged operations.
    pub fn is_empty(&self) -> bool {
        match &self.backing {
            BatchBacking::Write(wb) => wb.is_empty(),
            BatchBacking::Shard(sb) => sb.per_cf.values().all(BTreeMap::is_empty),
        }
    }

    /// Returns the number of staged operations.
    pub fn len(&self) -> usize {
        match &self.backing {
            BatchBacking::Write(wb) => wb.len(),
            BatchBacking::Shard(sb) => sb.len(),
        }
    }

    /// Returns the (approximate) size in bytes of the staged
    /// operations.
    ///
    /// For a write-backed batch this is the exact serialized form
    /// size. For a shard-backed batch it is a rough estimate (sum
    /// of key + value bytes plus a one-byte op tag per entry) that
    /// excludes the per-CF map overhead. Useful for choosing when
    /// to flush a long-running batch.
    pub fn size_in_bytes(&self) -> usize {
        match &self.backing {
            BatchBacking::Write(wb) => wb.size_in_bytes(),
            BatchBacking::Shard(sb) => sb.size_in_bytes(),
        }
    }
}

/// Newtype that round-trips raw bytes through [`Encode`] so the
/// shard-backing finalize path can drive a typed
/// [`SstWriter<RawBytes, RawBytes>`] with pre-encoded contents.
/// Crate-internal: not part of the public surface.
struct RawBytes(Vec<u8>);

impl Encode for RawBytes {
    fn encode_into<B: bytes::BufMut>(
        &self,
        buf: &mut B,
    ) -> Result<(), crate::error::EncodeError> {
        buf.put_slice(&self.0);
        Ok(())
    }
}

/// Encode `key` and `value` into freshly-owned byte vectors. Used
/// by the shard backing where the encoded bytes must outlive a
/// single method call. Allocates twice per call; the write backing
/// avoids this by writing directly into a thread-local scratch buf.
fn encode_key_and_value<K: Encode, V: Encode>(
    key: &K,
    value: &V,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let mut k = Vec::new();
    key.encode_into(&mut k)?;
    let mut v = Vec::new();
    value.encode_into(&mut v)?;
    Ok((k, v))
}

/// Encode `key` into a freshly-owned byte vector. See
/// [`encode_key_and_value`].
fn encode_key<K: Encode>(key: &K) -> Result<Vec<u8>, Error> {
    let mut k = Vec::new();
    key.encode_into(&mut k)?;
    Ok(k)
}

impl fmt::Debug for Batch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `rocksdb::WriteBatch` does not implement Debug, so
        // summarize. The number of staged operations is reported as
        // a rough indication of batch state.
        let kind = match &self.backing {
            BatchBacking::Write(_) => "write",
            BatchBacking::Shard(_) => "shard",
        };
        f.debug_struct("Batch")
            .field("backing", &kind)
            .field("ops", &self.len())
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
    use bytes::BufMut;

    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

    /// Hand-rolled big-endian `u64` for tests.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct U64Be(u64);

    impl Encode for U64Be {
        fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
            buf.put_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    impl Decode for U64Be {
        fn decode<B: bytes::Buf>(buf: &mut B) -> Result<Self, DecodeError> {
            if buf.remaining() != 8 {
                return Err(DecodeError::msg("expected 8 bytes"));
            }
            Ok(Self(buf.get_u64()))
        }
    }

    /// Type whose encode always fails. Used to assert that batch
    /// staging propagates encoding errors out of the closure that
    /// holds the thread-local scratch buffer.
    #[derive(Debug)]
    struct AlwaysFails;

    impl Encode for AlwaysFails {
        fn encode_into<B: BufMut>(&self, _: &mut B) -> Result<(), EncodeError> {
            Err(EncodeError::msg("always fails"))
        }
    }

    impl Decode for AlwaysFails {
        fn decode<B: bytes::Buf>(_: &mut B) -> Result<Self, DecodeError> {
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

    mod shard_backing {
        //! Tests for the shard-backed [`Batch`] mode used by the
        //! restore-time bulk-load path: writes accumulate in
        //! per-CF sorted buffers, finalize into SST files, ingest
        //! atomically.

        use tempfile::TempDir;

        use super::*;

        #[test]
        fn is_shard_reports_backing() {
            let (_dir, db, _schema) = open();
            let write = db.batch();
            assert!(!write.is_shard());
            let shard = db.shard_batch();
            assert!(shard.is_shard());
        }

        #[test]
        fn shard_batch_is_initially_empty() {
            let (_dir, db, _schema) = open();
            let batch = db.shard_batch();
            assert!(batch.is_empty());
            assert_eq!(batch.len(), 0);
            assert_eq!(batch.size_in_bytes(), 0);
        }

        #[test]
        fn shard_put_then_finalize_produces_ingestable_sst() {
            let (_dir, db, schema) = open();
            let staging = TempDir::new().unwrap();

            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            batch.put(&schema.items, &U64Be(2), &U64Be(20)).unwrap();
            assert_eq!(batch.len(), 2);

            let ssts = batch
                .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                .unwrap();
            assert_eq!(ssts.len(), 1);
            let (cf, path) = ssts.into_iter().next().unwrap();
            assert_eq!(cf, "items");
            assert!(path.exists());

            db.ingest_files_cf("items", vec![path]).unwrap();
            assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
            assert_eq!(schema.items.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
        }

        #[test]
        fn shard_merge_then_finalize_then_ingest_combines_via_operator() {
            // Shard-backed batch writes merge entries; ingest at
            // bottom level; the operator combines on read.
            let (_dir, db, schema) = open_merge();
            let staging = TempDir::new().unwrap();

            let mut batch = db.shard_batch();
            batch
                .merge(&schema.counters, &U64Be(1), &U64Be(10))
                .unwrap();
            batch
                .merge(&schema.counters, &U64Be(2), &U64Be(20))
                .unwrap();
            let ssts = batch
                .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                .unwrap();
            for (cf, path) in ssts {
                db.ingest_files_cf(&cf, vec![path]).unwrap();
            }
            // Both keys see exactly one merge operand, so the
            // operator's output equals that operand.
            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
            assert_eq!(schema.counters.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
        }

        #[test]
        fn shard_delete_then_finalize_writes_tombstone() {
            // Pre-populate the CF, then ingest an SST containing a
            // delete entry. The deleted key disappears.
            let (_dir, db, schema) = open();
            let staging = TempDir::new().unwrap();

            let mut wb = db.batch();
            wb.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            wb.put(&schema.items, &U64Be(2), &U64Be(20)).unwrap();
            wb.commit().unwrap();
            db.flush().unwrap();

            let mut shard = db.shard_batch();
            shard.delete(&schema.items, &U64Be(1)).unwrap();
            let ssts = shard
                .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                .unwrap();
            for (cf, path) in ssts {
                db.ingest_files_cf(&cf, vec![path]).unwrap();
            }
            assert!(schema.items.get(&U64Be(1)).unwrap().is_none());
            assert_eq!(schema.items.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
        }

        #[test]
        fn shard_duplicate_put_returns_duplicate_shard_op_error() {
            let (_dir, db, schema) = open();
            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            let err = batch
                .put(&schema.items, &U64Be(1), &U64Be(11))
                .unwrap_err();
            match err {
                Error::DuplicateShardOp { ref cf, key_len } => {
                    assert_eq!(cf, "items");
                    assert_eq!(key_len, 8);
                }
                other => panic!("expected DuplicateShardOp, got {other:?}"),
            }
        }

        #[test]
        fn shard_duplicate_merge_returns_duplicate_shard_op_error() {
            let (_dir, db, schema) = open_merge();
            let mut batch = db.shard_batch();
            batch
                .merge(&schema.counters, &U64Be(1), &U64Be(10))
                .unwrap();
            let err = batch
                .merge(&schema.counters, &U64Be(1), &U64Be(20))
                .unwrap_err();
            assert!(matches!(err, Error::DuplicateShardOp { .. }));
        }

        #[test]
        fn shard_put_then_delete_same_key_is_duplicate_op() {
            // SST ingest constrains the per-shard buffer to one op
            // per key. Mixed put-then-delete still counts as a
            // duplicate — the pipeline should resolve to a single
            // op in its accumulator.
            let (_dir, db, schema) = open();
            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            let err = batch.delete(&schema.items, &U64Be(1)).unwrap_err();
            assert!(matches!(err, Error::DuplicateShardOp { .. }));
        }

        #[test]
        fn shard_distinct_keys_in_same_cf_succeed() {
            let (_dir, db, schema) = open();
            let mut batch = db.shard_batch();
            for i in 0..32u64 {
                batch.put(&schema.items, &U64Be(i), &U64Be(i * 10)).unwrap();
            }
            assert_eq!(batch.len(), 32);
            assert!(batch.size_in_bytes() > 0);
        }

        #[test]
        fn shard_same_key_in_different_cfs_is_allowed() {
            // The dedup is per (cf, key), not just by key.
            let (_dir, db, schema) = open();
            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            batch.put(&schema.other, &U64Be(1), &U64Be(20)).unwrap();
            assert_eq!(batch.len(), 2);
        }

        #[test]
        fn finalize_emits_one_sst_per_cf_with_ops() {
            let (_dir, db, schema) = open();
            let staging = TempDir::new().unwrap();
            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            batch.put(&schema.other, &U64Be(2), &U64Be(20)).unwrap();
            let ssts = batch
                .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                .unwrap();
            assert_eq!(ssts.len(), 2);
            let cf_names: Vec<String> = ssts.iter().map(|(cf, _)| cf.clone()).collect();
            // BTreeMap key order is by &str.
            assert_eq!(cf_names, vec!["items".to_string(), "other".to_string()]);
            for (_, path) in ssts {
                assert!(path.exists());
            }
        }

        #[test]
        fn finalize_empty_shard_batch_returns_empty_vec() {
            let (_dir, db, _schema) = open();
            let staging = TempDir::new().unwrap();
            let batch = db.shard_batch();
            let ssts = batch
                .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                .unwrap();
            assert!(ssts.is_empty());
        }

        #[test]
        fn commit_on_shard_backed_batch_returns_internal_error() {
            let (_dir, db, _schema) = open();
            let batch = db.shard_batch();
            let err = batch.commit().unwrap_err();
            assert!(
                matches!(err, Error::Internal(msg) if msg.contains("shard-backed")),
                "expected Internal error mentioning shard backing",
            );
        }

        #[test]
        fn finalize_on_write_backed_batch_returns_internal_error() {
            let (_dir, db, _schema) = open();
            let staging = TempDir::new().unwrap();
            let batch = db.batch();
            let err = batch
                .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                .unwrap_err();
            assert!(
                matches!(err, Error::Internal(msg) if msg.contains("write-backed")),
                "expected Internal error mentioning write backing",
            );
        }

        #[test]
        fn shard_put_propagates_encode_error_for_key() {
            let (_dir, db, _schema) = open();
            let bad: DbMap<AlwaysFails, U64Be> = DbMap::new(db.clone(), "items").unwrap();
            let mut batch = db.shard_batch();
            let err = batch.put(&bad, &AlwaysFails, &U64Be(1)).unwrap_err();
            assert!(matches!(err, Error::Encode(_)));
        }

        #[test]
        fn shard_put_propagates_encode_error_for_value() {
            let (_dir, db, _schema) = open();
            let bad: DbMap<U64Be, AlwaysFails> = DbMap::new(db.clone(), "items").unwrap();
            let mut batch = db.shard_batch();
            let err = batch.put(&bad, &U64Be(1), &AlwaysFails).unwrap_err();
            assert!(matches!(err, Error::Encode(_)));
        }

        #[test]
        fn cross_shard_merges_combine_through_full_pipeline() {
            // End-to-end exercise of the design's restore story:
            // two independent shard-backed batches each emit a
            // merge for the same key, finalize into SSTs, ingest;
            // the registered merge operator combines them.
            let (_dir, db, schema) = open_merge();
            let staging = TempDir::new().unwrap();

            let mut shard1 = db.shard_batch();
            shard1
                .merge(&schema.counters, &U64Be(1), &U64Be(10))
                .unwrap();
            let mut shard2 = db.shard_batch();
            shard2
                .merge(&schema.counters, &U64Be(1), &U64Be(20))
                .unwrap();

            for batch in [shard1, shard2] {
                let ssts = batch
                    .finalize_into_ssts(staging.path(), &rocksdb::Options::default())
                    .unwrap();
                for (cf, path) in ssts {
                    db.ingest_files_cf(&cf, vec![path]).unwrap();
                }
            }
            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(30)));
        }
    }
}
