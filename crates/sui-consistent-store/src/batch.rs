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
//!     fn cfs(base_options: &rocksdb::Options) -> Vec<sui_consistent_store::CfDescriptor> {
//!         vec![sui_consistent_store::CfDescriptor::new("items", base_options.clone())]
//!     }
//!
//!     fn open(db: &Db) -> Result<Self, OpenError> {
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

use crate::Encode;
use crate::SstWriter;
use crate::db::Db;
use crate::encode_buf::with_encode_buf;
use crate::error::Error;
use crate::map::DbMap;
use crate::schema::RestoreMode;

/// An accumulating, typed write batch.
///
/// `Batch` has two backings, chosen at construction time:
///
/// - The **write backing** (default; via [`Db::batch`]) wraps
///   [`rocksdb::WriteBatch`] and commits to the live database via
///   [`commit`](Self::commit) — the tip-of-chain write path.
/// - The **shard backing** (via [`Db::shard_batch`]) is a *hybrid*
///   per-CF dispatch used by restore drivers. Each write is routed
///   based on the CF's [`RestoreMode`](crate::RestoreMode):
///   [`BulkIngest`](crate::RestoreMode::BulkIngest) CFs buffer into
///   per-CF sorted in-memory maps for SST emission;
///   [`MergeViaWriteBatch`](crate::RestoreMode::MergeViaWriteBatch)
///   CFs stream into an internal [`rocksdb::WriteBatch`] for atomic
///   commit. The driver finalizes the batch via
///   [`finalize_for_shard`](Self::finalize_for_shard), which
///   produces per-CF SST paths plus the WriteBatch holding the
///   merge-mode writes.
///
/// Both backings expose the same typed [`put`](Self::put),
/// [`merge`](Self::merge), and [`delete`](Self::delete) API against
/// [`DbMap`] handles, so a pipeline's
/// [`commit`](crate::Pipeline::commit) writes the same code against
/// either.
///
/// Shard-backed batches enforce at most one operation per key per
/// [`BulkIngest`](crate::RestoreMode::BulkIngest) column family
/// (the invariant SST ingestion requires). A second write to the
/// same key in the same `BulkIngest` CF returns
/// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp).
/// [`MergeViaWriteBatch`](crate::RestoreMode::MergeViaWriteBatch)
/// CFs have *no* per-key constraint: pipelines emit as many
/// operations per key per shard as their merge operator's semantics
/// require.
///
/// All staged operations on a write-backed batch either become
/// visible together or not at all. Encoding failures during staging
/// propagate as [`Error::Encode`](crate::error::Error::Encode); the
/// underlying RocksDB write can fail with
/// [`Error::Rocksdb`](crate::error::Error::Rocksdb) at commit time.
pub struct Batch {
    db: Db,
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

/// Hybrid shard-mode buffer. Routes per-CF writes between a sorted
/// per-CF map (for [`RestoreMode::BulkIngest`] CFs) and an internal
/// [`rocksdb::WriteBatch`] (for [`RestoreMode::MergeViaWriteBatch`]
/// CFs).
struct ShardBuffer {
    /// Per-CF sorted buffer of operations for CFs that finalize
    /// into SSTs. SST ingestion requires at most one op per key, so
    /// duplicates are rejected at write time.
    bulk_ingest: BTreeMap<String, BTreeMap<Vec<u8>, ShardOp>>,
    /// Operations for CFs whose merge operator should see many
    /// operands per key per shard. Buffered into a
    /// [`rocksdb::WriteBatch`] that the runner commits atomically
    /// alongside the shard's partition-complete marker.
    merge_write: rocksdb::WriteBatch,
    /// Count of ops staged into `merge_write`; tracked separately
    /// because [`rocksdb::WriteBatch::len`] reports the total
    /// including any externally appended writes.
    merge_write_ops: usize,
    /// Approximate byte size of ops staged into `merge_write`.
    /// Same caveat as the per-CF buffer: excludes per-CF metadata.
    merge_write_size: usize,
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
            bulk_ingest: BTreeMap::new(),
            merge_write: rocksdb::WriteBatch::default(),
            merge_write_ops: 0,
            merge_write_size: 0,
        }
    }

    /// Insert `op` at `key` in a [`RestoreMode::BulkIngest`] CF's
    /// per-key buffer. Errors if a different op already exists at
    /// the same key in the same CF.
    fn insert_bulk_ingest(
        &mut self,
        cf_name: &str,
        key: Vec<u8>,
        op: ShardOp,
    ) -> Result<(), Error> {
        let per_key = self.bulk_ingest.entry(cf_name.to_string()).or_default();
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
        self.bulk_ingest_len() + self.merge_write_ops
    }

    fn bulk_ingest_len(&self) -> usize {
        self.bulk_ingest.values().map(BTreeMap::len).sum()
    }

    /// Rough size in bytes: sum of encoded key + value bytes plus a
    /// one-byte op tag per entry. Useful for the `Batch::size_in_bytes`
    /// surface but not exact (excludes per-CF map overhead).
    fn size_in_bytes(&self) -> usize {
        let bulk_bytes: usize = self
            .bulk_ingest
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
            .sum();
        bulk_bytes + self.merge_write_size
    }
}

impl Batch {
    pub(crate) fn new(db: Db) -> Self {
        Self {
            db,
            backing: BatchBacking::Write(rocksdb::WriteBatch::default()),
        }
    }

    pub(crate) fn new_shard(db: Db) -> Self {
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
    /// synchronously; for a shard-backed batch the routing depends
    /// on the CF's [`RestoreMode`](crate::RestoreMode):
    /// [`BulkIngest`](crate::RestoreMode::BulkIngest) buffers into a
    /// per-CF sorted map (one op per key); other modes stream into
    /// the internal [`rocksdb::WriteBatch`].
    ///
    /// `map` is constrained to a [`Db`]-bound handle: writes always
    /// go to the live tip, and snapshot-bound projections (or
    /// borrowed-`&Db` projections) are statically refused.
    ///
    /// On a shard-backed batch targeting a
    /// [`BulkIngest`](crate::RestoreMode::BulkIngest) CF, returns
    /// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp)
    /// if `key` already has a staged operation.
    pub fn put<K, V>(
        &mut self,
        map: &DbMap<K, V, Db>,
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
            BatchBacking::Shard(sb) => match shard_mode_for(&self.db, map.cf_name())? {
                RestoreMode::BulkIngest => {
                    let (key_bytes, value_bytes) = encode_key_and_value(key, value)?;
                    sb.insert_bulk_ingest(map.cf_name(), key_bytes, ShardOp::Put(value_bytes))?;
                }
                RestoreMode::MergeViaWriteBatch => {
                    let cf = self
                        .db
                        .cf_handle(map.cf_name())
                        .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;
                    let total_len = with_encode_buf(|buf| -> Result<usize, Error> {
                        key.encode_into(buf)?;
                        let k_end = buf.len();
                        value.encode_into(buf)?;
                        let bytes = buf.as_slice();
                        sb.merge_write.put_cf(&cf, &bytes[..k_end], &bytes[k_end..]);
                        Ok(bytes.len())
                    })?;
                    sb.merge_write_ops += 1;
                    sb.merge_write_size += total_len + 1;
                }
            },
        }
        Ok(self)
    }

    /// Stage a delete on the column family backing `map`.
    ///
    /// `map` is constrained to a [`Db`]-bound handle.
    ///
    /// On a shard-backed batch targeting a
    /// [`BulkIngest`](crate::RestoreMode::BulkIngest) CF, returns
    /// [`Error::DuplicateShardOp`](crate::error::Error::DuplicateShardOp)
    /// if `key` already has a staged operation.
    pub fn delete<K, V>(&mut self, map: &DbMap<K, V, Db>, key: &K) -> Result<&mut Self, Error>
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
            BatchBacking::Shard(sb) => match shard_mode_for(&self.db, map.cf_name())? {
                RestoreMode::BulkIngest => {
                    let key_bytes = encode_key(key)?;
                    sb.insert_bulk_ingest(map.cf_name(), key_bytes, ShardOp::Delete)?;
                }
                RestoreMode::MergeViaWriteBatch => {
                    let cf = self
                        .db
                        .cf_handle(map.cf_name())
                        .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;
                    let k_len = with_encode_buf(|buf| -> Result<usize, Error> {
                        key.encode_into(buf)?;
                        sb.merge_write.delete_cf(&cf, buf.as_slice());
                        Ok(buf.len())
                    })?;
                    sb.merge_write_ops += 1;
                    sb.merge_write_size += k_len + 1;
                }
            },
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
    /// For a shard-backed batch the routing depends on the CF's
    /// [`RestoreMode`](crate::RestoreMode):
    ///
    /// - [`BulkIngest`](crate::RestoreMode::BulkIngest): the operand
    ///   is staged into the per-CF buffer and, on
    ///   [`finalize_for_shard`](Self::finalize_for_shard), written
    ///   as a merge entry into the produced SST file. RocksDB
    ///   applies the operator at read or compaction time after the
    ///   SST is ingested. *One merge operand per key per shard*;
    ///   the pipeline must pre-fold by key in its accumulator.
    ///
    /// - [`MergeViaWriteBatch`](crate::RestoreMode::MergeViaWriteBatch):
    ///   the operand streams into the batch's internal
    ///   [`rocksdb::WriteBatch`]. Many operands per key per shard
    ///   are allowed; the runner commits the WriteBatch atomically
    ///   when the shard finalizes.
    ///
    /// Cross-shard collisions on the same key combine via the
    /// operator regardless of mode.
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
    /// `map` is constrained to a [`Db`]-bound handle.
    pub fn merge<K, V>(
        &mut self,
        map: &DbMap<K, V, Db>,
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
            BatchBacking::Shard(sb) => match shard_mode_for(&self.db, map.cf_name())? {
                RestoreMode::BulkIngest => {
                    let (key_bytes, operand_bytes) = encode_key_and_value(key, operand)?;
                    sb.insert_bulk_ingest(map.cf_name(), key_bytes, ShardOp::Merge(operand_bytes))?;
                }
                RestoreMode::MergeViaWriteBatch => {
                    let cf = self
                        .db
                        .cf_handle(map.cf_name())
                        .ok_or_else(|| Error::MissingColumnFamily(map.cf_name().to_string()))?;
                    let total_len = with_encode_buf(|buf| -> Result<usize, Error> {
                        key.encode_into(buf)?;
                        let k_end = buf.len();
                        operand.encode_into(buf)?;
                        let bytes = buf.as_slice();
                        sb.merge_write
                            .merge_cf(&cf, &bytes[..k_end], &bytes[k_end..]);
                        Ok(bytes.len())
                    })?;
                    sb.merge_write_ops += 1;
                    sb.merge_write_size += total_len + 1;
                }
            },
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

    /// Drain a shard-backed batch into a [`ShardFinalize`] ready to
    /// commit.
    ///
    /// The returned [`ShardFinalize`] carries:
    ///
    /// - One SST per [`RestoreMode::BulkIngest`](crate::RestoreMode::BulkIngest)
    ///   CF that has staged operations, finalized into
    ///   `staging_dir`. The runner ingests these via
    ///   [`Db::ingest_files_cf`](crate::Db::ingest_files_cf).
    ///
    /// - A [`rocksdb::WriteBatch`] holding writes for every
    ///   [`RestoreMode::MergeViaWriteBatch`](crate::RestoreMode::MergeViaWriteBatch)
    ///   CF the pipeline touched. The runner is expected to stage
    ///   the partition-complete marker into this same batch (via
    ///   [`Db::stage_restore_state`](crate::Db::stage_restore_state))
    ///   and commit it atomically *after* the SSTs are ingested —
    ///   so a crash between the two steps leaves the partition not
    ///   marked, and the resume re-runs the shard without
    ///   double-merging.
    ///
    /// CFs with no staged operations are skipped. SST entries are
    /// returned in CF-name order.
    ///
    /// `sst_options` configures the produced SST files. For the
    /// SSTs to ingest cleanly the comparator must match the target
    /// CF's; default options work for schemas that use the default
    /// byte comparator.
    ///
    /// Consumes `self`. Only valid for a shard-backed batch;
    /// write-backed batches return
    /// [`Error::Internal`](crate::error::Error::Internal).
    pub fn finalize_for_shard(
        self,
        staging_dir: &Path,
        sst_options: &rocksdb::Options,
    ) -> Result<ShardFinalize, Error> {
        let buf = match self.backing {
            BatchBacking::Shard(b) => b,
            BatchBacking::Write(_) => {
                return Err(Error::Internal(
                    "Batch::finalize_for_shard called on a write-backed batch; use commit",
                ));
            }
        };

        let mut ssts = Vec::with_capacity(buf.bulk_ingest.len());
        for (cf_name, ops) in buf.bulk_ingest {
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
            ssts.push((cf_name, path));
        }

        Ok(ShardFinalize {
            db: self.db,
            ssts,
            write_batch: buf.merge_write,
        })
    }

    /// Returns whether the batch has no staged operations.
    pub fn is_empty(&self) -> bool {
        match &self.backing {
            BatchBacking::Write(wb) => wb.is_empty(),
            BatchBacking::Shard(sb) => sb.len() == 0,
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

/// The output of [`Batch::finalize_for_shard`] — everything a
/// restore driver needs to atomically commit one shard's writes.
///
/// `ssts` holds finalized SST file paths for every
/// [`RestoreMode::BulkIngest`](crate::RestoreMode::BulkIngest) CF
/// the shard touched, in CF-name order. The driver ingests these
/// via [`Db::ingest_files_cf`](crate::Db::ingest_files_cf).
///
/// `write_batch` holds operations against every
/// [`RestoreMode::MergeViaWriteBatch`](crate::RestoreMode::MergeViaWriteBatch)
/// CF the shard touched. The driver extends it with the shard's
/// partition-complete marker (typically via
/// [`Db::stage_restore_state`](crate::Db::stage_restore_state)) and
/// commits it atomically via [`commit`](Self::commit).
///
/// # Commit ordering
///
/// Always ingest SSTs first, then commit the WriteBatch. A crash
/// between the two steps leaves the partition-complete marker
/// unwritten, so resume re-runs the shard from scratch:
///
/// - Re-ingest of the same SST contents is idempotent for puts
///   (last-write-wins) and for deletes; the bottom-most level
///   ends up with the same observable state.
/// - The WriteBatch never committed on the first run, so no
///   merge-mode operations landed — no double-merge on resume.
///
/// [`commit`](Self::commit) implements this ordering. Drivers that
/// need finer control (e.g. to ingest multiple shards' SSTs before
/// committing any of their markers) can drain
/// [`take_ssts`](Self::take_ssts) and
/// [`take_write_batch`](Self::take_write_batch) manually.
pub struct ShardFinalize {
    db: Db,
    ssts: Vec<(String, PathBuf)>,
    write_batch: rocksdb::WriteBatch,
}

impl ShardFinalize {
    /// The per-CF SST paths to ingest.
    pub fn ssts(&self) -> &[(String, PathBuf)] {
        &self.ssts
    }

    /// Borrow the internal [`rocksdb::WriteBatch`] mutably so the
    /// driver can stage additional operations (typically the
    /// partition-complete marker via
    /// [`Db::stage_restore_state`](crate::Db::stage_restore_state)).
    pub fn write_batch_mut(&mut self) -> &mut rocksdb::WriteBatch {
        &mut self.write_batch
    }

    /// Take the SST paths, leaving the [`ShardFinalize`] usable for
    /// a later WriteBatch commit. Useful for drivers that batch
    /// ingest calls across shards.
    pub fn take_ssts(&mut self) -> Vec<(String, PathBuf)> {
        std::mem::take(&mut self.ssts)
    }

    /// Take the WriteBatch, leaving the [`ShardFinalize`] empty.
    /// Useful for drivers that commit through a custom path.
    pub fn take_write_batch(&mut self) -> rocksdb::WriteBatch {
        std::mem::take(&mut self.write_batch)
    }

    /// Atomically commit this shard's writes.
    ///
    /// Sequence:
    /// 1. Ingest each SST via
    ///    [`Db::ingest_files_cf`](crate::Db::ingest_files_cf).
    /// 2. Commit the (caller-extended) [`rocksdb::WriteBatch`].
    ///
    /// See the [`ShardFinalize`] docs for the crash-safety story
    /// of this ordering.
    pub fn commit(self) -> Result<(), Error> {
        for (cf, path) in &self.ssts {
            self.db.ingest_files_cf(cf, vec![path.clone()])?;
        }
        self.db.rocksdb().write(self.write_batch)?;
        Ok(())
    }
}

impl fmt::Debug for ShardFinalize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShardFinalize")
            .field("ssts", &self.ssts.len())
            .field("write_batch_ops", &self.write_batch.len())
            .finish_non_exhaustive()
    }
}

/// Look up the [`RestoreMode`] for `cf_name` on `db`, returning
/// [`Error::MissingColumnFamily`] if the CF was not registered at
/// open time. Used by the shard backing to dispatch per-CF.
fn shard_mode_for(db: &Db, cf_name: &str) -> Result<RestoreMode, Error> {
    db.restore_mode(cf_name)
        .ok_or_else(|| Error::MissingColumnFamily(cf_name.to_string()))
}

/// Newtype that round-trips raw bytes through [`Encode`] so the
/// shard-backing finalize path can drive a typed
/// [`SstWriter<RawBytes, RawBytes>`] with pre-encoded contents.
/// Crate-internal: not part of the public surface.
struct RawBytes(Vec<u8>);

impl Encode for RawBytes {
    fn encode_into<B: bytes::BufMut>(&self, buf: &mut B) -> Result<(), crate::error::EncodeError> {
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
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            vec![
                crate::CfDescriptor::new("items", base_options.clone()),
                crate::CfDescriptor::new("other", base_options.clone()),
            ]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                items: DbMap::new(db.clone(), "items")?,
                other: DbMap::new(db.clone(), "other")?,
            })
        }
    }

    fn open() -> (TempDir, Db, TestSchema) {
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
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            let mut counter_opts = base_options.clone();
            counter_opts.set_merge_operator_associative("u64-add", add_u64_merge_op);
            // The merge-operator CF opts into the WriteBatch shard
            // mode so per-shard tests can stage many merges per key.
            vec![
                crate::CfDescriptor::new("counters", counter_opts)
                    .with_restore_mode(crate::RestoreMode::MergeViaWriteBatch),
            ]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                counters: DbMap::new(db.clone(), "counters")?,
            })
        }
    }

    fn open_merge() -> (TempDir, Db, MergeSchema) {
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

            let finalize = batch
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                .unwrap();
            assert_eq!(finalize.ssts().len(), 1);
            assert_eq!(finalize.ssts()[0].0, "items");
            assert!(finalize.ssts()[0].1.exists());

            finalize.commit().unwrap();
            assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
            assert_eq!(schema.items.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
        }

        #[test]
        fn shard_merge_via_write_batch_combines_via_operator() {
            // The merge-operator CF in MergeSchema opts into the
            // MergeViaWriteBatch shard mode. Many merges per key in
            // one shard are allowed; the WriteBatch commit applies
            // them in order through the registered operator.
            let (_dir, db, schema) = open_merge();
            let staging = TempDir::new().unwrap();

            let mut batch = db.shard_batch();
            batch
                .merge(&schema.counters, &U64Be(1), &U64Be(10))
                .unwrap();
            batch.merge(&schema.counters, &U64Be(1), &U64Be(5)).unwrap();
            batch
                .merge(&schema.counters, &U64Be(2), &U64Be(20))
                .unwrap();

            let finalize = batch
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                .unwrap();
            // No SSTs are produced — the merge-mode CF goes via
            // WriteBatch only.
            assert!(finalize.ssts().is_empty());
            finalize.commit().unwrap();

            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(15)));
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
            let finalize = shard
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                .unwrap();
            finalize.commit().unwrap();
            assert!(schema.items.get(&U64Be(1)).unwrap().is_none());
            assert_eq!(schema.items.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
        }

        #[test]
        fn shard_duplicate_put_in_bulk_ingest_cf_returns_duplicate_shard_op() {
            let (_dir, db, schema) = open();
            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            let err = batch.put(&schema.items, &U64Be(1), &U64Be(11)).unwrap_err();
            match err {
                Error::DuplicateShardOp { ref cf, key_len } => {
                    assert_eq!(cf, "items");
                    assert_eq!(key_len, 8);
                }
                other => panic!("expected DuplicateShardOp, got {other:?}"),
            }
        }

        #[test]
        fn shard_duplicate_merge_in_merge_mode_cf_is_allowed() {
            // The merge-operator CF declares MergeViaWriteBatch mode;
            // pipelines may emit many merge operands per key per
            // shard without DuplicateShardOp firing.
            let (_dir, db, schema) = open_merge();
            let mut batch = db.shard_batch();
            batch
                .merge(&schema.counters, &U64Be(1), &U64Be(10))
                .unwrap();
            batch
                .merge(&schema.counters, &U64Be(1), &U64Be(20))
                .unwrap();
            assert_eq!(batch.len(), 2);
        }

        #[test]
        fn shard_put_then_delete_same_key_in_bulk_cf_is_duplicate_op() {
            // SST ingest constrains BulkIngest CFs to one op per key.
            // Mixed put-then-delete still counts as a duplicate — the
            // pipeline should resolve to a single op in its
            // accumulator.
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
        fn finalize_emits_one_sst_per_bulk_ingest_cf_with_ops() {
            let (_dir, db, schema) = open();
            let staging = TempDir::new().unwrap();
            let mut batch = db.shard_batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(10)).unwrap();
            batch.put(&schema.other, &U64Be(2), &U64Be(20)).unwrap();
            let finalize = batch
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                .unwrap();
            let ssts = finalize.ssts();
            assert_eq!(ssts.len(), 2);
            let cf_names: Vec<String> = ssts.iter().map(|(cf, _)| cf.clone()).collect();
            // BTreeMap key order is by &str.
            assert_eq!(cf_names, vec!["items".to_string(), "other".to_string()]);
            for (_, path) in ssts {
                assert!(path.exists());
            }
        }

        #[test]
        fn finalize_empty_shard_batch_returns_empty_finalize() {
            let (_dir, db, _schema) = open();
            let staging = TempDir::new().unwrap();
            let batch = db.shard_batch();
            let finalize = batch
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                .unwrap();
            assert!(finalize.ssts().is_empty());
            // Committing the empty finalize is a no-op.
            finalize.commit().unwrap();
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
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
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
            // two independent shard-backed batches each emit one or
            // more merges for the same key, finalize, commit; the
            // registered merge operator combines across shards.
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
                let finalize = batch
                    .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                    .unwrap();
                finalize.commit().unwrap();
            }
            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(30)));
        }

        #[test]
        fn shard_finalize_extends_write_batch_with_runner_marker() {
            // The runner stages the partition-complete marker into
            // the same WriteBatch that holds the shard's merge-mode
            // writes. Verify the surface: get the WriteBatch, stage
            // a marker, commit, observe both effects.
            let (_dir, db, schema) = open_merge();
            let staging = TempDir::new().unwrap();

            let mut batch = db.shard_batch();
            batch.merge(&schema.counters, &U64Be(1), &U64Be(7)).unwrap();
            let mut finalize = batch
                .finalize_for_shard(staging.path(), &rocksdb::Options::default())
                .unwrap();

            let marker = crate::RestoreState::Complete { restored_at: 42 };
            db.stage_restore_state(finalize.write_batch_mut(), "p", &marker)
                .unwrap();
            finalize.commit().unwrap();

            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(7)));
            assert_eq!(db.restore_state("p").unwrap(), Some(marker));
        }
    }
}
