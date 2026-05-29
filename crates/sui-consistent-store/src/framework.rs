// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The auto-registered bookkeeping schema, [`FrameworkSchema`],
//! plus the on-disk types it persists.
//!
//! Three column families are owned by the framework rather than by
//! any consumer's schema, so the crate registers them automatically
//! on every [`Db::open`](crate::Db::open) and exposes them as a
//! cheaply-constructible [`FrameworkSchema`] handle:
//!
//! - [`RESTORE_CF`] (`PipelineTaskKey → RestoreState`) — per-pipeline
//!   restore progress; used by [`RestoreRunner`](crate::RestoreRunner)
//!   to skip already-ingested partitions on resume.
//! - [`WATERMARK_CF`] (`PipelineTaskKey → Watermark`) — per-pipeline
//!   committer watermark; used by tip-mode drivers to learn what
//!   checkpoint each pipeline resumes from.
//! - [`CHAIN_ID_CF`] (`PipelineTaskKey → ChainId`) — per-pipeline
//!   chain identifier; used by tip-mode drivers to refuse
//!   checkpoints from a different chain than the pipeline was
//!   originally bound to.
//!
//! The double-underscore prefix marks each CF as crate-internal so
//! user schemas avoid colliding on the name.
//!
//! # Access
//!
//! Hold a [`Db`] (or [`Snapshot`](crate::Snapshot)) and call
//! [`framework`](Db::framework) /
//! [`framework`](crate::Snapshot::framework) to obtain a
//! `FrameworkSchema<&Db>` / `FrameworkSchema<&Snapshot>`. Both
//! return values are zero-`Arc`-bump — three [`DbMap`]s borrowing
//! the same reader — and scoped to the borrow that produced them.
//!
//! For an owned handle (e.g. to hold inside a longer-lived
//! [`Store`]-like struct, or to use with [`Batch`](crate::Batch)'s
//! typed writes), construct with
//! [`FrameworkSchema::new(db.clone())`](FrameworkSchema::new).

use std::collections::BTreeSet;

use bytes::Buf;
use bytes::BufMut;

use crate::Decode;
use crate::Encode;
use crate::db::Db;
use crate::error::DecodeError;
use crate::error::EncodeError;
use crate::map::DbMap;
use crate::reader::Reader;

/// Name of the column family holding per-pipeline [`RestoreState`]
/// entries. Crate-internal; user code reaches the CF through
/// [`FrameworkSchema::restore`].
pub(crate) const RESTORE_CF: &str = "__restore";

/// Name of the column family holding per-pipeline [`Watermark`]s.
/// Crate-internal; user code reaches the CF through
/// [`FrameworkSchema::watermarks`].
pub(crate) const WATERMARK_CF: &str = "__watermark";

/// Name of the column family holding per-pipeline [`ChainId`]s.
/// Crate-internal; user code reaches the CF through
/// [`FrameworkSchema::chain_ids`].
pub(crate) const CHAIN_ID_CF: &str = "__chain_id";

/// Typed `pipeline_task` key used by the framework's internal CFs.
///
/// Encoded as raw UTF-8 bytes — the framework CFs are internal, so
/// we pick the simplest representation. Decoding produces an owned
/// `String`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineTaskKey(pub String);

impl PipelineTaskKey {
    /// Build a key from any string-ish input.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl Encode for PipelineTaskKey {
    fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
        buf.put_slice(self.0.as_bytes());
        Ok(())
    }
}

impl Decode for PipelineTaskKey {
    fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
        let mut bytes = vec![0u8; buf.remaining()];
        buf.copy_to_slice(&mut bytes);
        let s = String::from_utf8(bytes)
            .map_err(|e| DecodeError::with_source("PipelineTaskKey not valid UTF-8", e))?;
        Ok(Self(s))
    }
}

/// Per-pipeline restore progress, persisted in [`RESTORE_CF`].
///
/// Drivers transition a pipeline:
/// 1. `None` → `InProgress { partitions_complete: empty, target_checkpoint: T }`
///    when restore begins.
/// 2. `InProgress` → `InProgress` with one more partition marked
///    complete, atomically with each shard's data writes.
/// 3. `InProgress` → `Complete { restored_at: T }` when every
///    partition has been committed.
///
/// Tip indexing for a pipeline must wait until its state reaches
/// `Complete`. Drivers check this on startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreState {
    /// Restore has started for this pipeline.
    ///
    /// `partitions_complete` enumerates which of the driver's
    /// shards or partitions have been atomically ingested. The
    /// driver decides what bytes identify a partition; this crate
    /// treats them as opaque.
    ///
    /// `target_checkpoint` is the checkpoint number the restore is
    /// aimed at. The restore source dictates this value (the
    /// formal snapshot's anchor checkpoint, or the validator's
    /// genesis-up-to checkpoint), and tip indexing resumes at
    /// `target_checkpoint + 1` once the pipeline reaches
    /// [`Complete`](RestoreState::Complete).
    InProgress {
        target_checkpoint: u64,
        partitions_complete: BTreeSet<Vec<u8>>,
    },
    /// Restore has finished for this pipeline.
    ///
    /// `restored_at` is the checkpoint number the pipeline was
    /// restored to. Tip indexing resumes at `restored_at + 1`.
    Complete { restored_at: u64 },
}

/// Tag byte distinguishing [`RestoreState::InProgress`] from
/// [`RestoreState::Complete`] on the wire.
const TAG_IN_PROGRESS: u8 = 0;
const TAG_COMPLETE: u8 = 1;

impl Encode for RestoreState {
    fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
        match self {
            Self::InProgress {
                target_checkpoint,
                partitions_complete,
            } => {
                buf.put_u8(TAG_IN_PROGRESS);
                buf.put_u64(*target_checkpoint);
                let count: u32 = partitions_complete
                    .len()
                    .try_into()
                    .map_err(|_| EncodeError::msg("partitions_complete count exceeds u32::MAX"))?;
                buf.put_u32(count);
                for partition in partitions_complete {
                    let len: u32 = partition
                        .len()
                        .try_into()
                        .map_err(|_| EncodeError::msg("partition id exceeds u32::MAX bytes"))?;
                    buf.put_u32(len);
                    buf.put_slice(partition);
                }
            }
            Self::Complete { restored_at } => {
                buf.put_u8(TAG_COMPLETE);
                buf.put_u64(*restored_at);
            }
        }
        Ok(())
    }
}

impl Decode for RestoreState {
    fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
        if buf.remaining() < 1 {
            return Err(DecodeError::msg("RestoreState missing tag byte"));
        }
        let tag = buf.get_u8();
        match tag {
            TAG_IN_PROGRESS => {
                if buf.remaining() < 8 + 4 {
                    return Err(DecodeError::msg(
                        "RestoreState::InProgress truncated header",
                    ));
                }
                let target_checkpoint = buf.get_u64();
                let count = buf.get_u32() as usize;
                let mut partitions_complete = BTreeSet::new();
                for _ in 0..count {
                    if buf.remaining() < 4 {
                        return Err(DecodeError::msg(
                            "RestoreState::InProgress truncated partition length",
                        ));
                    }
                    let len = buf.get_u32() as usize;
                    if buf.remaining() < len {
                        return Err(DecodeError::msg(
                            "RestoreState::InProgress truncated partition bytes",
                        ));
                    }
                    let mut partition = vec![0u8; len];
                    buf.copy_to_slice(&mut partition);
                    if !partitions_complete.insert(partition) {
                        return Err(DecodeError::msg(
                            "RestoreState::InProgress duplicate partition id",
                        ));
                    }
                }
                if buf.has_remaining() {
                    return Err(DecodeError::msg(
                        "RestoreState::InProgress trailing bytes after partition list",
                    ));
                }
                Ok(Self::InProgress {
                    target_checkpoint,
                    partitions_complete,
                })
            }
            TAG_COMPLETE => {
                if buf.remaining() != 8 {
                    return Err(DecodeError::msg(
                        "RestoreState::Complete wrong length after tag",
                    ));
                }
                let restored_at = buf.get_u64();
                Ok(Self::Complete { restored_at })
            }
            other => Err(DecodeError::msg(format!(
                "RestoreState unknown tag byte: {other}"
            ))),
        }
    }
}

/// Per-pipeline committer watermark, persisted in [`WATERMARK_CF`].
///
/// Holds the highest checkpoint each pipeline has committed plus
/// the corresponding epoch / tx / timestamp. Tip-mode drivers
/// advance this atomically with each pipeline's data writes, and
/// read it on restart to decide where to resume.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Watermark {
    /// Highest epoch number observed at or before
    /// `checkpoint_hi_inclusive`.
    pub epoch_hi_inclusive: u64,
    /// Inclusive upper bound of checkpoints whose effects are
    /// persisted for this pipeline.
    pub checkpoint_hi_inclusive: u64,
    /// Network-total transaction count at
    /// `checkpoint_hi_inclusive`.
    pub tx_hi: u64,
    /// Wall-clock timestamp (ms since epoch) of the checkpoint at
    /// `checkpoint_hi_inclusive`.
    pub timestamp_ms_hi_inclusive: u64,
}

/// On-wire size of a [`Watermark`]. Used by both the encoder and
/// the decoder to validate buffer sizes precisely.
const WATERMARK_WIRE_SIZE: usize = 4 * 8;

impl Encode for Watermark {
    fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
        buf.put_u64(self.epoch_hi_inclusive);
        buf.put_u64(self.checkpoint_hi_inclusive);
        buf.put_u64(self.tx_hi);
        buf.put_u64(self.timestamp_ms_hi_inclusive);
        Ok(())
    }
}

impl Decode for Watermark {
    fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
        if buf.remaining() != WATERMARK_WIRE_SIZE {
            return Err(DecodeError::msg(format!(
                "Watermark wire size mismatch: expected {WATERMARK_WIRE_SIZE} bytes, got {}",
                buf.remaining()
            )));
        }
        Ok(Self {
            epoch_hi_inclusive: buf.get_u64(),
            checkpoint_hi_inclusive: buf.get_u64(),
            tx_hi: buf.get_u64(),
            timestamp_ms_hi_inclusive: buf.get_u64(),
        })
    }
}

/// Typed chain-identifier value (`[u8; 32]`) persisted in
/// [`CHAIN_ID_CF`].
///
/// Tip-mode drivers store the chain id the pipeline was first
/// bound to so the framework can refuse checkpoints from a
/// different chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainId(pub [u8; 32]);

impl Encode for ChainId {
    fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
        buf.put_slice(&self.0);
        Ok(())
    }
}

impl Decode for ChainId {
    fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
        if buf.remaining() != 32 {
            return Err(DecodeError::msg("ChainId wire size mismatch"));
        }
        let mut id = [0u8; 32];
        buf.copy_to_slice(&mut id);
        Ok(Self(id))
    }
}

/// Typed handles into the framework's auto-registered column
/// families.
///
/// The three CFs ([`RESTORE_CF`], [`WATERMARK_CF`], [`CHAIN_ID_CF`])
/// are registered automatically by
/// [`Db::open`](crate::Db::open), so this schema does not need to
/// be declared by user schemas. Obtain a borrowed handle via
/// [`Db::framework`] or [`Snapshot::framework`](crate::Snapshot::framework);
/// construct an owned one via [`FrameworkSchema::new`].
///
/// `R` defaults to [`Db`] for symmetry with [`DbMap`].
pub struct FrameworkSchema<R: Reader + Clone = Db> {
    /// Per-pipeline [`RestoreState`] entries. See [`RESTORE_CF`].
    pub restore: DbMap<PipelineTaskKey, RestoreState, R>,
    /// Per-pipeline [`Watermark`] entries. See [`WATERMARK_CF`].
    pub watermarks: DbMap<PipelineTaskKey, Watermark, R>,
    /// Per-pipeline [`ChainId`] entries. See [`CHAIN_ID_CF`].
    pub chain_ids: DbMap<PipelineTaskKey, ChainId, R>,
}

impl<R: Reader + Clone> FrameworkSchema<R> {
    /// Construct a `FrameworkSchema` bound to `reader`.
    ///
    /// The constructor is infallible because the three framework
    /// CFs are auto-registered by [`Db::open`](crate::Db::open), so
    /// the CF-existence check that [`DbMap::new`] performs is
    /// redundant here. Each field clones `reader` once.
    pub fn new(reader: R) -> Self {
        Self {
            restore: DbMap::new_unchecked(reader.clone(), RESTORE_CF),
            watermarks: DbMap::new_unchecked(reader.clone(), WATERMARK_CF),
            chain_ids: DbMap::new_unchecked(reader, CHAIN_ID_CF),
        }
    }
}

impl<R: Reader + Clone> std::fmt::Debug for FrameworkSchema<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameworkSchema")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::DbOptions;
    use crate::Schema;
    use crate::error::OpenError;

    /// Minimal user schema (no extra CFs) used to open a database
    /// purely to exercise the auto-registered framework CFs.
    #[derive(Debug)]
    struct EmptySchema;

    impl Schema for EmptySchema {
        fn cfs(_: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            vec![]
        }

        fn open(_: &Db) -> Result<Self, OpenError> {
            Ok(Self)
        }
    }

    fn open() -> (TempDir, Db) {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<EmptySchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db)
    }

    #[test]
    fn framework_cfs_are_auto_registered() {
        let (_dir, db) = open();
        assert!(db.cf_handle(RESTORE_CF).is_some());
        assert!(db.cf_handle(WATERMARK_CF).is_some());
        assert!(db.cf_handle(CHAIN_ID_CF).is_some());
    }

    #[test]
    fn db_framework_returns_borrowed_schema() {
        let (_dir, db) = open();
        let fw = db.framework();
        // No data written; reads return None.
        let key = PipelineTaskKey::new("balances");
        assert!(fw.restore.get(&key).unwrap().is_none());
        assert!(fw.watermarks.get(&key).unwrap().is_none());
        assert!(fw.chain_ids.get(&key).unwrap().is_none());
    }

    #[test]
    fn owned_framework_round_trips_watermark() {
        let (_dir, db) = open();
        let fw = FrameworkSchema::new(db.clone());
        let key = PipelineTaskKey::new("p");
        let w = Watermark {
            epoch_hi_inclusive: 3,
            checkpoint_hi_inclusive: 42,
            tx_hi: 99,
            timestamp_ms_hi_inclusive: 1_700_000_000_000,
        };

        let mut batch = db.batch();
        batch.put(&fw.watermarks, &key, &w).unwrap();
        batch.commit().unwrap();

        assert_eq!(db.framework().watermarks.get(&key).unwrap(), Some(w));
    }

    #[test]
    fn owned_framework_round_trips_chain_id() {
        let (_dir, db) = open();
        let fw = FrameworkSchema::new(db.clone());
        let key = PipelineTaskKey::new("p");
        let chain_id = ChainId([7u8; 32]);

        let mut batch = db.batch();
        batch.put(&fw.chain_ids, &key, &chain_id).unwrap();
        batch.commit().unwrap();

        assert_eq!(db.framework().chain_ids.get(&key).unwrap(), Some(chain_id));
    }

    #[test]
    fn owned_framework_round_trips_restore_state() {
        let (_dir, db) = open();
        let fw = FrameworkSchema::new(db.clone());
        let key = PipelineTaskKey::new("p");
        let state = RestoreState::Complete { restored_at: 7 };

        let mut batch = db.batch();
        batch.put(&fw.restore, &key, &state).unwrap();
        batch.commit().unwrap();

        assert_eq!(db.framework().restore.get(&key).unwrap(), Some(state));
    }

    #[test]
    fn pipeline_task_key_round_trips() {
        let key = PipelineTaskKey::new("balances@indexer_a");
        let mut buf = Vec::new();
        key.encode_into(&mut buf).unwrap();
        let mut slice = buf.as_slice();
        let decoded = PipelineTaskKey::decode(&mut slice).unwrap();
        assert_eq!(decoded, key);
    }

    #[test]
    fn watermark_round_trips_through_db() {
        let w = Watermark {
            epoch_hi_inclusive: 3,
            checkpoint_hi_inclusive: 12345,
            tx_hi: 99,
            timestamp_ms_hi_inclusive: 1_700_000_000_000,
        };
        let mut buf = Vec::new();
        w.encode_into(&mut buf).unwrap();
        assert_eq!(buf.len(), WATERMARK_WIRE_SIZE);
        let mut slice = buf.as_slice();
        let decoded = Watermark::decode(&mut slice).unwrap();
        assert_eq!(decoded, w);
    }

    #[test]
    fn watermark_decode_rejects_short_buffer() {
        let bytes = [0u8; WATERMARK_WIRE_SIZE - 1];
        let mut slice = bytes.as_slice();
        let err = Watermark::decode(&mut slice).unwrap_err();
        assert!(format!("{err:#}").contains("wire size mismatch"));
    }

    #[test]
    fn watermark_decode_rejects_long_buffer() {
        let bytes = [0u8; WATERMARK_WIRE_SIZE + 1];
        let mut slice = bytes.as_slice();
        let err = Watermark::decode(&mut slice).unwrap_err();
        assert!(format!("{err:#}").contains("wire size mismatch"));
    }

    #[test]
    fn chain_id_decode_rejects_wrong_length() {
        let bytes = [0u8; 16];
        let mut slice = bytes.as_slice();
        let err = ChainId::decode(&mut slice).unwrap_err();
        assert!(format!("{err:#}").contains("wire size mismatch"));
    }

    #[test]
    fn default_watermark_has_zero_fields() {
        let w = Watermark::default();
        assert_eq!(w.epoch_hi_inclusive, 0);
        assert_eq!(w.checkpoint_hi_inclusive, 0);
        assert_eq!(w.tx_hi, 0);
        assert_eq!(w.timestamp_ms_hi_inclusive, 0);
    }

    #[test]
    fn snapshot_framework_reads_pre_snapshot_state() {
        // The borrowed-snapshot accessor returns a FrameworkSchema
        // whose reads see the captured snapshot state.
        let (_dir, db) = open();
        let key = PipelineTaskKey::new("p");
        let w = Watermark {
            checkpoint_hi_inclusive: 10,
            ..Watermark::default()
        };
        let fw = FrameworkSchema::new(db.clone());
        let mut batch = db.batch();
        batch.put(&fw.watermarks, &key, &w).unwrap();
        batch.commit().unwrap();

        db.take_snapshot(1);

        let w2 = Watermark {
            checkpoint_hi_inclusive: 999,
            ..Watermark::default()
        };
        let mut batch = db.batch();
        batch.put(&fw.watermarks, &key, &w2).unwrap();
        batch.commit().unwrap();

        let snap = db.at_snapshot(1).unwrap();
        let fw_snap = snap.framework();
        assert_eq!(fw_snap.watermarks.get(&key).unwrap(), Some(w));
    }

    // RestoreState wire format tests.

    fn round_trip(state: &RestoreState) -> RestoreState {
        let mut buf = Vec::new();
        state.encode_into(&mut buf).unwrap();
        let mut slice = buf.as_slice();
        let decoded = RestoreState::decode(&mut slice).unwrap();
        assert!(
            !slice.has_remaining(),
            "decode did not consume the full buffer",
        );
        decoded
    }

    #[test]
    fn restore_state_round_trip_complete() {
        let s = RestoreState::Complete {
            restored_at: 12_345,
        };
        assert_eq!(round_trip(&s), s);
    }

    #[test]
    fn restore_state_round_trip_in_progress_empty() {
        let s = RestoreState::InProgress {
            target_checkpoint: 999,
            partitions_complete: BTreeSet::new(),
        };
        assert_eq!(round_trip(&s), s);
    }

    #[test]
    fn restore_state_round_trip_in_progress_with_partitions() {
        let mut partitions = BTreeSet::new();
        partitions.insert(vec![0u8, 1, 2, 3]);
        partitions.insert(b"shard-7".to_vec());
        partitions.insert(vec![]); // Zero-length partition id is allowed.
        let s = RestoreState::InProgress {
            target_checkpoint: 1,
            partitions_complete: partitions,
        };
        assert_eq!(round_trip(&s), s);
    }

    #[test]
    fn restore_state_decode_rejects_unknown_tag() {
        let bytes = [42u8];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("unknown tag"));
    }

    #[test]
    fn restore_state_decode_rejects_empty_buffer() {
        let bytes: [u8; 0] = [];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("missing tag"));
    }

    #[test]
    fn restore_state_decode_rejects_truncated_in_progress_header() {
        let bytes = [TAG_IN_PROGRESS];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("truncated header"));
    }

    #[test]
    fn restore_state_decode_rejects_truncated_partition_length() {
        let mut bytes = vec![TAG_IN_PROGRESS];
        bytes.extend_from_slice(&1u64.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes()); // claim 1 partition
        // No partition-length bytes follow.
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("truncated partition length"));
    }

    #[test]
    fn restore_state_decode_rejects_truncated_partition_bytes() {
        let mut bytes = vec![TAG_IN_PROGRESS];
        bytes.extend_from_slice(&1u64.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes()); // claim 1 partition
        bytes.extend_from_slice(&10u32.to_be_bytes()); // ...of 10 bytes
        bytes.extend_from_slice(&[1, 2, 3]); // ...but only 3 follow.
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("truncated partition bytes"));
    }

    #[test]
    fn restore_state_decode_rejects_trailing_bytes_in_in_progress() {
        let mut bytes = vec![TAG_IN_PROGRESS];
        bytes.extend_from_slice(&0u64.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes()); // zero partitions
        bytes.push(0xFF); // unexpected trailing byte
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn restore_state_decode_rejects_wrong_length_for_complete() {
        let bytes = [TAG_COMPLETE, 1, 2];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("wrong length"));
    }
}
