// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The internal `__restore` column family and the [`RestoreState`]
//! typed accessor on it.
//!
//! Restore is one-shot — there is no resume from an arbitrary
//! mid-restore point at object granularity — but process death
//! between partitions must not waste already-ingested work. The
//! `__restore` CF persists per-pipeline progress markers so the
//! restore driver can detect partial restores on startup and skip
//! the partitions that have already been ingested.
//!
//! # Schema visibility
//!
//! The CF is registered automatically by
//! [`Db::open`](crate::Db::open) alongside the schema's CFs (the
//! same way the `default` CF is auto-registered). Schema authors
//! do not declare it; the typed accessor methods on [`Db`] are the
//! only way to read or write it.
//!
//! # Partition IDs are opaque bytes
//!
//! Different restore drivers use different partition identifiers
//! (formal-snapshot partition number, `ObjectID`-range shard
//! index, etc.). This crate's role is to persist them, not to
//! interpret them, so partition IDs round-trip as `Vec<u8>`.

use std::collections::BTreeSet;

use bytes::Buf;
use bytes::BufMut;

use crate::Decode;
use crate::Encode;
use crate::error::DecodeError;
use crate::error::EncodeError;

/// Name of the internal column family that holds per-pipeline
/// [`RestoreState`] entries.
///
/// The double-underscore prefix marks this as crate-internal so
/// schemas avoid colliding on the name.
pub const RESTORE_CF: &str = "__restore";

/// Per-pipeline restore progress, persisted in the `__restore`
/// column family.
///
/// Drivers transition a pipeline:
/// 1. `None` → `InProgress { partitions_complete: empty, target_checkpoint: T }`
///    when restore begins.
/// 2. `InProgress` → `InProgress` with one more partition marked
///    complete, after each `Db::ingest_files_cf` for that
///    partition succeeds.
/// 3. `InProgress` → `Complete { restored_at: T }` when every
///    partition has been ingested.
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
                let count: u32 = partitions_complete.len().try_into().map_err(|_| {
                    EncodeError::msg("partitions_complete count exceeds u32::MAX")
                })?;
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
                    return Err(DecodeError::msg("RestoreState::InProgress truncated header"));
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn round_trip_complete() {
        let s = RestoreState::Complete {
            restored_at: 12_345,
        };
        assert_eq!(round_trip(&s), s);
    }

    #[test]
    fn round_trip_in_progress_empty() {
        let s = RestoreState::InProgress {
            target_checkpoint: 999,
            partitions_complete: BTreeSet::new(),
        };
        assert_eq!(round_trip(&s), s);
    }

    #[test]
    fn round_trip_in_progress_with_partitions() {
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
    fn decode_rejects_unknown_tag() {
        let bytes = [42u8];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("unknown tag"));
    }

    #[test]
    fn decode_rejects_empty_buffer() {
        let bytes: [u8; 0] = [];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("missing tag"));
    }

    #[test]
    fn decode_rejects_truncated_in_progress_header() {
        let bytes = [TAG_IN_PROGRESS];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("truncated header"));
    }

    #[test]
    fn decode_rejects_truncated_partition_length() {
        let mut bytes = vec![TAG_IN_PROGRESS];
        bytes.extend_from_slice(&1u64.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes()); // claim 1 partition
        // No partition-length bytes follow.
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("truncated partition length"));
    }

    #[test]
    fn decode_rejects_truncated_partition_bytes() {
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
    fn decode_rejects_trailing_bytes_in_in_progress() {
        let mut bytes = vec![TAG_IN_PROGRESS];
        bytes.extend_from_slice(&0u64.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes()); // zero partitions
        bytes.push(0xFF); // unexpected trailing byte
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn decode_rejects_wrong_length_for_complete() {
        let bytes = [TAG_COMPLETE, 1, 2];
        let mut slice = bytes.as_slice();
        let err = RestoreState::decode(&mut slice).unwrap_err();
        assert!(err.to_string().contains("wrong length"));
    }
}
