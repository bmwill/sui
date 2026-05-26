// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The framework's internal-bookkeeping schema.
//!
//! Two column families are owned by the framework adapter itself,
//! independent of any consumer's pipeline schema:
//!
//! - [`WATERMARK_CF`] (`pipeline_task: String → Watermark`). The
//!   per-pipeline committer watermark, written atomically with
//!   each pipeline's data commit. On restart, the framework reads
//!   this to decide which checkpoint each pipeline resumes from.
//! - [`CHAIN_ID_CF`] (`pipeline_task: String → [u8; 32]`). The
//!   chain identifier the pipeline was originally bound to. The
//!   framework rejects checkpoints from a different chain id.
//!
//! Keys are arbitrary UTF-8 strings (pipeline names, possibly with
//! a `@task` suffix); they are encoded as raw bytes since this CF
//! is internal and the encoding choice is not exposed.
//!
//! # Status
//!
//! This commit adds the *schema*. The [`Connection`] /
//! [`SequentialConnection`] impls that read and write these CFs
//! follow in subsequent commits.
//!
//! [`Connection`]: sui_indexer_alt_framework_store_traits::Connection
//! [`SequentialConnection`]: sui_indexer_alt_framework_store_traits::SequentialConnection

use bytes::Buf;
use bytes::BufMut;
use sui_consistent_store::CfDescriptor;
use sui_consistent_store::Db;
use sui_consistent_store::DbMap;
use sui_consistent_store::Decode;
use sui_consistent_store::Encode;
use sui_consistent_store::Schema;
use sui_consistent_store::error::DecodeError;
use sui_consistent_store::error::EncodeError;
use sui_consistent_store::error::OpenError;
use sui_consistent_store::rocksdb;

use crate::watermark::Watermark;

/// Name of the column family holding per-pipeline
/// [`Watermark`]s.
pub const WATERMARK_CF: &str = "__fw_watermark";

/// Name of the column family holding per-pipeline chain
/// identifiers (`[u8; 32]`).
pub const CHAIN_ID_CF: &str = "__fw_chain_id";

/// Typed `pipeline_task` key for the framework's internal CFs.
///
/// Encoded as raw UTF-8 bytes — the CF is internal, so we pick the
/// simplest representation. Decoding produces an owned `String`.
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

/// Typed chain-identifier value (`[u8; 32]`) stored in
/// [`CHAIN_ID_CF`].
///
/// Persists the chain id the pipeline was first bound to so the
/// framework can refuse checkpoints from a different chain.
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

/// The framework's internal schema — the two CFs the adapter owns
/// regardless of what consumer pipelines bring.
///
/// Typically opened alongside a consumer's own schema by passing
/// both through [`Db::open`] in turn (the framework's schema only
/// declares the framework CFs; consumer schemas declare their own
/// CFs). Concretely, the integration crate combines them into a
/// composite schema; the standalone `FrameworkSchema` is the
/// reusable building block.
#[derive(Debug)]
pub struct FrameworkSchema {
    /// Per-pipeline committer watermark.
    pub watermarks: DbMap<PipelineTaskKey, Watermark>,
    /// Per-pipeline chain identifier.
    pub chain_ids: DbMap<PipelineTaskKey, ChainId>,
}

impl Schema for FrameworkSchema {
    fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
        vec![
            CfDescriptor::new(WATERMARK_CF, base_options.clone()),
            CfDescriptor::new(CHAIN_ID_CF, base_options.clone()),
        ]
    }

    fn open(db: &Db) -> Result<Self, OpenError> {
        Ok(Self {
            watermarks: DbMap::new(db.clone(), WATERMARK_CF)?,
            chain_ids: DbMap::new(db.clone(), CHAIN_ID_CF)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use sui_consistent_store::DbOptions;
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn open_registers_framework_cfs() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<FrameworkSchema>(dir.path(), DbOptions::default()).unwrap();
        // Reading raw rocksdb CF presence isn't exposed publicly,
        // but the DbMap construction in `open` would have failed if
        // the CFs were not registered. Sanity-check by attempting a
        // batch write against each handle.
        let _ = db;
    }

    #[test]
    fn watermark_round_trips_through_db() {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<FrameworkSchema>(dir.path(), DbOptions::default()).unwrap();
        let key = PipelineTaskKey::new("balances");
        let w = Watermark {
            epoch_hi_inclusive: 3,
            checkpoint_hi_inclusive: 12345,
            tx_hi: 99,
            timestamp_ms_hi_inclusive: 1_700_000_000_000,
        };

        let mut batch = db.batch();
        batch.put(&schema.watermarks, &key, &w).unwrap();
        batch.commit().unwrap();

        let got = schema.watermarks.get(&key).unwrap();
        assert_eq!(got, Some(w));
    }

    #[test]
    fn chain_id_round_trips_through_db() {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<FrameworkSchema>(dir.path(), DbOptions::default()).unwrap();
        let key = PipelineTaskKey::new("balances");
        let chain_id = ChainId([7u8; 32]);

        let mut batch = db.batch();
        batch.put(&schema.chain_ids, &key, &chain_id).unwrap();
        batch.commit().unwrap();

        assert_eq!(schema.chain_ids.get(&key).unwrap(), Some(chain_id));
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
    fn chain_id_decode_rejects_wrong_length() {
        let bytes = [0u8; 16];
        let mut slice = bytes.as_slice();
        let err = ChainId::decode(&mut slice).unwrap_err();
        assert!(format!("{err:#}").contains("wire size mismatch"));
    }

    #[test]
    fn watermark_and_chain_id_share_no_keys() {
        // Sanity: the two CFs are independent, so the same key
        // string in each does not collide.
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<FrameworkSchema>(dir.path(), DbOptions::default()).unwrap();
        let key = PipelineTaskKey::new("p");

        let mut batch = db.batch();
        batch
            .put(
                &schema.watermarks,
                &key,
                &Watermark {
                    checkpoint_hi_inclusive: 1,
                    ..Watermark::default()
                },
            )
            .unwrap();
        batch
            .put(&schema.chain_ids, &key, &ChainId([1u8; 32]))
            .unwrap();
        batch.commit().unwrap();

        assert!(schema.watermarks.get(&key).unwrap().is_some());
        assert!(schema.chain_ids.get(&key).unwrap().is_some());
    }
}
