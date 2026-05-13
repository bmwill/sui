// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`Watermark`] — the framework-side per-pipeline progress
//! marker. Equivalent on the wire to
//! [`CommitterWatermark`](sui_indexer_alt_framework_store_traits::CommitterWatermark),
//! but with [`Encode`](sui_consistent_store::Encode) /
//! [`Decode`](sui_consistent_store::Decode) impls so it lives
//! cleanly in a typed [`DbMap`](sui_consistent_store::DbMap).
//!
//! The framework's `Connection::set_committer_watermark` updates a
//! row in the watermark CF atomically with the pipeline's data
//! writes; on restart, the framework calls `committer_watermark`
//! to learn what checkpoint to resume from. This crate's future
//! `Connection` impl will route through the methods on
//! [`Watermark`].

use bytes::Buf;
use bytes::BufMut;
use sui_consistent_store::Decode;
use sui_consistent_store::Encode;
use sui_consistent_store::error::DecodeError;
use sui_consistent_store::error::EncodeError;
use sui_indexer_alt_framework_store_traits::CommitterWatermark;

/// Per-pipeline committer watermark, persisted in the framework's
/// internal watermark CF.
///
/// Trivially convertible to and from
/// [`CommitterWatermark`](sui_indexer_alt_framework_store_traits::CommitterWatermark)
/// — they share the same four fields. The crate-local type exists
/// so we can hang [`Encode`] / [`Decode`] impls on it without
/// depending on the framework crate from `sui-consistent-store`
/// itself.
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

impl From<Watermark> for CommitterWatermark {
    fn from(w: Watermark) -> Self {
        Self {
            epoch_hi_inclusive: w.epoch_hi_inclusive,
            checkpoint_hi_inclusive: w.checkpoint_hi_inclusive,
            tx_hi: w.tx_hi,
            timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
        }
    }
}

impl From<CommitterWatermark> for Watermark {
    fn from(w: CommitterWatermark) -> Self {
        Self {
            epoch_hi_inclusive: w.epoch_hi_inclusive,
            checkpoint_hi_inclusive: w.checkpoint_hi_inclusive,
            tx_hi: w.tx_hi,
            timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Watermark {
        Watermark {
            epoch_hi_inclusive: 7,
            checkpoint_hi_inclusive: 42,
            tx_hi: 1_000,
            timestamp_ms_hi_inclusive: 1_700_000_000_000,
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let w = sample();
        let mut buf = Vec::new();
        w.encode_into(&mut buf).unwrap();
        assert_eq!(buf.len(), WATERMARK_WIRE_SIZE);
        let mut slice = buf.as_slice();
        let decoded = Watermark::decode(&mut slice).unwrap();
        assert!(!slice.has_remaining());
        assert_eq!(decoded, w);
    }

    #[test]
    fn decode_rejects_short_buffer() {
        let bytes = [0u8; WATERMARK_WIRE_SIZE - 1];
        let mut slice = bytes.as_slice();
        let err = Watermark::decode(&mut slice).unwrap_err();
        assert!(format!("{err:#}").contains("wire size mismatch"));
    }

    #[test]
    fn decode_rejects_long_buffer() {
        let bytes = [0u8; WATERMARK_WIRE_SIZE + 1];
        let mut slice = bytes.as_slice();
        let err = Watermark::decode(&mut slice).unwrap_err();
        assert!(format!("{err:#}").contains("wire size mismatch"));
    }

    #[test]
    fn committer_watermark_conversion_is_bijective() {
        let w = sample();
        let cw: CommitterWatermark = w.into();
        let back: Watermark = cw.into();
        assert_eq!(back, w);
    }

    #[test]
    fn default_watermark_has_zero_fields() {
        let w = Watermark::default();
        assert_eq!(w.epoch_hi_inclusive, 0);
        assert_eq!(w.checkpoint_hi_inclusive, 0);
        assert_eq!(w.tx_hi, 0);
        assert_eq!(w.timestamp_ms_hi_inclusive, 0);
    }
}
