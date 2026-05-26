// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Conversion helpers between [`sui_consistent_store::Watermark`]
//! (the on-disk representation persisted in
//! [`WATERMARK_CF`](sui_consistent_store::WATERMARK_CF)) and
//! [`CommitterWatermark`](sui_indexer_alt_framework_store_traits::CommitterWatermark)
//! (the indexer framework's per-pipeline progress type).
//!
//! The two types share the same four fields. We can't supply `From`
//! impls because neither type is local to this crate (orphan rule),
//! so the conversion is exposed as plain functions instead.

use sui_consistent_store::Watermark;
use sui_indexer_alt_framework_store_traits::CommitterWatermark;

/// Convert an on-disk [`Watermark`] into the framework's
/// [`CommitterWatermark`].
pub(crate) fn to_committer(w: Watermark) -> CommitterWatermark {
    CommitterWatermark {
        epoch_hi_inclusive: w.epoch_hi_inclusive,
        checkpoint_hi_inclusive: w.checkpoint_hi_inclusive,
        tx_hi: w.tx_hi,
        timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
    }
}

/// Convert a framework [`CommitterWatermark`] into the on-disk
/// [`Watermark`].
pub(crate) fn from_committer(w: CommitterWatermark) -> Watermark {
    Watermark {
        epoch_hi_inclusive: w.epoch_hi_inclusive,
        checkpoint_hi_inclusive: w.checkpoint_hi_inclusive,
        tx_hi: w.tx_hi,
        timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committer_watermark_conversion_is_bijective() {
        let w = Watermark {
            epoch_hi_inclusive: 7,
            checkpoint_hi_inclusive: 42,
            tx_hi: 1_000,
            timestamp_ms_hi_inclusive: 1_700_000_000_000,
        };
        let cw = to_committer(w);
        let back = from_committer(cw);
        assert_eq!(back, w);
    }
}
