// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Foundational, type-safe wrapper around RocksDB for Sui's on-disk
//! indexes.
//!
//! See `PLAN.md` at the crate root for the design and the
//! implementation roadmap. The crate is in active build-out: the
//! encoding traits, the database wrapper, typed point reads, atomic
//! batched writes, typed iteration, and the in-memory snapshot
//! model have landed; typed merge operators and a filesystem
//! checkpoint helper follow.

pub mod batch;
pub mod db;
pub mod encode;
mod encode_buf;
pub mod error;
pub mod iter;
pub mod map;
pub mod schema;
pub mod snapshot;

pub use crate::batch::Batch;
pub use crate::db::Db;
pub use crate::db::DbOptions;
pub use crate::db::RocksMetrics;
pub use crate::encode::Decode;
pub use crate::encode::Encode;
pub use crate::iter::Iter;
pub use crate::iter::RevIter;
pub use crate::map::DbMap;
pub use crate::schema::Schema;
pub use crate::snapshot::SnapshotHandle;
