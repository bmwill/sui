// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Foundational, type-safe wrapper around RocksDB for Sui's on-disk
//! indexes.
//!
//! See `PLAN.md` at the crate root for the design and the implementation
//! roadmap. The crate is in active build-out; the encoding traits and
//! the database wrapper land first, followed by typed column-family
//! access, batched writes, iteration, and the in-memory snapshot model.

pub mod db;
pub mod encode;
pub mod error;
pub mod schema;

pub use crate::db::Db;
pub use crate::db::DbOptions;
pub use crate::encode::Decode;
pub use crate::encode::Encode;
pub use crate::schema::Schema;
