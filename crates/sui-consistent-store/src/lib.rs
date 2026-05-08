// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Foundational, type-safe wrapper around RocksDB for Sui's on-disk
//! indexes.
//!
//! See `PLAN.md` at the crate root for the design and the implementation
//! roadmap. The crate is in active build-out; the encoding traits land
//! first, followed by the database wrapper, typed column-family access,
//! batched writes, iteration, and the in-memory snapshot model.

pub mod encode;
pub mod error;

pub use crate::encode::Decode;
pub use crate::encode::Encode;
