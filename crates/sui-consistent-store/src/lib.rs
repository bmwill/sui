// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Foundational, type-safe wrapper around RocksDB for Sui's on-disk
//! indexes.
//!
//! See `PLAN.md` at the crate root for the design and the implementation
//! roadmap. The crate is currently a skeleton; subsequent commits land
//! the encoding traits, the database wrapper, typed column-family access,
//! batched writes, iteration, and the in-memory snapshot model.
