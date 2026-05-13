// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Bridge between [`sui_consistent_store`]'s `Pipeline` trait and
//! [`sui_indexer_alt_framework_store_traits`]'s `Store` /
//! `Connection` traits, so a pipeline written once can be driven
//! by the indexer-alt framework with no extra plumbing.
//!
//! # Status
//!
//! In active build-out. This commit lands the crate skeleton plus
//! the framework-side watermark schema (`Watermark` + the internal
//! column families that hold per-pipeline watermarks and
//! per-pipeline chain ids). The pieces that depend on the
//! framework's store traits — `Store`, `Connection`,
//! `SequentialStore`, `SequentialConnection`, and the blanket
//! `Processor` + `sequential::Handler` impls for any `Pipeline`
//! — and the cross-pipeline snapshot `Synchronizer` follow in
//! subsequent commits. Until those land, this crate's surface is
//! just the watermark types and schema; it does not yet bridge
//! anything end-to-end.
//!
//! # Layering
//!
//! The crate sits *above* [`sui_consistent_store`] and *below* the
//! indexer-alt framework. Pipelines that consumers want to drive
//! from the framework define themselves once against
//! [`sui_consistent_store::Pipeline`]; the framework-side adapter
//! lives entirely in this crate so the foundational crate stays
//! free of indexer-alt-specific concerns.

pub mod schema;
pub mod store;
pub mod watermark;

pub use crate::schema::CHAIN_ID_CF;
pub use crate::schema::FrameworkSchema;
pub use crate::schema::WATERMARK_CF;
pub use crate::store::Connection;
pub use crate::store::Store;
pub use crate::watermark::Watermark;
