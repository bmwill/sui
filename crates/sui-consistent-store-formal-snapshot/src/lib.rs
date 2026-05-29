// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Formal-snapshot restore for [`sui_consistent_store`].
//!
//! The crate ships four modules:
//!
//! - [`snapshot_format`]: pure parsers for the formal-snapshot wire
//!   format (root manifest, per-epoch manifest, live-objects file).
//! - [`storage`]: a minimal byte-fetching abstraction with impls
//!   over [`object_store`] backends and raw HTTP.
//! - [`restore_runner`]: the shared bulk-load lifecycle
//!   ([`RestoreRunner`]) that drives one pipeline through a stream
//!   of objects and persists per-partition progress.
//! - [`formal_snapshot`]: the [`FormalSnapshot`] handle on an
//!   opened snapshot plus the
//!   [`restore_pipeline_from_formal_snapshot`] driver that feeds
//!   the snapshot's objects through a [`RestoreRunner`].

pub mod formal_snapshot;
pub mod restore_runner;
pub mod snapshot_format;
pub mod storage;

pub use crate::formal_snapshot::FormalSnapshot;
pub use crate::formal_snapshot::restore_pipeline_from_formal_snapshot;
pub use crate::restore_runner::RestoreRunner;
