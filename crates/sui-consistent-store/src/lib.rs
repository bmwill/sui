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
//!
//! [`rocksdb`] is re-exported so consumers can construct the
//! [`rocksdb::Options`] / [`rocksdb::WriteOptions`] values our
//! public API takes without adding a direct rocksdb dependency
//! themselves.
pub use rocksdb;

pub mod batch;
pub mod db;
pub mod encode;
mod encode_buf;
pub mod error;
pub mod formal_snapshot;
pub mod framework;
pub mod iter;
pub mod map;
pub mod object_source;
pub mod pipeline;
pub mod reader;
pub mod restore_runner;
pub mod schema;
pub mod snapshot;
pub mod snapshot_format;
pub mod storage;

pub use crate::batch::Batch;
pub use crate::db::Db;
pub use crate::db::DbOptions;
pub use crate::db::RocksMetrics;
pub use crate::encode::Decode;
pub use crate::encode::Encode;
pub use crate::formal_snapshot::FormalSnapshot;
pub use crate::framework::ChainId;
pub use crate::framework::FrameworkSchema;
pub use crate::framework::PipelineTaskKey;
pub use crate::framework::RestoreState;
pub use crate::framework::Watermark;
pub use crate::iter::Iter;
pub use crate::iter::RevIter;
pub use crate::map::DbMap;
pub use crate::object_source::LiveObjectSource;
pub use crate::object_source::restore_pipeline_from_object_source;
pub use crate::pipeline::Pipeline;
pub use crate::reader::Reader;
pub use crate::restore_runner::RestoreRunner;
pub use crate::schema::CfDescriptor;
pub use crate::schema::Schema;
pub use crate::schema::SchemaAtSnapshot;
pub use crate::snapshot::Snapshot;
