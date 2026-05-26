// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Schema`] trait used to register column families with the
//! database and to construct typed handles into them, plus the
//! [`SchemaAtSnapshot`] companion trait used to re-bind a schema at
//! a captured snapshot.
//!
//! Schemas are hand-written Rust structs whose fields are typed
//! handles into individual column families ([`DbMap<K, V, R>`](crate::DbMap)).
//! The struct is parameterized by a [`Reader`](crate::Reader) (defaulted
//! to [`Live`](crate::Live)) so the same schema body serves both the
//! live tip and snapshot-bound projections.
//!
//! [`Schema`] is implemented for the live variant (`MySchema<Live>`)
//! and pairs the static set of column families a schema requires
//! ([`Schema::cfs`]) with the constructor that builds the schema
//! struct from an opened database ([`Schema::open`]).
//!
//! [`SchemaAtSnapshot`] is a separate trait the schema author opts
//! into; it declares a `MySchema<Snapshot<'s>>` projection and a
//! one-line constructor that re-binds each field via
//! [`DbMap::at`](crate::DbMap::at).
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//!
//! use sui_consistent_store::CfDescriptor;
//! use sui_consistent_store::Db;
//! use sui_consistent_store::DbMap;
//! use sui_consistent_store::DbOptions;
//! use sui_consistent_store::Live;
//! use sui_consistent_store::Reader;
//! use sui_consistent_store::Schema;
//! use sui_consistent_store::SchemaAtSnapshot;
//! use sui_consistent_store::Snapshot;
//! use sui_consistent_store::SnapshotHandle;
//! use sui_consistent_store::error::OpenError;
//!
//! struct MySchema<R: Reader = Live> {
//!     _reader: std::marker::PhantomData<R>,
//!     _db: Arc<Db>,
//! }
//!
//! impl Schema for MySchema<Live> {
//!     fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
//!         vec![CfDescriptor::new("my_cf", base_options.clone())]
//!     }
//!
//!     fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
//!         Ok(Self {
//!             _reader: std::marker::PhantomData,
//!             _db: db.clone(),
//!         })
//!     }
//! }
//!
//! impl SchemaAtSnapshot for MySchema<Live> {
//!     type At<'s> = MySchema<Snapshot<'s>>;
//!     fn at<'s>(&'s self, _snap: &'s SnapshotHandle) -> Self::At<'s> {
//!         MySchema {
//!             _reader: std::marker::PhantomData,
//!             _db: self._db.clone(),
//!         }
//!     }
//! }
//!
//! let dir = tempfile::tempdir().unwrap();
//! let (_db, _schema) = Db::open::<MySchema>(dir.path(), DbOptions::default()).unwrap();
//! ```

use std::sync::Arc;

use crate::db::Db;
use crate::error::OpenError;
use crate::snapshot::SnapshotHandle;

/// Declares the column families a database needs and constructs the
/// typed handle struct against an opened database at the live tip.
///
/// Implementations are typically hand-written structs parameterized
/// by a [`Reader`](crate::Reader) (defaulted to [`Live`](crate::Live))
/// whose fields are typed column-family handles. The trait itself is
/// implemented only for the [`Live`](crate::Live) variant of the
/// schema; snapshot-bound variants are constructed by re-binding,
/// not by re-opening.
///
/// The trait has two responsibilities:
///
/// - [`cfs`](Self::cfs) returns the column families this schema
///   requires, with per-CF [`rocksdb::Options`]. It is called once at
///   open time, with the database-level base options as input. The
///   order of entries does not matter.
/// - [`open`](Self::open) constructs the schema struct against an
///   already-opened database. Each column family named by `cfs()` is
///   guaranteed to exist on the database before this is called.
pub trait Schema: Sized {
    /// The column families this schema requires.
    ///
    /// Each entry is a [`CfDescriptor`] carrying a column-family
    /// name (a `&'static str` so the schema's CF set is fixed at
    /// compile time), its [`rocksdb::Options`] applied at create
    /// time, and a [`RestoreMode`] that controls how restore-time
    /// shard writes are routed for the CF. `base_options` is
    /// supplied by [`Db::open`] and is the database-level options
    /// configured on [`DbOptions::db_options`](crate::DbOptions::db_options);
    /// implementations typically clone it as the starting point for
    /// each CF and layer per-CF tweaks (merge operators, compaction
    /// filters, custom block sizes) on top.
    ///
    /// The default column family (`"default"`) is registered
    /// automatically by [`Db::open`] and need not be included here,
    /// though including it is harmless.
    fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor>;

    /// Construct the schema struct against `db`.
    ///
    /// Implementations typically clone the supplied `Arc<Db>` into
    /// each column-family handle they construct. The default
    /// implementation in user schemas is usually a one-line
    /// `Self::new(db.clone())` that delegates to inherent methods on
    /// the schema struct.
    fn open(db: &Arc<Db>) -> Result<Self, OpenError>;
}

/// How restore-time shard writes are routed for a column family.
///
/// Restore drivers process objects in shards (a partition of the
/// `ObjectID` space, or a partition of a formal snapshot). Each
/// shard's writes finalize either into a sorted SST atomically
/// ingested at the bottommost LSM level, or into a
/// [`rocksdb::WriteBatch`] committed through the memtable + WAL
/// path. CFs choose the mode in [`Schema::cfs`] via their
/// [`CfDescriptor`].
///
/// # Trade-offs
///
/// - [`BulkIngest`](Self::BulkIngest): the default. Restore writes
///   accumulate in a per-CF sorted in-memory buffer, finalized into
///   a single SST per shard, and atomically ingested. Bypasses the
///   memtable, the WAL, and L0 entirely. *Constraint:* one
///   operation per key per shard, because an SST file rejects
///   duplicate consecutive keys. Pipelines must fold by key in
///   their accumulator before emitting writes.
///
/// - [`MergeViaWriteBatch`](Self::MergeViaWriteBatch): restore
///   writes route into a [`rocksdb::WriteBatch`], committed
///   atomically alongside the shard's partition-complete marker.
///   *No per-key constraint:* the pipeline can emit many
///   [`merge`](crate::Batch::merge) operands for the same key per
///   shard, and the registered merge operator combines them. Goes
///   through the memtable + WAL like a regular write — slower than
///   bulk ingest, but correct for CFs whose merge semantics require
///   accumulation of many operands per key without an external
///   fold.
///
/// # Atomicity
///
/// Within a single shard's commit, [`BulkIngest`](Self::BulkIngest)
/// SSTs are ingested *first*; the
/// [`MergeViaWriteBatch`](Self::MergeViaWriteBatch) operations and
/// the shard's partition-complete marker land together in a single
/// [`rocksdb::WriteBatch`] commit *second*. A crash between the two
/// steps leaves the shard not marked complete, so resume re-runs
/// the shard from scratch: SST re-ingest is idempotent (last write
/// wins for puts), and merge-mode writes from the prior run never
/// committed, so no double-merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RestoreMode {
    /// SST bulk ingestion. The default; suitable for CFs without
    /// merge operators (or with merge operators whose restore-time
    /// values are pre-folded by the pipeline). One op per key per
    /// shard.
    #[default]
    BulkIngest,

    /// Routed through a [`rocksdb::WriteBatch`]. Suitable for CFs
    /// with merge operators that require many operands per key per
    /// shard. Slower than bulk ingest but allows ordinary merge
    /// semantics.
    MergeViaWriteBatch,
}

/// Describes one column family in a [`Schema`].
///
/// Construct via [`CfDescriptor::new`]; layer optional behavior
/// (restore mode, etc.) via the builder methods.
///
/// # Examples
///
/// ```
/// use sui_consistent_store::CfDescriptor;
/// use sui_consistent_store::RestoreMode;
///
/// let opts = rocksdb::Options::default();
/// // CFs without a merge operator use bulk ingestion by default.
/// let owners = CfDescriptor::new("owners", opts.clone());
/// // CFs whose merge operator should accept many operands per key
/// // per shard during restore opt into the WriteBatch mode.
/// let balances = CfDescriptor::new("balances", opts)
///     .with_restore_mode(RestoreMode::MergeViaWriteBatch);
/// ```
pub struct CfDescriptor {
    /// Column-family name.
    pub name: &'static str,
    /// Per-CF RocksDB options applied at create time.
    pub options: rocksdb::Options,
    /// How restore-time shard writes are routed for this CF.
    pub restore_mode: RestoreMode,
}

impl std::fmt::Debug for CfDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `rocksdb::Options` does not implement Debug, so summarize.
        f.debug_struct("CfDescriptor")
            .field("name", &self.name)
            .field("restore_mode", &self.restore_mode)
            .finish_non_exhaustive()
    }
}

impl CfDescriptor {
    /// Construct a descriptor with the default
    /// [`RestoreMode::BulkIngest`].
    pub fn new(name: &'static str, options: rocksdb::Options) -> Self {
        Self {
            name,
            options,
            restore_mode: RestoreMode::default(),
        }
    }

    /// Set the restore-time write-routing mode for this CF.
    pub fn with_restore_mode(mut self, mode: RestoreMode) -> Self {
        self.restore_mode = mode;
        self
    }
}

/// Re-binds a [`Schema`] at a captured [`SnapshotHandle`].
///
/// The schema author declares the projection's body type as `At<'s>`
/// and writes a one-line constructor that re-binds each field via
/// [`DbMap::at`](crate::DbMap::at). The trait is independent of
/// [`Schema`] so authors who never need snapshot-bound reads can
/// skip the impl entirely.
///
/// # Cost
///
/// Each call to [`at`](Self::at) constructs a fresh schema struct
/// containing a [`DbMap<_, _, Snapshot<'s>>`](crate::DbMap) per
/// field. Each per-field re-bind clones the column-family name (a
/// `Box<str>` allocation). For an N-CF schema this is N allocations
/// per re-bind. Re-bind once per request handler and read many
/// times against the same projection.
pub trait SchemaAtSnapshot {
    /// The projected schema body — typically `MySchema<Snapshot<'s>>`
    /// when the schema is parameterized by a [`Reader`](crate::Reader).
    type At<'s>
    where
        Self: 's;

    /// Re-bind this schema at `snap`.
    ///
    /// The returned projection's reads see the database state
    /// captured by the snapshot, regardless of writes that occur
    /// after [`Db::take_snapshot`](crate::Db::take_snapshot) was
    /// called. The projection borrows from both `self` and `snap`;
    /// either ending its borrow ends the projection.
    fn at<'s>(&'s self, snap: &'s SnapshotHandle) -> Self::At<'s>;
}
