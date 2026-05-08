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
//!     fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
//!         vec![("my_cf", base_options.clone())]
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
    /// Each entry is a column-family name (a `&'static str` so the
    /// schema's CF set is fixed at compile time) and its
    /// [`rocksdb::Options`] applied at create time. `base_options` is
    /// supplied by [`Db::open`] and is the database-level options
    /// configured on [`DbOptions::db_options`](crate::DbOptions::db_options);
    /// implementations typically clone it as the starting point for
    /// each CF and layer per-CF tweaks (merge operators, compaction
    /// filters, custom block sizes) on top.
    ///
    /// The default column family (`"default"`) is registered
    /// automatically by [`Db::open`] and need not be included here,
    /// though including it is harmless.
    fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)>;

    /// Construct the schema struct against `db`.
    ///
    /// Implementations typically clone the supplied `Arc<Db>` into
    /// each column-family handle they construct. The default
    /// implementation in user schemas is usually a one-line
    /// `Self::new(db.clone())` that delegates to inherent methods on
    /// the schema struct.
    fn open(db: &Arc<Db>) -> Result<Self, OpenError>;
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
