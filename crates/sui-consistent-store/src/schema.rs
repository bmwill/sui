// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Schema`] trait used to register column families with the
//! database and to construct typed handles into them.
//!
//! Schemas are hand-written Rust structs whose fields are typed
//! handles into individual column families (typically `DbMap<K, V>`,
//! arriving in a later commit). The trait pairs the static set of
//! column families a schema requires (returned from [`Schema::cfs`])
//! with the constructor that builds the schema struct from an opened
//! database (the [`Schema::open`] method).
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//!
//! use sui_consistent_store::Db;
//! use sui_consistent_store::DbOptions;
//! use sui_consistent_store::Schema;
//! use sui_consistent_store::error::OpenError;
//!
//! struct MySchema {
//!     _db: Arc<Db>,
//! }
//!
//! impl Schema for MySchema {
//!     fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
//!         vec![("my_cf", base_options.clone())]
//!     }
//!
//!     fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
//!         Ok(Self { _db: db.clone() })
//!     }
//! }
//!
//! let dir = tempfile::tempdir().unwrap();
//! let (_db, _schema) = Db::open::<MySchema>(dir.path(), DbOptions::default()).unwrap();
//! ```

use std::sync::Arc;

use crate::db::Db;
use crate::error::OpenError;

/// Declares the column families a database needs and constructs the
/// typed handle struct against an opened database.
///
/// Implementations are typically hand-written structs whose fields are
/// typed column-family handles. The trait has two responsibilities:
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
