// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Reader`] trait and its implementations: [`Live`] /
//! [`LiveRef`] (live-tip) and [`Snapshot`](crate::Snapshot), plus a
//! blanket impl on `&Snapshot` for the borrowed snapshot-bound
//! case.
//!
//! Every [`DbMap<K, V, R>`](crate::DbMap) is parameterized by a
//! [`Reader`]. The default is [`Live`], so today's call sites
//! (`DbMap::new(db, "items")`, `Db::open::<MySchema>(...)`) keep
//! working unchanged. Re-projecting a schema or a single map at a
//! captured snapshot — via
//! [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at) or
//! [`DbMap::at`](crate::DbMap::at) — produces a parallel handle whose
//! reader is [`Snapshot`](crate::Snapshot).
//!
//! # Why this exists
//!
//! The crate previously routed snapshot reads through an entirely
//! separate type (the snapshot handle's read methods, plus a
//! borrowed `SnapshotView`). That kept the read API duplicated and
//! forced call sites to choose between `map.get(&k)?` (live) and
//! `snap.get(&map, &k)?` (snapshot). With the reader generic, both
//! call sites read identically: the choice of consistency context is
//! made once, at the point the schema or map is bound, and from
//! there every method call is `map.get(&k)?` against whatever reader
//! is in scope.
//!
//! # Cost model
//!
//! `Live` owns a [`Db`] handle (one `Arc` bump per [`DbMap`] field,
//! same cost as today). [`Snapshot`](crate::Snapshot) owns a `Db`
//! handle plus an `Arc<SnapshotEntry>` plus a `u64`; constructing
//! one is two atomic increments. Each call to
//! [`DbMap::at`](crate::DbMap::at) clones the snapshot (two `Arc`
//! bumps); the column-family name is a [`&'static str`](prim@str),
//! so it copies without allocating. Re-projecting an N-CF schema
//! costs 2N `Arc` bumps and N struct constructions per call to
//! [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at). For a
//! per-request handler that projects once and reads many times,
//! this is amortized; for a hot path that projects on every read,
//! project once outside the loop.
//!
//! [`LiveRef`] and the `&Snapshot` blanket impl are the
//! zero-`Arc`-bump variants: they hold a borrow rather than an owned
//! [`Db`] / [`Snapshot`](crate::Snapshot). Construct a
//! [`LiveRef`]-bound map with [`DbMap::new_ref`](crate::DbMap::new_ref);
//! re-bind an existing map at a borrowed snapshot with
//! [`DbMap::at_ref`](crate::DbMap::at_ref). Use these when the
//! returned [`DbMap`](crate::DbMap) is scoped to a single function
//! body and can be tied to a [`Db`] or
//! [`Snapshot`](crate::Snapshot) the caller already holds.

use rocksdb::ReadOptions;

use crate::db::Db;

/// Abstracts the read context a [`DbMap`](crate::DbMap) is bound to.
///
/// Every implementation supplies (1) the [`Db`] handle needed to
/// look up the column-family handle and (2) a fresh [`ReadOptions`]
/// tuned for the reader's consistency context. [`Live`] and
/// [`LiveRef`] return [`ReadOptions::default()`];
/// [`Snapshot`](crate::Snapshot) (and its `&Snapshot` blanket impl)
/// return one with [`set_snapshot`](ReadOptions::set_snapshot)
/// pointed at the captured snapshot.
///
/// # Sealed
///
/// The crate ships three implementations — [`Live`], [`LiveRef`],
/// and [`Snapshot`](crate::Snapshot) — plus a blanket impl on
/// `&Snapshot` that delegates to [`Snapshot`](crate::Snapshot)'s
/// own impl. The trait is sealed via a pub(crate) supertrait so
/// downstream code cannot add another — a custom reader could
/// return [`ReadOptions`] referencing a snapshot pointer not
/// co-owned through the [`Db`] handle story, leading to UB inside
/// RocksDB.
pub trait Reader: sealed::Sealed {
    /// The shared database handle the column family lives on.
    fn db(&self) -> &Db;

    /// Construct a fresh [`ReadOptions`] configured for this reader.
    ///
    /// Implementations are expected to be cheap; the returned options
    /// are consumed by a single read and dropped. Callers that issue
    /// many reads in a tight loop pay one fresh allocation per call,
    /// which matches RocksDB's expected pattern.
    fn read_options(&self) -> ReadOptions;
}

pub(crate) mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Live {}
    impl<'a> Sealed for super::LiveRef<'a> {}
}

/// Reader bound to the database's live tip.
///
/// Reads see the most-recently-committed state at the moment of the
/// call. `Live` is the default reader for any
/// [`DbMap`](crate::DbMap) and any schema opened through
/// [`Db::open`](crate::Db::open); construction is automatic when the
/// reader type parameter is left at its default.
#[derive(Debug)]
pub struct Live {
    db: Db,
}

impl Live {
    pub(crate) fn new(db: Db) -> Self {
        Self { db }
    }
}

impl Reader for Live {
    fn db(&self) -> &Db {
        &self.db
    }

    fn read_options(&self) -> ReadOptions {
        ReadOptions::default()
    }
}

/// Borrowed counterpart to [`Live`]. Reads from the database's live
/// tip without taking ownership of (or cloning) the underlying
/// [`Db`] handle.
///
/// Construct a [`DbMap`](crate::DbMap) bound to `LiveRef` via
/// [`DbMap::new_ref`](crate::DbMap::new_ref). Use this when the
/// resulting handle is scoped to a single function body and can
/// borrow a [`Db`] the caller already holds, instead of paying an
/// extra `Arc` bump per [`DbMap`](crate::DbMap) field.
///
/// `LiveRef` is `Copy + Clone` (it is just a reference); cloning
/// does no work.
#[derive(Debug, Clone, Copy)]
pub struct LiveRef<'a> {
    db: &'a Db,
}

impl<'a> LiveRef<'a> {
    /// Construct a `LiveRef` bound to `db`.
    pub fn new(db: &'a Db) -> Self {
        Self { db }
    }
}

impl<'a> Reader for LiveRef<'a> {
    fn db(&self) -> &Db {
        self.db
    }

    fn read_options(&self) -> ReadOptions {
        ReadOptions::default()
    }
}
