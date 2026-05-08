// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Reader`] trait and its two implementations,
//! [`Live`] and [`Snapshot`].
//!
//! Every [`DbMap<K, V, R>`](crate::DbMap) is parameterized by a
//! [`Reader`]. The default is [`Live`], so today's call sites
//! (`DbMap::new(db, "items")`, `Db::open::<MySchema>(...)`) keep
//! working unchanged. Re-projecting a schema or a single map at a
//! captured snapshot — via
//! [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at) or
//! [`DbMap::at`](crate::DbMap::at) — produces a parallel handle whose
//! reader is [`Snapshot<'s>`].
//!
//! # Why this exists
//!
//! The crate previously routed snapshot reads through an entirely
//! separate type ([`SnapshotHandle`]'s read methods, plus a borrowed
//! `SnapshotView`). That kept the read API duplicated and forced
//! call sites to choose between `map.get(&k)?` (live) and
//! `snap.get(&map, &k)?` (snapshot). With the reader generic, both
//! call sites read identically: the choice of consistency context is
//! made once, at the point the schema or map is bound, and from
//! there every method call is `map.get(&k)?` against whatever reader
//! is in scope.
//!
//! # Cost model
//!
//! `Live` owns an `Arc<Db>` (one Arc clone per [`DbMap`] field, same
//! cost as today). `Snapshot<'s>` borrows the
//! [`SnapshotHandle`] — zero allocation. Each call to
//! [`DbMap::at`](crate::DbMap::at) clones the column-family name
//! ([`Box<str>`]) once, so re-projecting an N-CF schema costs N name
//! clones and N struct constructions per call to
//! [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at). For a
//! per-request handler that projects once and reads many times, this
//! is amortized; for a hot path that projects on every read, project
//! once outside the loop.

use std::sync::Arc;

use rocksdb::ReadOptions;

use crate::db::Db;
use crate::snapshot::SnapshotHandle;

/// Abstracts the read context a [`DbMap`](crate::DbMap) is bound to.
///
/// Both implementations supply (1) the [`Arc<Db>`] needed to look up
/// the column-family handle and (2) a fresh [`ReadOptions`] tuned for
/// the reader's consistency context. [`Live`] returns
/// [`ReadOptions::default()`]; [`Snapshot`] returns one with
/// [`set_snapshot`](ReadOptions::set_snapshot) pointed at the
/// captured snapshot.
///
/// # Sealed-by-convention
///
/// The crate ships exactly two implementations, [`Live`] and
/// [`Snapshot`]. The trait is technically open, but downstream
/// implementations are not supported: every consumer's read path
/// makes assumptions about the two known reader shapes, and a
/// custom reader could break the snapshot-borrow lifetime story.
pub trait Reader {
    /// The shared database handle the column family lives on.
    fn db(&self) -> &Arc<Db>;

    /// Construct a fresh [`ReadOptions`] configured for this reader.
    ///
    /// Implementations are expected to be cheap; the returned options
    /// are consumed by a single read and dropped. Callers that issue
    /// many reads in a tight loop pay one fresh allocation per call,
    /// which matches RocksDB's expected pattern.
    fn read_options(&self) -> ReadOptions;
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
    db: Arc<Db>,
}

impl Live {
    pub(crate) fn new(db: Arc<Db>) -> Self {
        Self { db }
    }
}

impl Reader for Live {
    fn db(&self) -> &Arc<Db> {
        &self.db
    }

    fn read_options(&self) -> ReadOptions {
        ReadOptions::default()
    }
}

/// Reader bound to a captured snapshot.
///
/// Borrows the [`SnapshotHandle`] so the projection's lifetime is
/// tied to the caller's handle. Constructed via
/// [`DbMap::at`](crate::DbMap::at) or
/// [`SchemaAtSnapshot::at`](crate::SchemaAtSnapshot::at) rather than
/// directly.
///
/// All reads through a [`DbMap`](crate::DbMap) parameterized by
/// `Snapshot<'s>` see the database state captured by
/// [`Db::take_snapshot`](crate::Db::take_snapshot), regardless of
/// writes that occur after the snapshot was taken.
#[derive(Debug, Clone, Copy)]
pub struct Snapshot<'s> {
    handle: &'s SnapshotHandle,
}

impl<'s> Snapshot<'s> {
    pub(crate) fn new(handle: &'s SnapshotHandle) -> Self {
        Self { handle }
    }

    /// The underlying [`SnapshotHandle`] this reader borrows.
    pub fn handle(&self) -> &'s SnapshotHandle {
        self.handle
    }
}

impl<'s> Reader for Snapshot<'s> {
    fn db(&self) -> &Arc<Db> {
        self.handle.db()
    }

    fn read_options(&self) -> ReadOptions {
        let mut opts = ReadOptions::default();
        opts.set_snapshot(self.handle.entry().as_snapshot());
        opts
    }
}
