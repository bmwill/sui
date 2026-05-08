// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Db`] handle wrapping an opened RocksDB database.
//!
//! [`Db`] is shared across the typed column-family handles that a
//! schema constructs against it; consumers hold an [`Arc<Db>`] for
//! the database's lifetime, with the Drop on the last clone
//! triggering RocksDB's own shutdown sequence (flush plus close).
//!
//! `Db` also holds the in-memory snapshot buffer used to serve
//! consistent reads at a given checkpoint. See [`take_snapshot`],
//! [`at_snapshot`], and [`SnapshotHandle`](crate::SnapshotHandle).
//!
//! RocksDB is internally thread-safe; the only external locking the
//! crate adds is a [`parking_lot::RwLock`] over the snapshot buffer.
//!
//! [`take_snapshot`]: Db::take_snapshot
//! [`at_snapshot`]: Db::at_snapshot

use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use rocksdb::BoundColumnFamily;

use crate::batch::Batch;
use crate::error::OpenError;
use crate::schema::Schema;
use crate::snapshot::SnapshotHandle;

/// Configuration for opening a [`Db`].
///
/// The default value enables `create_if_missing` and
/// `create_missing_column_families`, which is the configuration most
/// callers want; tweak [`db_options`](Self::db_options) to override
/// individual settings.
///
/// # Examples
///
/// ```
/// use sui_consistent_store::DbOptions;
///
/// let mut opts = DbOptions::default();
/// // Refuse to create a new database if the path is empty.
/// opts.db_options.create_if_missing(false);
/// ```
pub struct DbOptions {
    /// Underlying RocksDB options applied to the database itself.
    pub db_options: rocksdb::Options,

    /// Maximum number of in-memory snapshots retained on the database
    /// at any one time. When [`Db::take_snapshot`] is called and the
    /// buffer is at capacity, the snapshot with the lowest checkpoint
    /// number is evicted. Set high enough to retain the consistency
    /// window the application requires; long-lived snapshots pressure
    /// RocksDB compaction, so this is not free.
    pub snapshot_capacity: usize,
}

/// An opened RocksDB database.
///
/// `Db` is not constructed directly; obtain one via [`Db::open`],
/// which also constructs the typed schema struct that names its
/// column families.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use sui_consistent_store::Db;
/// use sui_consistent_store::DbOptions;
/// use sui_consistent_store::Schema;
/// use sui_consistent_store::error::OpenError;
///
/// struct MySchema {
///     _db: Arc<Db>,
/// }
///
/// impl Schema for MySchema {
///     fn cfs() -> Vec<(String, rocksdb::Options)> {
///         vec![("my_cf".to_string(), rocksdb::Options::default())]
///     }
///
///     fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
///         Ok(Self { _db: db.clone() })
///     }
/// }
///
/// let dir = tempfile::tempdir().unwrap();
/// let (_db, _schema) = Db::open::<MySchema>(dir.path(), DbOptions::default()).unwrap();
/// ```
pub struct Db {
    /// Snapshots are declared *before* `inner` so that, on `Db` drop,
    /// every retained snapshot drops (and releases its borrow on
    /// `inner`) before `inner` itself is freed.
    snapshots: RwLock<BTreeMap<u64, Arc<SnapshotEntry>>>,
    snapshot_capacity: usize,
    inner: rocksdb::DB,
}

/// Storage for a single snapshot. The contained [`rocksdb::Snapshot`]
/// borrows from [`Db::inner`]; the borrow's lifetime is extended to
/// `'static` via [`std::mem::transmute`] inside
/// [`Db::take_snapshot`] so the snapshot can be stored in a long-lived
/// map. Two invariants make this sound:
///
/// 1. `Db::inner` is declared after `Db::snapshots`, so `inner` is
///    dropped only after every retained snapshot has dropped (and
///    released its borrow).
/// 2. Outstanding [`SnapshotHandle`]s co-own the same [`Arc<Db>`],
///    so `Db` cannot drop while a handle exists. Field ordering
///    inside `SnapshotHandle` ensures the `Arc<SnapshotEntry>` drops
///    before the `Arc<Db>`.
pub(crate) struct SnapshotEntry {
    snapshot: rocksdb::Snapshot<'static>,
}

impl SnapshotEntry {
    pub(crate) fn as_snapshot(&self) -> &rocksdb::Snapshot<'static> {
        &self.snapshot
    }
}

impl Default for DbOptions {
    fn default() -> Self {
        let mut db_options = rocksdb::Options::default();
        db_options.create_if_missing(true);
        db_options.create_missing_column_families(true);
        Self {
            db_options,
            snapshot_capacity: 32,
        }
    }
}

impl fmt::Debug for DbOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `rocksdb::Options` does not implement Debug, so summarize.
        f.debug_struct("DbOptions").finish_non_exhaustive()
    }
}

impl fmt::Debug for Db {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `rocksdb::DB` does not implement Debug; print only the path.
        f.debug_struct("Db")
            .field("path", &self.inner.path())
            .finish_non_exhaustive()
    }
}

impl Db {
    /// Open a database at `path` with the given schema.
    ///
    /// The default column family is registered automatically (RocksDB
    /// requires it). Column families named by [`Schema::cfs`] that do
    /// not yet exist on disk are created when
    /// [`DbOptions::db_options`] has `create_missing_column_families`
    /// enabled, which is the default.
    ///
    /// On success, returns the database handle (as an [`Arc`] so it
    /// can be shared with column-family wrappers) and the constructed
    /// schema.
    pub fn open<S: Schema>(
        path: impl AsRef<Path>,
        opts: DbOptions,
    ) -> Result<(Arc<Self>, S), OpenError> {
        let DbOptions {
            db_options,
            snapshot_capacity,
        } = opts;

        let mut cfs = S::cfs();
        // RocksDB requires the default column family to be declared
        // when opening with `open_cf_descriptors`. Register it
        // automatically so schemas don't have to.
        if !cfs.iter().any(|(name, _)| name == "default") {
            cfs.push((String::from("default"), rocksdb::Options::default()));
        }

        let descriptors = cfs
            .into_iter()
            .map(|(name, opts)| rocksdb::ColumnFamilyDescriptor::new(name, opts));
        let inner = rocksdb::DB::open_cf_descriptors(&db_options, &path, descriptors)?;

        let path_str = path.as_ref().display().to_string();
        tracing::info!(path = %path_str, "opened consistent-store database");

        let db = Arc::new(Self {
            snapshots: RwLock::new(BTreeMap::new()),
            snapshot_capacity,
            inner,
        });
        let schema = S::open(&db)?;
        Ok((db, schema))
    }

    /// Look up a column family handle by name.
    ///
    /// Returns `None` if no column family with the given name was
    /// registered when the database was opened. The returned handle
    /// borrows from `self`; callers must not retain it beyond that
    /// borrow.
    pub(crate) fn cf_handle(&self, name: &str) -> Option<Arc<BoundColumnFamily<'_>>> {
        self.inner.cf_handle(name)
    }

    /// Borrow the underlying RocksDB handle.
    ///
    /// Used by typed wrappers (`DbMap`) to call read and write methods
    /// on the database. Not part of the public API.
    pub(crate) fn rocksdb(&self) -> &rocksdb::DB {
        &self.inner
    }

    /// Construct an empty atomic write batch tied to this database.
    ///
    /// Stage operations against the returned [`Batch`] using
    /// [`Batch::put`] and [`Batch::delete`], then call
    /// [`Batch::commit`] to apply them atomically.
    pub fn batch(self: &Arc<Self>) -> Batch {
        Batch::new(self.clone())
    }

    /// Take a snapshot of the database state and store it under
    /// `checkpoint`.
    ///
    /// Snapshots are point-in-time consistent views of the data; a
    /// subsequent [`at_snapshot`](Self::at_snapshot) lookup at the
    /// same `checkpoint` returns a handle that reads from this state
    /// regardless of any writes that happen after this call returns.
    ///
    /// The snapshot is taken and inserted while holding the snapshot
    /// buffer's write lock, so concurrent `take_snapshot` calls are
    /// serialized. The snapshot's state is captured by RocksDB at
    /// the [`rocksdb::DB::snapshot`] call, before the lock is taken;
    /// callers who require a strict ordering between writes and the
    /// snapshot's state should ensure no concurrent writers race
    /// with this call.
    ///
    /// If a snapshot already exists at `checkpoint`, it is replaced.
    /// If the buffer is at
    /// [`DbOptions::snapshot_capacity`](crate::DbOptions::snapshot_capacity),
    /// the snapshot with the lowest checkpoint number is evicted.
    pub fn take_snapshot(&self, checkpoint: u64) {
        let snapshot = self.inner.snapshot();
        // SAFETY: `Snapshot::<'_, rocksdb::DB>::'_` is a borrow of
        // `self.inner`. The transmute to `'static` is sound because
        // (1) `Db::inner` is declared after `Db::snapshots`, so
        // `inner` outlives every snapshot retained in the map; and
        // (2) `SnapshotHandle`s co-own `Arc<Db>` and drop their
        // `Arc<SnapshotEntry>` before their `Arc<Db>`, so no
        // snapshot can survive a `Db` drop.
        let snapshot: rocksdb::Snapshot<'static> = unsafe { std::mem::transmute(snapshot) };
        let entry = Arc::new(SnapshotEntry { snapshot });

        let mut snaps = self.snapshots.write();
        snaps.insert(checkpoint, entry);
        while snaps.len() > self.snapshot_capacity {
            snaps.pop_first();
        }
    }

    /// Look up the snapshot stored at `checkpoint`.
    ///
    /// Returns `None` if no snapshot exists at that checkpoint.
    /// Cloning the returned [`SnapshotHandle`] is cheap; clones share
    /// the same underlying snapshot.
    pub fn at_snapshot(self: &Arc<Self>, checkpoint: u64) -> Option<SnapshotHandle> {
        let snaps = self.snapshots.read();
        let entry = snaps.get(&checkpoint)?.clone();
        Some(SnapshotHandle::new(self.clone(), entry, checkpoint))
    }

    /// Returns the inclusive range of checkpoints covered by the
    /// snapshot buffer, or `None` if the buffer is empty.
    pub fn snapshot_range(&self) -> Option<RangeInclusive<u64>> {
        let snaps = self.snapshots.read();
        let lo = *snaps.keys().next()?;
        let hi = *snaps.keys().next_back()?;
        Some(lo..=hi)
    }

    /// Drop the snapshot at `checkpoint`. Returns `true` if a
    /// snapshot was removed.
    ///
    /// Outstanding [`SnapshotHandle`]s for this checkpoint remain
    /// usable until they themselves drop; only the buffer's
    /// reference is released.
    pub fn drop_snapshot(&self, checkpoint: u64) -> bool {
        self.snapshots.write().remove(&checkpoint).is_some()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    /// Two-CF schema used by the open/close tests in this module.
    #[derive(Debug)]
    struct TestSchema {
        _db: Arc<Db>,
    }

    impl Schema for TestSchema {
        fn cfs() -> Vec<(String, rocksdb::Options)> {
            vec![
                (String::from("foo"), rocksdb::Options::default()),
                (String::from("bar"), rocksdb::Options::default()),
            ]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self { _db: db.clone() })
        }
    }

    #[test]
    fn open_creates_database_with_schema_cfs() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        assert!(db.cf_handle("foo").is_some());
        assert!(db.cf_handle("bar").is_some());
    }

    #[test]
    fn open_registers_default_cf() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        assert!(db.cf_handle("default").is_some());
    }

    #[test]
    fn cf_handle_returns_none_for_unknown_cf() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        assert!(db.cf_handle("not_in_schema").is_none());
    }

    #[test]
    fn reopen_existing_database() {
        let dir = TempDir::new().unwrap();
        {
            let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
            assert!(db.cf_handle("foo").is_some());
        }
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        assert!(db.cf_handle("foo").is_some());
        assert!(db.cf_handle("bar").is_some());
    }

    #[test]
    fn open_without_create_if_missing_errors_on_missing_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent");
        let mut opts = DbOptions::default();
        opts.db_options.create_if_missing(false);
        let result = Db::open::<TestSchema>(&path, opts);
        let err = result.expect_err("open should fail when path is missing");
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn open_propagates_rocksdb_lock_error() {
        let dir = TempDir::new().unwrap();
        let (_db1, _schema1) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let result = Db::open::<TestSchema>(dir.path(), DbOptions::default());
        let err = result.expect_err("second open of the same path should fail");
        assert!(std::error::Error::source(&err).is_some());
    }
}
