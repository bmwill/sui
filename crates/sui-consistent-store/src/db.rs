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
///     fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
///         vec![("my_cf", base_options.clone())]
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

        let mut cfs = S::cfs(&db_options);
        // RocksDB requires the default column family to be declared
        // when opening with `open_cf_descriptors`. Register it
        // automatically so schemas don't have to.
        if !cfs.iter().any(|(name, _)| *name == "default") {
            cfs.push(("default", db_options.clone()));
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

    /// Look up the snapshot with the highest checkpoint number in
    /// the buffer.
    ///
    /// Returns `None` if no snapshots have been taken (or all have
    /// been evicted or dropped). Equivalent to
    /// [`at_snapshot`](Self::at_snapshot) called with the upper
    /// bound of [`snapshot_range`](Self::snapshot_range).
    pub fn latest_snapshot(self: &Arc<Self>) -> Option<SnapshotHandle> {
        let snaps = self.snapshots.read();
        let (checkpoint, entry) = snaps.iter().next_back()?;
        Some(SnapshotHandle::new(
            self.clone(),
            entry.clone(),
            *checkpoint,
        ))
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

    /// Read RocksDB's per-column-family runtime properties for
    /// `cf_name`.
    ///
    /// Returns a [`RocksMetrics`] struct populated from RocksDB's
    /// `property_int_value_cf` API. Fields default to `-1` when the
    /// column family is not registered or RocksDB cannot report a
    /// value (some properties depend on subsystems that are not
    /// always active, for example blob-file totals on a CF without
    /// blob storage configured).
    pub fn cf_metrics(&self, cf_name: &str) -> RocksMetrics {
        let Some(cf) = self.cf_handle(cf_name) else {
            return RocksMetrics::default();
        };
        let read = |property: &str| -> i64 {
            self.inner
                .property_int_value_cf(&cf, property)
                .ok()
                .flatten()
                .map(|v| v as i64)
                .unwrap_or(METRICS_ERROR)
        };
        RocksMetrics {
            block_cache_capacity: read("rocksdb.block-cache-capacity"),
            block_cache_usage: read("rocksdb.block-cache-usage"),
            block_cache_pinned_usage: read("rocksdb.block-cache-pinned-usage"),
            current_size_active_mem_tables: read("rocksdb.cur-size-active-mem-table"),
            size_all_mem_tables: read("rocksdb.size-all-mem-tables"),
            num_immutable_mem_tables: read("rocksdb.num-immutable-mem-table"),
            mem_table_flush_pending: read("rocksdb.mem-table-flush-pending"),
            estimate_table_readers_mem: read("rocksdb.estimate-table-readers-mem"),
            num_level0_files: read("rocksdb.num-files-at-level0"),
            base_level: read("rocksdb.base-level"),
            compaction_pending: read("rocksdb.compaction-pending"),
            num_running_compactions: read("rocksdb.num-running-compactions"),
            num_running_flushes: read("rocksdb.num-running-flushes"),
            estimate_pending_compaction_bytes: read("rocksdb.estimate-pending-compaction-bytes"),
            num_snapshots: read("rocksdb.num-snapshots"),
            oldest_snapshot_time: read("rocksdb.oldest-snapshot-time"),
            estimate_oldest_key_time: read("rocksdb.estimate-oldest-key-time"),
            estimated_num_keys: read("rocksdb.estimate-num-keys"),
            background_errors: read("rocksdb.background-errors"),
            total_sst_files_size: read("rocksdb.total-sst-files-size"),
            total_blob_files_size: read("rocksdb.total-blob-file-size"),
            actual_delayed_write_rate: read("rocksdb.actual-delayed-write-rate"),
            is_write_stopped: read("rocksdb.is-write-stopped"),
        }
    }
}

/// Sentinel value used in [`RocksMetrics`] when a property is
/// unavailable: the column family is not registered, or RocksDB
/// returned an error or an empty result for the property.
const METRICS_ERROR: i64 = -1;

/// Per-column-family runtime metrics read from RocksDB on demand.
///
/// Populated by [`Db::cf_metrics`]. Each field corresponds to a
/// `rocksdb.*` integer property; fields hold [`METRICS_ERROR`]
/// (`-1`) when the property cannot be read. The struct is plain
/// data; consumers are expected to convert it into whatever their
/// monitoring stack wants (Prometheus gauges, structured logs,
/// etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocksMetrics {
    /// `rocksdb.block-cache-capacity` — configured size (bytes).
    pub block_cache_capacity: i64,
    /// `rocksdb.block-cache-usage` — current size (bytes).
    pub block_cache_usage: i64,
    /// `rocksdb.block-cache-pinned-usage` — bytes currently pinned
    /// (held alive by outstanding readers).
    pub block_cache_pinned_usage: i64,
    /// `rocksdb.cur-size-active-mem-table` — active memtable bytes.
    pub current_size_active_mem_tables: i64,
    /// `rocksdb.size-all-mem-tables` — active plus immutable
    /// memtable bytes.
    pub size_all_mem_tables: i64,
    /// `rocksdb.num-immutable-mem-table` — count of immutable
    /// memtables waiting to be flushed.
    pub num_immutable_mem_tables: i64,
    /// `rocksdb.mem-table-flush-pending` — `1` if a flush is
    /// pending, else `0`.
    pub mem_table_flush_pending: i64,
    /// `rocksdb.estimate-table-readers-mem` — approximate memory
    /// used by table readers (excluding the block cache).
    pub estimate_table_readers_mem: i64,
    /// `rocksdb.num-files-at-level0` — number of level-0 SST files.
    pub num_level0_files: i64,
    /// `rocksdb.base-level` — RocksDB's current base level.
    pub base_level: i64,
    /// `rocksdb.compaction-pending` — `1` if compaction is pending.
    pub compaction_pending: i64,
    /// `rocksdb.num-running-compactions` — currently running
    /// compactions.
    pub num_running_compactions: i64,
    /// `rocksdb.num-running-flushes` — currently running flushes.
    pub num_running_flushes: i64,
    /// `rocksdb.estimate-pending-compaction-bytes` — bytes the
    /// compaction backlog will rewrite.
    pub estimate_pending_compaction_bytes: i64,
    /// `rocksdb.num-snapshots` — count of unreleased
    /// `rocksdb::Snapshot` handles.
    pub num_snapshots: i64,
    /// `rocksdb.oldest-snapshot-time` — unix-time of the oldest
    /// live snapshot.
    pub oldest_snapshot_time: i64,
    /// `rocksdb.estimate-oldest-key-time` — unix-time estimate of
    /// the oldest live key.
    pub estimate_oldest_key_time: i64,
    /// `rocksdb.estimate-num-keys` — approximate live key count.
    pub estimated_num_keys: i64,
    /// `rocksdb.background-errors` — accumulated background errors.
    pub background_errors: i64,
    /// `rocksdb.total-sst-files-size` — bytes occupied by SST files.
    pub total_sst_files_size: i64,
    /// `rocksdb.total-blob-file-size` — bytes occupied by blob
    /// files.
    pub total_blob_files_size: i64,
    /// `rocksdb.actual-delayed-write-rate` — current write-rate
    /// throttling level (bytes/sec, `0` when not throttled).
    pub actual_delayed_write_rate: i64,
    /// `rocksdb.is-write-stopped` — `1` if writes are stopped.
    pub is_write_stopped: i64,
}

impl Default for RocksMetrics {
    fn default() -> Self {
        Self {
            block_cache_capacity: METRICS_ERROR,
            block_cache_usage: METRICS_ERROR,
            block_cache_pinned_usage: METRICS_ERROR,
            current_size_active_mem_tables: METRICS_ERROR,
            size_all_mem_tables: METRICS_ERROR,
            num_immutable_mem_tables: METRICS_ERROR,
            mem_table_flush_pending: METRICS_ERROR,
            estimate_table_readers_mem: METRICS_ERROR,
            num_level0_files: METRICS_ERROR,
            base_level: METRICS_ERROR,
            compaction_pending: METRICS_ERROR,
            num_running_compactions: METRICS_ERROR,
            num_running_flushes: METRICS_ERROR,
            estimate_pending_compaction_bytes: METRICS_ERROR,
            num_snapshots: METRICS_ERROR,
            oldest_snapshot_time: METRICS_ERROR,
            estimate_oldest_key_time: METRICS_ERROR,
            estimated_num_keys: METRICS_ERROR,
            background_errors: METRICS_ERROR,
            total_sst_files_size: METRICS_ERROR,
            total_blob_files_size: METRICS_ERROR,
            actual_delayed_write_rate: METRICS_ERROR,
            is_write_stopped: METRICS_ERROR,
        }
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
        fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)> {
            vec![("foo", base_options.clone()), ("bar", base_options.clone())]
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
    fn cf_metrics_returns_default_for_unknown_cf() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let metrics = db.cf_metrics("not_in_schema");
        // All sentinel values for an unknown CF.
        assert_eq!(metrics, RocksMetrics::default());
    }

    #[test]
    fn cf_metrics_reports_real_values_for_known_cf() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let metrics = db.cf_metrics("foo");
        // We don't assert specific numbers (they depend on RocksDB
        // internals), but a known CF should yield non-sentinel
        // values for at least the always-available properties:
        // block cache state and memtable sizes.
        assert!(metrics.block_cache_capacity >= 0);
        assert!(metrics.size_all_mem_tables >= 0);
        assert!(metrics.num_immutable_mem_tables >= 0);
        assert!(metrics.is_write_stopped >= 0);
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
