// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Db`] handle wrapping an opened RocksDB database.
//!
//! [`Db`] is a cheap-to-clone handle: every clone is one `Arc` bump
//! against the shared [`DbInner`] holding the actual RocksDB
//! database. Clones are shared freely across typed column-family
//! handles and across threads; the database itself stays alive
//! until the last clone drops, at which point RocksDB's own
//! shutdown sequence (flush plus close) runs.
//!
//! `Db` also holds the in-memory snapshot buffer used to serve
//! consistent reads at a given checkpoint. See [`take_snapshot`],
//! [`at_snapshot`], and [`Snapshot`](crate::Snapshot).
//!
//! RocksDB is internally thread-safe; the only external locking the
//! crate adds is a [`parking_lot::RwLock`] over the snapshot buffer.
//!
//! [`take_snapshot`]: Db::take_snapshot
//! [`at_snapshot`]: Db::at_snapshot

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt;
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use rocksdb::BoundColumnFamily;

use crate::batch::Batch;
use crate::error::Error;
use crate::error::OpenError;
use crate::framework::CHAIN_ID_CF;
use crate::framework::FrameworkSchema;
use crate::framework::RESTORE_CF;
use crate::framework::WATERMARK_CF;
use crate::schema::CfDescriptor;
use crate::schema::RestoreMode;
use crate::schema::Schema;
use crate::snapshot::Snapshot;

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
    ///
    /// Set to `0` to disable snapshotting entirely: [`Db::take_snapshot`]
    /// becomes a no-op and no snapshot-related work is performed.
    pub snapshot_capacity: usize,
}

/// An opened RocksDB database.
///
/// `Db` is not constructed directly; obtain one via [`Db::open`],
/// which also constructs the typed schema struct that names its
/// column families. `Db` is `Clone` and cheap to clone — every
/// clone shares the same underlying database via an internal
/// [`Arc`], so handles can be freely passed by value to typed
/// column-family wrappers and across threads.
///
/// # Examples
///
/// ```
/// use sui_consistent_store::CfDescriptor;
/// use sui_consistent_store::Db;
/// use sui_consistent_store::DbOptions;
/// use sui_consistent_store::Schema;
/// use sui_consistent_store::error::OpenError;
///
/// struct MySchema {
///     _db: Db,
/// }
///
/// impl Schema for MySchema {
///     fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
///         vec![CfDescriptor::new("my_cf", base_options.clone())]
///     }
///
///     fn open(db: &Db) -> Result<Self, OpenError> {
///         Ok(Self { _db: db.clone() })
///     }
/// }
///
/// let dir = tempfile::tempdir().unwrap();
/// let (_db, _schema) = Db::open::<MySchema>(dir.path(), DbOptions::default()).unwrap();
/// ```
#[derive(Clone)]
pub struct Db {
    inner: Arc<DbInner>,
}

/// The shared storage backing a [`Db`].
///
/// Held inside an [`Arc`] inside [`Db`] so every clone of the
/// public handle co-owns the same underlying database. The last
/// clone's drop triggers RocksDB's own shutdown sequence.
///
/// Field declaration order is load-bearing: `snapshots` is
/// declared *before* `db` so, on drop, every retained snapshot
/// drops (and releases its borrow on `db`) before `db` itself is
/// freed.
struct DbInner {
    snapshots: RwLock<BTreeMap<u64, Arc<SnapshotEntry>>>,
    snapshot_capacity: usize,
    /// Per-CF restore mode, captured at open time from
    /// [`Schema::cfs`]. Used by shard-backed [`Batch`]es to dispatch
    /// per-CF writes between SST bulk ingestion and the
    /// [`rocksdb::WriteBatch`] path. Lookup-only after open, so a
    /// plain [`HashMap`] (no lock) is sufficient.
    restore_modes: HashMap<String, RestoreMode>,
    db: rocksdb::DB,
}

/// Storage for a single snapshot. The contained [`rocksdb::Snapshot`]
/// borrows from [`DbInner::db`]; the borrow's lifetime is extended
/// to `'static` via [`std::mem::transmute`] inside
/// [`Db::take_snapshot`] so the snapshot can be stored in a
/// long-lived map. Two invariants make this sound:
///
/// 1. `DbInner::db` is declared after `DbInner::snapshots`, so `db`
///    is dropped only after every retained snapshot has dropped
///    (and released its borrow).
/// 2. Outstanding [`Snapshot`](crate::Snapshot) values co-own the
///    same [`Db`] handle (and therefore the same [`Arc<DbInner>`]),
///    so `DbInner` cannot drop while a `Snapshot` exists. Field
///    ordering inside `Snapshot` ensures the `Arc<SnapshotEntry>`
///    drops before the `Db`.
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
            .field("path", &self.inner.db.path())
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
    /// On success, returns the database handle (cheap to clone via
    /// the inner [`Arc`], so it can be shared with column-family
    /// wrappers) and the constructed schema.
    pub fn open<S: Schema>(
        path: impl AsRef<Path>,
        opts: DbOptions,
    ) -> Result<(Self, S), OpenError> {
        let DbOptions {
            db_options,
            snapshot_capacity,
        } = opts;

        let mut cfs = S::cfs(&db_options);
        // RocksDB requires the default column family to be declared
        // when opening with `open_cf_descriptors`. Register it
        // automatically so schemas don't have to.
        if !cfs.iter().any(|cf| cf.name == "default") {
            cfs.push(CfDescriptor::new("default", db_options.clone()));
        }
        // The framework's bookkeeping CFs (restore, watermark,
        // chain-id) are auto-registered alongside the schema's CFs
        // so [`FrameworkSchema`] is always available via
        // [`Db::framework`] without the user having to declare it.
        //
        // `RESTORE_CF` opts into the WriteBatch restore mode so
        // partition-complete markers commit atomically alongside
        // merge-mode shard writes (see [`RestoreMode`]). The other
        // two framework CFs use the default mode.
        if !cfs.iter().any(|cf| cf.name == RESTORE_CF) {
            cfs.push(
                CfDescriptor::new(RESTORE_CF, db_options.clone())
                    .with_restore_mode(RestoreMode::MergeViaWriteBatch),
            );
        }
        if !cfs.iter().any(|cf| cf.name == WATERMARK_CF) {
            cfs.push(CfDescriptor::new(WATERMARK_CF, db_options.clone()));
        }
        if !cfs.iter().any(|cf| cf.name == CHAIN_ID_CF) {
            cfs.push(CfDescriptor::new(CHAIN_ID_CF, db_options.clone()));
        }

        let mut restore_modes: HashMap<String, RestoreMode> = HashMap::with_capacity(cfs.len());
        for cf in &cfs {
            restore_modes.insert(cf.name.to_string(), cf.restore_mode);
        }

        let descriptors = cfs
            .into_iter()
            .map(|cf| rocksdb::ColumnFamilyDescriptor::new(cf.name, cf.options));
        let db = rocksdb::DB::open_cf_descriptors(&db_options, &path, descriptors)?;

        let path_str = path.as_ref().display().to_string();
        tracing::info!(path = %path_str, "opened consistent-store database");

        let db = Self {
            inner: Arc::new(DbInner {
                snapshots: RwLock::new(BTreeMap::new()),
                snapshot_capacity,
                restore_modes,
                db,
            }),
        };
        let schema = S::open(&db)?;
        Ok((db, schema))
    }

    /// Returns `true` if `self` and `other` are handles to the same
    /// underlying database (i.e. clones of each other), `false`
    /// otherwise. The comparison is a single pointer equality on
    /// the shared inner [`Arc`].
    pub fn ptr_eq(&self, other: &Db) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Borrowed handle to the auto-registered [`FrameworkSchema`].
    ///
    /// Returns a `FrameworkSchema<&Db>` borrowing `self`. Zero
    /// `Arc` bumps; the returned schema is scoped to the borrow.
    /// Use for ad-hoc reads (or read-mostly access). For an owned
    /// handle to hold inside a longer-lived struct, construct one
    /// with [`FrameworkSchema::new(db.clone())`](FrameworkSchema::new).
    pub fn framework(&self) -> FrameworkSchema<&Db> {
        FrameworkSchema::new(self)
    }

    /// The [`RestoreMode`] declared for `cf_name`, or `None` if no
    /// such column family is registered on this database.
    ///
    /// Used by shard-backed [`Batch`]es to route per-CF writes
    /// between SST bulk ingestion and the [`rocksdb::WriteBatch`]
    /// path. Schema authors do not call this directly — the routing
    /// is internal to [`Batch`] — but the accessor is public so
    /// drivers can introspect the mode when assembling shard
    /// commits.
    pub fn restore_mode(&self, cf_name: &str) -> Option<RestoreMode> {
        self.inner.restore_modes.get(cf_name).copied()
    }

    /// Look up a column family handle by name.
    ///
    /// Returns `None` if no column family with the given name was
    /// registered when the database was opened. The returned handle
    /// borrows from `self`; callers must not retain it beyond that
    /// borrow.
    pub(crate) fn cf_handle(&self, name: &str) -> Option<Arc<BoundColumnFamily<'_>>> {
        self.inner.db.cf_handle(name)
    }

    /// Borrow the underlying RocksDB handle.
    ///
    /// Used by typed wrappers (`DbMap`) to call read and write methods
    /// on the database. Not part of the public API.
    pub(crate) fn rocksdb(&self) -> &rocksdb::DB {
        &self.inner.db
    }

    /// Construct an empty atomic write batch tied to this database.
    ///
    /// Stage operations against the returned [`Batch`] using
    /// [`Batch::put`] and [`Batch::delete`], then call
    /// [`Batch::commit`] to apply them atomically.
    pub fn batch(&self) -> Batch {
        Batch::new(self.clone())
    }

    /// Construct an empty shard-backed [`Batch`] tied to this
    /// database, suitable for restore-time bulk loads.
    ///
    /// A shard-backed batch records operations into per-CF sorted
    /// in-memory buffers rather than a [`rocksdb::WriteBatch`].
    /// Once populated, drain it into one SST per CF via
    /// [`Batch::finalize_into_ssts`] and atomically ingest the
    /// resulting files via
    /// [`Db::ingest_files_cf`](Self::ingest_files_cf).
    ///
    /// The shard backing enforces at most one operation per key
    /// per CF — the invariant SST files require. Pipelines should
    /// fold by key in their accumulator before calling
    /// [`commit`](crate::Pipeline::commit) against a shard-backed
    /// batch.
    pub fn shard_batch(&self) -> Batch {
        Batch::new_shard(self.clone())
    }

    /// Take a snapshot of the database state and store it under
    /// `checkpoint`.
    ///
    /// Snapshots are point-in-time consistent views of the data; a
    /// subsequent [`at_snapshot`](Self::at_snapshot) lookup at the
    /// same `checkpoint` returns a handle that reads from this state
    /// regardless of any writes that happen after this call returns.
    ///
    /// Both the snapshot capture and the buffer insertion happen
    /// while holding the snapshot buffer's write lock, so concurrent
    /// `take_snapshot` calls are fully serialized: the snapshot's
    /// observed state and its order in the buffer are committed
    /// together. Callers who require a strict ordering between
    /// writes and the snapshot's state still need to fence external
    /// writers themselves; this method only ensures internal
    /// consistency between competing `take_snapshot` callers.
    ///
    /// If a snapshot already exists at `checkpoint`, it is replaced.
    /// If the buffer is at
    /// [`DbOptions::snapshot_capacity`](crate::DbOptions::snapshot_capacity),
    /// the snapshot with the lowest checkpoint number is evicted. If
    /// capacity is `0`, snapshotting is disabled and this call is a
    /// no-op.
    pub fn take_snapshot(&self, checkpoint: u64) {
        if self.inner.snapshot_capacity == 0 {
            return;
        }
        // The rocksdb snapshot is captured *inside* the lock so that
        // two concurrent `take_snapshot(N)` calls cannot land in
        // checkpoint-N → older-state order: with the capture outside
        // the lock, a thread that captures earlier may insert later
        // and overwrite a fresher snapshot.
        let mut snaps = self.inner.snapshots.write();
        let snapshot = self.inner.db.snapshot();
        // SAFETY: `rocksdb::Snapshot<'_>` is a borrow of
        // `self.inner.db`. The transmute to `'static` is sound
        // because (1) `DbInner::db` is declared after
        // `DbInner::snapshots`, so `db` outlives every snapshot
        // retained in the map; and (2) `Snapshot`s co-own the same
        // [`Db`] handle (and therefore the same `Arc<DbInner>`) and
        // drop their `Arc<SnapshotEntry>` before their `Db`, so no
        // snapshot can survive a `DbInner` drop.
        let snapshot: rocksdb::Snapshot<'static> = unsafe { std::mem::transmute(snapshot) };
        let entry = Arc::new(SnapshotEntry { snapshot });

        snaps.insert(checkpoint, entry);
        while snaps.len() > self.inner.snapshot_capacity {
            snaps.pop_first();
        }
    }

    /// Look up the snapshot stored at `checkpoint`.
    ///
    /// Returns `None` if no snapshot exists at that checkpoint.
    /// Cloning the returned [`Snapshot`](crate::Snapshot) is cheap;
    /// clones share the same underlying snapshot.
    pub fn at_snapshot(&self, checkpoint: u64) -> Option<Snapshot> {
        let snaps = self.inner.snapshots.read();
        let entry = snaps.get(&checkpoint)?.clone();
        Some(Snapshot::new(self.clone(), entry, checkpoint))
    }

    /// Look up the snapshot with the highest checkpoint number in
    /// the buffer.
    ///
    /// Returns `None` if no snapshots have been taken (or all have
    /// been evicted or dropped). Equivalent to
    /// [`at_snapshot`](Self::at_snapshot) called with the upper
    /// bound of [`snapshot_range`](Self::snapshot_range).
    pub fn latest_snapshot(&self) -> Option<Snapshot> {
        let snaps = self.inner.snapshots.read();
        let (checkpoint, entry) = snaps.iter().next_back()?;
        Some(Snapshot::new(self.clone(), entry.clone(), *checkpoint))
    }

    /// Returns the inclusive range of checkpoints covered by the
    /// snapshot buffer, or `None` if the buffer is empty.
    pub fn snapshot_range(&self) -> Option<RangeInclusive<u64>> {
        let snaps = self.inner.snapshots.read();
        let lo = *snaps.keys().next()?;
        let hi = *snaps.keys().next_back()?;
        Some(lo..=hi)
    }

    /// Apply caller-supplied runtime-mutable options to `cf_name`.
    ///
    /// Thin typed wrapper around
    /// [`rocksdb::DB::set_options_cf`] that surfaces an unknown
    /// column-family name as
    /// [`Error::MissingColumnFamily`](crate::error::Error::MissingColumnFamily)
    /// rather than as a generic RocksDB error.
    ///
    /// `opts` is a slice of `(name, value)` pairs. Names must come
    /// from RocksDB's runtime-mutable options list (see
    /// [`advanced_options.h`]); unknown or non-mutable names fail
    /// with [`Error::Rocksdb`](crate::error::Error::Rocksdb).
    ///
    /// An empty `opts` slice is a no-op (returns `Ok(())` without
    /// touching RocksDB); the unknown-CF check is still applied.
    /// RocksDB itself rejects an empty option set with an
    /// `Invalid argument: empty input` error, but at the typed
    /// surface "apply nothing" is a sensible default for callers
    /// that build option lists dynamically.
    ///
    /// [`advanced_options.h`]: https://github.com/facebook/rocksdb/blob/main/include/rocksdb/advanced_options.h
    pub fn set_options_cf(&self, cf_name: &str, opts: &[(&str, &str)]) -> Result<(), Error> {
        let cf = self
            .cf_handle(cf_name)
            .ok_or_else(|| Error::MissingColumnFamily(cf_name.to_string()))?;
        if opts.is_empty() {
            return Ok(());
        }
        self.inner.db.set_options_cf(&cf, opts)?;
        Ok(())
    }

    /// Apply restore-friendly compaction settings to `cf_name`.
    ///
    /// Disables auto-compaction and raises the three L0 triggers to
    /// ceiling values so a bulk load that produces many L0 files
    /// (or relies on
    /// [`ingest_files_cf`](Self::ingest_files_cf) into a fresh CF)
    /// does not stall on slowdown / stop thresholds.
    ///
    /// Pairs with [`set_tip_options_cf`](Self::set_tip_options_cf):
    /// the application transitions restore → tip with the matching
    /// toggle and no reopen. Both toggles touch only runtime-mutable
    /// option keys, so this method can be called at any time on an
    /// open database.
    ///
    /// The keys touched are:
    /// - `disable_auto_compactions = true`
    /// - `level0_file_num_compaction_trigger = i32::MAX`
    /// - `level0_slowdown_writes_trigger = -1` (the disabled
    ///   sentinel)
    /// - `level0_stop_writes_trigger = i32::MAX`
    ///
    /// # Schema-set tip defaults are not preserved
    ///
    /// Per-CF tip-mode values for these specific keys set via
    /// [`Schema::cfs`](crate::Schema::cfs) at open time are *not*
    /// captured for reversal by
    /// [`set_tip_options_cf`](Self::set_tip_options_cf), which
    /// applies RocksDB defaults. A schema that needs non-default
    /// values for these knobs at tip should re-apply them via
    /// [`set_options_cf`](Self::set_options_cf) after the tip-mode
    /// toggle.
    pub fn set_restore_options_cf(&self, cf_name: &str) -> Result<(), Error> {
        // The string values are RocksDB option-parser inputs:
        // bools are "true"/"false", integers are decimal strings.
        let max = i32::MAX.to_string();
        self.set_options_cf(
            cf_name,
            &[
                ("disable_auto_compactions", "true"),
                ("level0_file_num_compaction_trigger", &max),
                ("level0_slowdown_writes_trigger", "-1"),
                ("level0_stop_writes_trigger", &max),
            ],
        )
    }

    /// Apply tip-mode compaction settings to `cf_name`.
    ///
    /// Restores RocksDB defaults for the four compaction-trigger
    /// knobs raised by
    /// [`set_restore_options_cf`](Self::set_restore_options_cf).
    /// The values applied are the upstream defaults from
    /// [`advanced_options.h`]:
    ///
    /// - `disable_auto_compactions = false`
    /// - `level0_file_num_compaction_trigger = 4`
    /// - `level0_slowdown_writes_trigger = 20`
    /// - `level0_stop_writes_trigger = 36`
    ///
    /// [`advanced_options.h`]: https://github.com/facebook/rocksdb/blob/main/include/rocksdb/advanced_options.h
    pub fn set_tip_options_cf(&self, cf_name: &str) -> Result<(), Error> {
        self.set_options_cf(
            cf_name,
            &[
                ("disable_auto_compactions", "false"),
                ("level0_file_num_compaction_trigger", "4"),
                ("level0_slowdown_writes_trigger", "20"),
                ("level0_stop_writes_trigger", "36"),
            ],
        )
    }

    /// Atomically ingest a set of pre-built SST files into the
    /// column family named `cf_name`.
    ///
    /// SSTs are typically produced by
    /// [`SstWriter`](crate::SstWriter). Each SST must contain keys
    /// in strict comparator order; across SSTs in a single call,
    /// files may overlap or not. RocksDB places each file at the
    /// lowest LSM level whose key range does not overlap existing
    /// data, so ingesting into an empty CF lands files at the
    /// bottommost level — bypassing the memtable, the WAL, and L0
    /// entirely.
    ///
    /// # Behavior on overlap
    ///
    /// - With memtable entries: RocksDB flushes the memtable before
    ///   ingestion (controlled by `allow_blocking_flush`, default
    ///   true).
    /// - With existing SST data: RocksDB walks levels until it finds
    ///   one with no overlap, falling back to L0. Ingested keys
    ///   shadow earlier values at the same key.
    /// - With in-memory snapshots taken before this call:
    ///   `snapshot_consistency` (default true) ensures pre-ingest
    ///   snapshots do not see the ingested keys.
    ///
    /// # Defaults
    ///
    /// This method uses
    /// [`IngestExternalFileOptions::default`](rocksdb::IngestExternalFileOptions)
    /// with `move_files = true`. Moving rather than copying requires
    /// the SST files to live on the same filesystem as the
    /// database directory; otherwise the call fails. Callers that
    /// need different options should use
    /// [`ingest_files_cf_opts`](Self::ingest_files_cf_opts).
    pub fn ingest_files_cf<P: AsRef<Path>>(
        &self,
        cf_name: &str,
        paths: Vec<P>,
    ) -> Result<(), Error> {
        let mut opts = rocksdb::IngestExternalFileOptions::default();
        opts.set_move_files(true);
        self.ingest_files_cf_opts(cf_name, &opts, paths)
    }

    /// Atomically ingest SST files with caller-supplied
    /// [`IngestExternalFileOptions`](rocksdb::IngestExternalFileOptions).
    ///
    /// Use this when the defaults from
    /// [`ingest_files_cf`](Self::ingest_files_cf) do not fit — for
    /// example, to disable `move_files` when the SSTs live on a
    /// different filesystem, or to enable `ingest_behind` for an
    /// append-only restore on a CF whose schema opened the database
    /// with `allow_ingest_behind=true`.
    pub fn ingest_files_cf_opts<P: AsRef<Path>>(
        &self,
        cf_name: &str,
        opts: &rocksdb::IngestExternalFileOptions,
        paths: Vec<P>,
    ) -> Result<(), Error> {
        let cf = self
            .cf_handle(cf_name)
            .ok_or_else(|| Error::MissingColumnFamily(cf_name.to_string()))?;
        self.inner
            .db
            .ingest_external_file_cf_opts(&cf, opts, paths)?;
        Ok(())
    }

    /// Flush all column families to disk, blocking until each
    /// memtable has been written.
    ///
    /// Equivalent to RocksDB's `flush()` operation. Useful before a
    /// graceful shutdown or before opening a [filesystem
    /// checkpoint][rocksdb::checkpoint::Checkpoint] of the
    /// database. Routine writes do not require this call; RocksDB
    /// flushes automatically as memtables fill.
    pub fn flush(&self) -> Result<(), Error> {
        self.inner.db.flush()?;
        Ok(())
    }

    /// Drop the snapshot at `checkpoint`. Returns `true` if a
    /// snapshot was removed.
    ///
    /// Outstanding [`Snapshot`](crate::Snapshot) values for this
    /// checkpoint remain usable until they themselves drop; only the
    /// buffer's reference is released.
    pub fn drop_snapshot(&self, checkpoint: u64) -> bool {
        self.inner.snapshots.write().remove(&checkpoint).is_some()
    }

    /// Drop a column family at runtime.
    ///
    /// Returns an [`Error`] if the column family does not exist or
    /// the underlying RocksDB call fails. After a successful drop,
    /// any outstanding [`DbMap`](crate::DbMap) handle that targeted
    /// the dropped CF will fail subsequent operations with
    /// [`Error::MissingColumnFamily`].
    ///
    /// The caller is responsible for ensuring no other thread is
    /// concurrently issuing reads or writes against the CF being
    /// dropped — a concurrent operation is technically synchronized
    /// by RocksDB but may surface as a `MissingColumnFamily` error
    /// at an unpredictable moment.
    pub fn drop_cf(&self, cf_name: &str) -> Result<(), Error> {
        self.inner.db.drop_cf(cf_name)?;
        Ok(())
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
                .db
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
        _db: Db,
    }

    impl Schema for TestSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
            vec![
                CfDescriptor::new("foo", base_options.clone()),
                CfDescriptor::new("bar", base_options.clone()),
            ]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
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
    fn flush_succeeds_on_open_db() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        db.flush().unwrap();
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

    #[test]
    fn drop_cf_removes_the_cf_at_runtime() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        assert!(db.cf_handle("foo").is_some());
        db.drop_cf("foo").unwrap();
        assert!(db.cf_handle("foo").is_none());
    }

    #[test]
    fn drop_cf_unknown_cf_is_an_error() {
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let err = db.drop_cf("not_in_schema").unwrap_err();
        assert!(matches!(err, Error::Rocksdb(_)));
    }

    #[test]
    fn data_persists_across_db_close_and_reopen() {
        // Mirrors alt's test_persistence (minus the framework's
        // watermark concerns). Writes through Batch survive a Db
        // drop and a fresh open at the same path; in-memory
        // snapshots do not (the buffer starts empty after reopen).
        use crate::DbMap;
        use crate::Encode;
        use crate::error::DecodeError;
        use crate::error::EncodeError;

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct U64Be(u64);

        impl Encode for U64Be {
            fn encode_into<B: bytes::BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
                buf.put_slice(&self.0.to_be_bytes());
                Ok(())
            }
        }

        impl crate::Decode for U64Be {
            fn decode<B: bytes::Buf>(buf: &mut B) -> Result<Self, DecodeError> {
                if buf.remaining() != 8 {
                    return Err(DecodeError::msg("expected 8 bytes"));
                }
                Ok(Self(buf.get_u64()))
            }
        }

        #[derive(Debug)]
        struct PersistSchema {
            items: DbMap<U64Be, U64Be>,
        }

        impl Schema for PersistSchema {
            fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
                vec![CfDescriptor::new("items", base_options.clone())]
            }

            fn open(db: &Db) -> Result<Self, OpenError> {
                Ok(Self {
                    items: DbMap::new(db.clone(), "items")?,
                })
            }
        }

        let dir = TempDir::new().unwrap();
        {
            let (db, schema) = Db::open::<PersistSchema>(dir.path(), DbOptions::default()).unwrap();
            let mut batch = db.batch();
            batch.put(&schema.items, &U64Be(42), &U64Be(43)).unwrap();
            batch.commit().unwrap();
            // Live-tip read confirms before close.
            assert_eq!(schema.items.get(&U64Be(42)).unwrap(), Some(U64Be(43)));
        }
        // Drop everything, then reopen at the same path.
        let (_db2, schema2) = Db::open::<PersistSchema>(dir.path(), DbOptions::default()).unwrap();
        assert_eq!(schema2.items.get(&U64Be(42)).unwrap(), Some(U64Be(43)));
    }

    #[test]
    fn dbmap_reads_fail_with_missing_cf_after_drop_cf() {
        use crate::DbMap;
        use crate::Encode;
        use crate::error::EncodeError;

        struct Bytes;
        impl Encode for Bytes {
            fn encode_into<B: bytes::BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
                buf.put_slice(b"k");
                Ok(())
            }
        }

        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let map: DbMap<Bytes, Vec<u8>> = DbMap::new(db.clone(), "foo").unwrap();

        // Dropping the CF underneath an existing handle: subsequent
        // reads surface MissingColumnFamily rather than panicking or
        // succeeding silently.
        db.drop_cf("foo").unwrap();
        let err = map.get(&Bytes).unwrap_err();
        assert!(matches!(err, Error::MissingColumnFamily(_)));
    }

    mod options_toggle {
        //! Tests for [`Db::set_options_cf`],
        //! [`Db::set_restore_options_cf`], and
        //! [`Db::set_tip_options_cf`].

        use bytes::BufMut;
        use tempfile::TempDir;

        use super::*;
        use crate::DbMap;
        use crate::Encode;
        use crate::error::EncodeError;

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct U64Be(u64);

        impl Encode for U64Be {
            fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
                buf.put_slice(&self.0.to_be_bytes());
                Ok(())
            }
        }

        impl crate::Decode for U64Be {
            fn decode<B: bytes::Buf>(buf: &mut B) -> Result<Self, crate::error::DecodeError> {
                if buf.remaining() != 8 {
                    return Err(crate::error::DecodeError::msg("expected 8 bytes"));
                }
                Ok(Self(buf.get_u64()))
            }
        }

        #[derive(Debug)]
        struct ItemsSchema {
            items: DbMap<U64Be, U64Be>,
        }

        impl Schema for ItemsSchema {
            fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
                vec![CfDescriptor::new("items", base_options.clone())]
            }

            fn open(db: &Db) -> Result<Self, OpenError> {
                Ok(Self {
                    items: DbMap::new(db.clone(), "items")?,
                })
            }
        }

        #[test]
        fn set_options_cf_applies_known_mutable_options() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            // `disable_auto_compactions` is a documented mutable
            // option; setting both true and false must succeed.
            db.set_options_cf("items", &[("disable_auto_compactions", "true")])
                .unwrap();
            db.set_options_cf("items", &[("disable_auto_compactions", "false")])
                .unwrap();
        }

        #[test]
        fn set_options_cf_unknown_key_returns_rocksdb_error() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            let err = db
                .set_options_cf("items", &[("not_a_real_option", "1")])
                .unwrap_err();
            assert!(matches!(err, Error::Rocksdb(_)));
        }

        #[test]
        fn set_options_cf_unknown_cf_returns_missing_column_family() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            let err = db
                .set_options_cf("not_in_schema", &[("disable_auto_compactions", "true")])
                .unwrap_err();
            assert!(matches!(err, Error::MissingColumnFamily(_)));
        }

        #[test]
        fn set_options_cf_empty_slice_is_ok() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            db.set_options_cf("items", &[]).unwrap();
        }

        #[test]
        fn restore_then_tip_toggles_succeed() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            db.set_restore_options_cf("items").unwrap();
            db.set_tip_options_cf("items").unwrap();
        }

        #[test]
        fn restore_options_unknown_cf_errors() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            let err = db.set_restore_options_cf("not_in_schema").unwrap_err();
            assert!(matches!(err, Error::MissingColumnFamily(_)));
        }

        #[test]
        fn tip_options_unknown_cf_errors() {
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            let err = db.set_tip_options_cf("not_in_schema").unwrap_err();
            assert!(matches!(err, Error::MissingColumnFamily(_)));
        }

        #[test]
        fn writes_proceed_under_restore_options() {
            // Bulk-load shape: many writes into a CF whose
            // compaction has been frozen. The L0 stop trigger is at
            // i32::MAX so no slowdown / stop fires; writes complete
            // without error.
            let dir = TempDir::new().unwrap();
            let (db, schema) = Db::open::<ItemsSchema>(dir.path(), DbOptions::default()).unwrap();
            db.set_restore_options_cf("items").unwrap();

            // Enough writes plus flushes to produce multiple L0
            // files. Under default tip options the L0 stop trigger
            // (36) would not be reached either, but the test still
            // exercises that writes go through the path we've
            // toggled.
            for batch_id in 0..8u64 {
                let mut batch = db.batch();
                for i in 0..256u64 {
                    let k = batch_id * 1024 + i;
                    batch.put(&schema.items, &U64Be(k), &U64Be(k)).unwrap();
                }
                batch.commit().unwrap();
                db.flush().unwrap();
            }

            db.set_tip_options_cf("items").unwrap();

            // Continuing writes under tip options also succeed.
            let mut batch = db.batch();
            batch
                .put(&schema.items, &U64Be(9999), &U64Be(9999))
                .unwrap();
            batch.commit().unwrap();

            assert_eq!(
                schema.items.get(&U64Be(0)).unwrap(),
                Some(U64Be(0)),
                "data written under restore options must still be readable after tip toggle",
            );
            assert_eq!(schema.items.get(&U64Be(9999)).unwrap(), Some(U64Be(9999)),);
        }

        #[test]
        fn restore_toggle_is_per_cf() {
            // Two CFs, restore-mode on one only; the other keeps
            // its tip defaults. Verified by exercising both CFs and
            // confirming both write paths succeed independently.
            #[derive(Debug)]
            struct TwoCfSchema {
                a: DbMap<U64Be, U64Be>,
                b: DbMap<U64Be, U64Be>,
            }

            impl Schema for TwoCfSchema {
                fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
                    vec![
                        CfDescriptor::new("a", base_options.clone()),
                        CfDescriptor::new("b", base_options.clone()),
                    ]
                }

                fn open(db: &Db) -> Result<Self, OpenError> {
                    Ok(Self {
                        a: DbMap::new(db.clone(), "a")?,
                        b: DbMap::new(db.clone(), "b")?,
                    })
                }
            }

            let dir = TempDir::new().unwrap();
            let (db, schema) = Db::open::<TwoCfSchema>(dir.path(), DbOptions::default()).unwrap();
            db.set_restore_options_cf("a").unwrap();

            let mut batch = db.batch();
            batch.put(&schema.a, &U64Be(1), &U64Be(10)).unwrap();
            batch.put(&schema.b, &U64Be(2), &U64Be(20)).unwrap();
            batch.commit().unwrap();

            db.set_tip_options_cf("a").unwrap();
            assert_eq!(schema.a.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
            assert_eq!(schema.b.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
        }
    }

    mod ingest {
        //! Tests for [`Db::ingest_files_cf`] and
        //! [`Db::ingest_files_cf_opts`].

        use bytes::Buf;
        use bytes::BufMut;
        use tempfile::TempDir;

        use super::*;
        use crate::DbMap;
        use crate::Decode;
        use crate::Encode;
        use crate::SstWriter;
        use crate::error::DecodeError;
        use crate::error::EncodeError;

        /// Hand-rolled big-endian `u64`. The byte representation
        /// matches the comparator order RocksDB uses by default, so
        /// the encoded form sorts the same way the typed values do.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct U64Be(u64);

        impl Encode for U64Be {
            fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
                buf.put_slice(&self.0.to_be_bytes());
                Ok(())
            }
        }

        impl Decode for U64Be {
            fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
                if buf.remaining() != 8 {
                    return Err(DecodeError::msg("expected 8 bytes"));
                }
                Ok(Self(buf.get_u64()))
            }
        }

        /// One-CF schema. Used by the put-flavored ingest tests.
        #[derive(Debug)]
        struct IngestSchema {
            items: DbMap<U64Be, U64Be>,
        }

        impl Schema for IngestSchema {
            fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
                vec![CfDescriptor::new("items", base_options.clone())]
            }

            fn open(db: &Db) -> Result<Self, OpenError> {
                Ok(Self {
                    items: DbMap::new(db.clone(), "items")?,
                })
            }
        }

        /// Associative merge operator that sums big-endian `u64`s.
        /// Mirrors the operator in `batch::tests` so the ingested
        /// merge entries combine the way a tip-mode merge would.
        fn add_u64_merge_op(
            _key: &[u8],
            existing: Option<&[u8]>,
            operands: &rocksdb::MergeOperands,
        ) -> Option<Vec<u8>> {
            let mut total: u64 = existing
                .and_then(|b| <[u8; 8]>::try_from(b).ok())
                .map(u64::from_be_bytes)
                .unwrap_or(0);
            for op in operands {
                if let Ok(arr) = <[u8; 8]>::try_from(op) {
                    total = total.saturating_add(u64::from_be_bytes(arr));
                }
            }
            Some(total.to_be_bytes().to_vec())
        }

        /// One-CF schema whose CF has the `add_u64` merge operator
        /// installed. Used by the merge-flavored ingest test.
        #[derive(Debug)]
        struct MergeIngestSchema {
            counters: DbMap<U64Be, U64Be>,
        }

        impl Schema for MergeIngestSchema {
            fn cfs(base_options: &rocksdb::Options) -> Vec<CfDescriptor> {
                let mut counter_opts = base_options.clone();
                counter_opts.set_merge_operator_associative("u64-add", add_u64_merge_op);
                // These tests exercise raw SST ingestion with merge
                // entries against the foundational [`Db::ingest_files_cf`]
                // API directly (not through the shard-backed [`Batch`]),
                // so the per-CF restore mode is irrelevant here — the
                // default `BulkIngest` is fine.
                vec![CfDescriptor::new("counters", counter_opts)]
            }

            fn open(db: &Db) -> Result<Self, OpenError> {
                Ok(Self {
                    counters: DbMap::new(db.clone(), "counters")?,
                })
            }
        }

        /// Build a finalized SST at `path` containing the supplied
        /// (key, value) pairs as `put`s. Caller ensures keys are
        /// sorted.
        fn build_put_sst(path: &std::path::Path, kvs: &[(u64, u64)]) {
            let mut w: SstWriter<U64Be, U64Be> =
                SstWriter::create(path, rocksdb::Options::default()).unwrap();
            for (k, v) in kvs {
                w.put(&U64Be(*k), &U64Be(*v)).unwrap();
            }
            let _ = w.finish().unwrap();
        }

        /// Build a finalized SST at `path` containing the supplied
        /// (key, operand) pairs as `merge` operations.
        fn build_merge_sst(path: &std::path::Path, kvs: &[(u64, u64)]) {
            let mut w: SstWriter<U64Be, U64Be> =
                SstWriter::create(path, rocksdb::Options::default()).unwrap();
            for (k, v) in kvs {
                w.merge(&U64Be(*k), &U64Be(*v)).unwrap();
            }
            let _ = w.finish().unwrap();
        }

        #[test]
        fn ingest_into_empty_cf_makes_keys_visible() {
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let path = sst_dir.path().join("init.sst");
            build_put_sst(&path, &[(1, 10), (2, 20), (3, 30)]);
            db.ingest_files_cf("items", vec![path]).unwrap();

            assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
            assert_eq!(schema.items.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
            assert_eq!(schema.items.get(&U64Be(3)).unwrap(), Some(U64Be(30)));
            assert!(schema.items.get(&U64Be(4)).unwrap().is_none());
        }

        #[test]
        fn ingest_preserves_non_overlapping_existing_data() {
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            // Pre-populate a low-range key via the normal write path.
            let mut batch = db.batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
            batch.commit().unwrap();

            // Ingest a non-overlapping high range via SST.
            let path = sst_dir.path().join("high.sst");
            build_put_sst(&path, &[(10, 1000), (11, 1100)]);
            db.ingest_files_cf("items", vec![path]).unwrap();

            assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(100)));
            assert_eq!(schema.items.get(&U64Be(10)).unwrap(), Some(U64Be(1000)));
            assert_eq!(schema.items.get(&U64Be(11)).unwrap(), Some(U64Be(1100)));
        }

        #[test]
        fn ingested_keys_shadow_prior_writes_on_overlap() {
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let mut batch = db.batch();
            batch.put(&schema.items, &U64Be(1), &U64Be(100)).unwrap();
            batch.commit().unwrap();
            // Force the prior write down to an SST so the ingest
            // sees existing-on-disk overlap rather than memtable
            // overlap.
            db.flush().unwrap();

            let path = sst_dir.path().join("shadow.sst");
            build_put_sst(&path, &[(1, 999)]);
            db.ingest_files_cf("items", vec![path]).unwrap();

            assert_eq!(schema.items.get(&U64Be(1)).unwrap(), Some(U64Be(999)));
        }

        #[test]
        fn ingest_with_memtable_overlap_flushes_then_succeeds() {
            // `allow_blocking_flush` defaults to true, so the
            // memtable is flushed before ingestion of an SST whose
            // key range overlaps memtable contents. The post-ingest
            // value at the overlapping key is the ingested one.
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let mut batch = db.batch();
            batch.put(&schema.items, &U64Be(7), &U64Be(70)).unwrap();
            batch.commit().unwrap();
            // No explicit flush — the entry sits in the memtable
            // for the duration of the ingest call.

            let path = sst_dir.path().join("memover.sst");
            build_put_sst(&path, &[(7, 777)]);
            db.ingest_files_cf("items", vec![path]).unwrap();

            assert_eq!(schema.items.get(&U64Be(7)).unwrap(), Some(U64Be(777)));
        }

        #[test]
        fn ingest_with_merge_operator_combines_at_read_time() {
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) =
                Db::open::<MergeIngestSchema>(dir.path(), DbOptions::default()).unwrap();

            // Pre-populate via the normal merge path.
            let mut batch = db.batch();
            batch
                .merge(&schema.counters, &U64Be(1), &U64Be(10))
                .unwrap();
            batch.commit().unwrap();

            // Ingest a merge SST against the same key plus a fresh key.
            let path = sst_dir.path().join("merges.sst");
            build_merge_sst(&path, &[(1, 5), (2, 100)]);
            db.ingest_files_cf("counters", vec![path]).unwrap();

            // Key 1: 10 (prior) + 5 (ingested merge) = 15. The
            // operator runs at read time when it sees both the prior
            // write and the ingested merge entry.
            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(15)));
            // Key 2: 100 (ingested merge alone, no prior value).
            assert_eq!(schema.counters.get(&U64Be(2)).unwrap(), Some(U64Be(100)));
        }

        #[test]
        fn ingest_multiple_non_overlapping_ssts_in_one_call() {
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let p1 = sst_dir.path().join("a.sst");
            let p2 = sst_dir.path().join("b.sst");
            build_put_sst(&p1, &[(1, 10), (2, 20)]);
            build_put_sst(&p2, &[(100, 1000), (200, 2000)]);
            db.ingest_files_cf("items", vec![p1, p2]).unwrap();

            assert_eq!(schema.items.get(&U64Be(2)).unwrap(), Some(U64Be(20)));
            assert_eq!(schema.items.get(&U64Be(100)).unwrap(), Some(U64Be(1000)));
        }

        #[test]
        fn ingest_unknown_cf_returns_missing_column_family() {
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, _schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let path = sst_dir.path().join("orphan.sst");
            build_put_sst(&path, &[(1, 10)]);
            let err = db.ingest_files_cf("not_in_schema", vec![path]).unwrap_err();
            assert!(matches!(err, Error::MissingColumnFamily(_)));
        }

        #[test]
        fn ingest_empty_paths_vec_returns_rocksdb_error() {
            // RocksDB requires at least one path; surface that as a
            // Rocksdb error rather than silently no-op'ing.
            let dir = TempDir::new().unwrap();
            let (db, _schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();
            let err = db
                .ingest_files_cf::<&std::path::Path>("items", vec![])
                .expect_err("empty paths must fail");
            assert!(matches!(err, Error::Rocksdb(_)));
        }

        #[test]
        fn ingest_files_cf_opts_can_disable_move_files() {
            // Verify the explicit-options entry point routes through.
            // With move_files=false the SST file is copied; the
            // source path remains visible to the caller after the
            // ingest succeeds.
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let path = sst_dir.path().join("copy.sst");
            build_put_sst(&path, &[(42, 4200)]);

            let mut opts = rocksdb::IngestExternalFileOptions::default();
            opts.set_move_files(false);
            db.ingest_files_cf_opts("items", &opts, vec![&path])
                .unwrap();

            assert_eq!(schema.items.get(&U64Be(42)).unwrap(), Some(U64Be(4200)));
            assert!(
                path.exists(),
                "with move_files=false the source SST must still exist after ingest",
            );
        }

        #[test]
        fn ingest_with_overlap_visible_to_post_ingest_snapshot() {
            // `snapshot_consistency=true` (the default) means
            // snapshots taken *before* an ingest must not see
            // ingested keys, while snapshots taken *after* the ingest
            // must see them. This test pins the second half.
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) = Db::open::<IngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let path = sst_dir.path().join("for_snap.sst");
            build_put_sst(&path, &[(1, 10)]);
            db.ingest_files_cf("items", vec![path]).unwrap();

            db.take_snapshot(1);
            let snap = db.at_snapshot(1).unwrap();
            let snap_items = schema.items.at(&snap);
            assert_eq!(snap_items.get(&U64Be(1)).unwrap(), Some(U64Be(10)));
        }

        // The next two tests pin the cross-SST merge story that the
        // restore design depends on. Multiple shards (modeled as
        // separate SST files) may each emit a merge entry for the
        // same key — at most one per shard, since `SstFileWriter`
        // rejects duplicate consecutive keys — and after ingest the
        // merge operator must combine them into the correct
        // aggregate value. This is the foundation that makes the
        // restore-via-SST design correct for merge-coalescing
        // schemas like `balances` (one merge per (owner, type) key
        // per shard, many shards across the live object set).

        #[test]
        fn cross_sst_merges_combine_via_operator_after_ingest() {
            // Three SSTs, each contributing one merge for the same
            // key plus one merge for a shard-local key. After
            // ingesting all three, reads must combine all three
            // shared-key merges via the operator while keeping the
            // shard-local keys independent.
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) =
                Db::open::<MergeIngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let p1 = sst_dir.path().join("shard1.sst");
            let p2 = sst_dir.path().join("shard2.sst");
            let p3 = sst_dir.path().join("shard3.sst");
            // (key=1 is the shared key; keys 10/11/12 are per-shard.)
            build_merge_sst(&p1, &[(1, 10), (10, 1000)]);
            build_merge_sst(&p2, &[(1, 20), (11, 1100)]);
            build_merge_sst(&p3, &[(1, 30), (12, 1200)]);

            // Ingest one at a time (the realistic restore flow:
            // shards finalize independently, each triggers its own
            // ingest). Each call lands the SST at the bottom level
            // because the key ranges of these three files overlap
            // only at the shared key.
            db.ingest_files_cf("counters", vec![p1]).unwrap();
            db.ingest_files_cf("counters", vec![p2]).unwrap();
            db.ingest_files_cf("counters", vec![p3]).unwrap();

            // Shared key: operator sums all three merge operands.
            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(60)));
            // Per-shard keys: each survives with its sole operand.
            assert_eq!(schema.counters.get(&U64Be(10)).unwrap(), Some(U64Be(1000)));
            assert_eq!(schema.counters.get(&U64Be(11)).unwrap(), Some(U64Be(1100)));
            assert_eq!(schema.counters.get(&U64Be(12)).unwrap(), Some(U64Be(1200)));

            // The merge must continue to apply correctly after a
            // manual compaction (which folds the merge entries into
            // the base value). This pins the post-compaction behavior
            // so a future restore is not surprised by a deferred
            // application of the operator.
            let cf = db.cf_handle("counters").unwrap();
            db.rocksdb()
                .compact_range_cf(&cf, None::<&[u8]>, None::<&[u8]>);
            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(60)));
            assert_eq!(schema.counters.get(&U64Be(10)).unwrap(), Some(U64Be(1000)));
        }

        #[test]
        fn cross_sst_merges_combine_when_ingested_together() {
            // Same shape as the previous test, but all three SSTs
            // are passed to a single `ingest_files_cf` call.
            // RocksDB places each file individually based on its
            // own key range, so the combine semantics are
            // identical, but the call site is one batched ingest
            // rather than three sequential ones — a path some
            // drivers may prefer.
            let dir = TempDir::new().unwrap();
            let sst_dir = TempDir::new_in(dir.path()).unwrap();
            let (db, schema) =
                Db::open::<MergeIngestSchema>(dir.path(), DbOptions::default()).unwrap();

            let p1 = sst_dir.path().join("shard1.sst");
            let p2 = sst_dir.path().join("shard2.sst");
            let p3 = sst_dir.path().join("shard3.sst");
            build_merge_sst(&p1, &[(1, 7)]);
            build_merge_sst(&p2, &[(1, 11)]);
            build_merge_sst(&p3, &[(1, 13)]);

            db.ingest_files_cf("counters", vec![p1, p2, p3]).unwrap();

            assert_eq!(schema.counters.get(&U64Be(1)).unwrap(), Some(U64Be(31)));
        }
    }
}
