// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The [`Db`] handle wrapping an opened RocksDB database.
//!
//! [`Db`] is shared across the typed column-family handles that a
//! schema constructs against it; consumers hold an [`Arc<Db>`] for
//! the database's lifetime, with the Drop on the last clone
//! triggering RocksDB's own shutdown sequence (flush plus close).
//!
//! RocksDB is internally thread-safe, so [`Db`] does not impose any
//! external locking in this version of the crate. The in-memory
//! snapshot buffer (which will require an [`std::sync::RwLock`] over
//! the buffer state and a self-referential wrapping for the snapshot
//! lifetimes) lands in a later commit; the public API will not change
//! when it does.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use rocksdb::BoundColumnFamily;

use crate::error::OpenError;
use crate::schema::Schema;

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
    inner: rocksdb::DB,
}

impl Default for DbOptions {
    fn default() -> Self {
        let mut db_options = rocksdb::Options::default();
        db_options.create_if_missing(true);
        db_options.create_missing_column_families(true);
        Self { db_options }
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
        let DbOptions { db_options } = opts;

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

        let db = Arc::new(Self { inner });
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
