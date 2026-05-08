// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed column-family handles.
//!
//! [`DbMap<K, V>`] is the primary read and write surface in the
//! crate. Each instance is tied to a single column family on an
//! [`Arc<Db>`] and to a key type and a value type that implement the
//! crate's encoding traits. Schemas are typically structs of `DbMap`
//! fields; see the [`Schema`](crate::Schema) trait for the
//! construction pattern.
//!
//! # Reading
//!
//! Two read methods are exposed for point lookups:
//!
//! - [`DbMap::get`] decodes the value into an owned `V` and is the
//!   common path. Internally it routes through RocksDB's
//!   `get_pinned_cf` to avoid the redundant copy that `DB::get`
//!   performs into a fresh `Vec<u8>`.
//! - [`DbMap::get_raw`] returns the raw value bytes as a
//!   [`bytes::Bytes`]. The `Bytes` is backed zero-copy by the
//!   RocksDB block cache where possible (cache hits) and by an
//!   internal copy where not (memtable hits, merge results,
//!   wide-column values). The handle co-owns the [`Arc<Db>`] so it
//!   is sound to hold the `Bytes` past the borrow it was read
//!   through.
//!
//! Both methods have batched counterparts ([`DbMap::multi_get`] and
//! [`DbMap::multi_get_raw`]) that issue a single RocksDB
//! `batched_multi_get_cf` call rather than N independent reads.
//!
//! # A note on pinned reads
//!
//! Each outstanding `Bytes` clone of a block-cache-backed value pins
//! an LRU handle in RocksDB's block cache. Long-lived or unbounded
//! pins can drive `block_cache.pinned-usage` past
//! `block_cache.capacity()`. Hold the `Bytes` for as short a scope as
//! the application allows, especially on high-fanout query paths.

use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

use bytes::Bytes;
use rocksdb::DBPinnableSlice;

use crate::Decode;
use crate::Encode;
use crate::db::Db;
use crate::encode_buf::with_encode_buf;
use crate::error::Error;
use crate::error::OpenError;

/// A typed handle to a single column family on a [`Db`].
///
/// Construct one with [`DbMap::new`], typically inside a schema's
/// [`Schema::open`](crate::Schema::open) implementation.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use sui_consistent_store::Db;
/// use sui_consistent_store::DbMap;
/// use sui_consistent_store::DbOptions;
/// use sui_consistent_store::Decode;
/// use sui_consistent_store::Encode;
/// use sui_consistent_store::Schema;
/// use sui_consistent_store::error::DecodeError;
/// use sui_consistent_store::error::EncodeError;
/// use sui_consistent_store::error::OpenError;
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct U64Be(u64);
///
/// impl Encode for U64Be {
///     fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
///         buf.extend_from_slice(&self.0.to_be_bytes());
///         Ok(())
///     }
/// }
///
/// impl Decode for U64Be {
///     fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
///         let arr: [u8; 8] = bytes
///             .try_into()
///             .map_err(|_| DecodeError::msg("expected 8 bytes"))?;
///         Ok(Self(u64::from_be_bytes(arr)))
///     }
/// }
///
/// struct MySchema {
///     items: DbMap<U64Be, U64Be>,
/// }
///
/// impl Schema for MySchema {
///     fn cfs() -> Vec<(String, rocksdb::Options)> {
///         vec![("items".to_string(), rocksdb::Options::default())]
///     }
///
///     fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
///         Ok(Self {
///             items: DbMap::new(db.clone(), "items")?,
///         })
///     }
/// }
///
/// let dir = tempfile::tempdir().unwrap();
/// let (_db, schema) = Db::open::<MySchema>(dir.path(), DbOptions::default()).unwrap();
/// // Nothing was inserted, so a lookup returns None.
/// assert!(schema.items.get(&U64Be(1)).unwrap().is_none());
/// ```
#[derive(Debug)]
pub struct DbMap<K, V> {
    db: Arc<Db>,
    cf_name: String,
    _data: PhantomData<fn(K) -> V>,
}

/// Owner struct used to tie a `DBPinnableSlice` to an `Arc<Db>` so
/// the slice can be wrapped in `bytes::Bytes::from_owner`. The
/// `'static` lifetime on the slice is justified by the `Arc<Db>`
/// co-owner; see `pinned_to_bytes` for the safety argument.
///
/// Field declaration order is load-bearing: `slice` must drop before
/// `_db` so the cleanup function the pin runs on `Drop` (typically a
/// block-cache `Cache::Release`) executes while the underlying DB
/// allocation is still alive.
struct PinnedOwner {
    slice: DBPinnableSlice<'static>,
    _db: Arc<Db>,
}

impl<K, V> DbMap<K, V> {
    /// Construct a typed handle for the column family named `cf_name`.
    ///
    /// Returns an [`OpenError`] if the named column family is not
    /// registered on `db`. Construction is the only place where a
    /// missing column family is reported as an open-time error;
    /// subsequent operations report it as
    /// [`Error::MissingColumnFamily`](crate::error::Error::MissingColumnFamily).
    pub fn new(db: Arc<Db>, cf_name: impl Into<String>) -> Result<Self, OpenError> {
        let cf_name = cf_name.into();
        if db.cf_handle(&cf_name).is_none() {
            return Err(OpenError::msg(format!(
                "column family `{cf_name}` is not registered",
            )));
        }
        Ok(Self {
            db,
            cf_name,
            _data: PhantomData,
        })
    }

    fn cf(&self) -> Result<Arc<rocksdb::BoundColumnFamily<'_>>, Error> {
        self.db
            .cf_handle(&self.cf_name)
            .ok_or_else(|| Error::MissingColumnFamily(self.cf_name.clone()))
    }
}

impl<K, V> DbMap<K, V>
where
    K: Encode,
    V: Decode,
{
    /// Read and decode the value for `key`.
    ///
    /// Returns `Ok(None)` if the key is not present; otherwise
    /// returns the decoded value. Internally uses RocksDB's pinned
    /// read path so that block-cache hits avoid the extra heap
    /// allocation that `DB::get` would do.
    pub fn get(&self, key: &K) -> Result<Option<V>, Error> {
        let cf = self.cf()?;
        with_encode_buf(|buf| {
            key.encode_into(buf)?;
            let pinned = self.db.rocksdb().get_pinned_cf(&cf, buf.as_slice())?;
            match pinned {
                Some(slice) => Ok(Some(V::decode(&slice)?)),
                None => Ok(None),
            }
        })
    }

    /// Batched counterpart to [`get`](Self::get).
    ///
    /// Issues one RocksDB `batched_multi_get_cf` for all the keys
    /// rather than `keys.len()` separate reads. The outer `Result`
    /// captures failures in the encoding step (which abort the whole
    /// batch); each inner `Result` captures per-key read or decode
    /// failures and is reported alongside successful results in the
    /// returned vector.
    pub fn multi_get<'k, I>(&self, keys: I) -> Result<Vec<Result<Option<V>, Error>>, Error>
    where
        I: IntoIterator<Item = &'k K>,
        K: 'k,
    {
        let keys: Vec<&K> = keys.into_iter().collect();
        let cf = self.cf()?;

        with_encode_buf(|buf| {
            let mut offsets = Vec::with_capacity(keys.len() + 1);
            offsets.push(0usize);
            for key in &keys {
                key.encode_into(buf)?;
                offsets.push(buf.len());
            }

            let bytes = buf.as_slice();
            let key_slices: Vec<&[u8]> = offsets
                .windows(2)
                .map(|window| &bytes[window[0]..window[1]])
                .collect();

            let raw_results = self
                .db
                .rocksdb()
                .batched_multi_get_cf(&cf, key_slices, false);

            let decoded = raw_results
                .into_iter()
                .map(|r| match r {
                    Ok(Some(slice)) => V::decode(&slice).map(Some).map_err(Error::Decode),
                    Ok(None) => Ok(None),
                    Err(e) => Err(Error::Rocksdb(e)),
                })
                .collect();
            Ok(decoded)
        })
    }
}

impl<K, V> DbMap<K, V>
where
    K: Encode,
{
    /// Read the raw bytes for `key`.
    ///
    /// Returns `Ok(None)` if the key is not present. The returned
    /// [`Bytes`] is backed zero-copy by the RocksDB block cache when
    /// the read hits a cached block; otherwise it backs onto a small
    /// internal copy made by RocksDB's pinned-read machinery. Either
    /// way, the `Bytes` co-owns the underlying [`Arc<Db>`] so it can
    /// outlive any borrow this method was called through.
    pub fn get_raw(&self, key: &K) -> Result<Option<Bytes>, Error> {
        let cf = self.cf()?;
        with_encode_buf(|buf| {
            key.encode_into(buf)?;
            let pinned = self.db.rocksdb().get_pinned_cf(&cf, buf.as_slice())?;
            Ok(pinned.map(|slice| pinned_to_bytes(self.db.clone(), slice)))
        })
    }

    /// Batched counterpart to [`get_raw`](Self::get_raw).
    ///
    /// Issues one RocksDB `batched_multi_get_cf` for all the keys.
    /// The outer `Result` captures encoding failures; the inner
    /// `Result` captures per-key read failures.
    pub fn multi_get_raw<'k, I>(&self, keys: I) -> Result<Vec<Result<Option<Bytes>, Error>>, Error>
    where
        I: IntoIterator<Item = &'k K>,
        K: 'k,
    {
        let keys: Vec<&K> = keys.into_iter().collect();
        let cf = self.cf()?;

        with_encode_buf(|buf| {
            let mut offsets = Vec::with_capacity(keys.len() + 1);
            offsets.push(0usize);
            for key in &keys {
                key.encode_into(buf)?;
                offsets.push(buf.len());
            }

            let bytes = buf.as_slice();
            let key_slices: Vec<&[u8]> = offsets
                .windows(2)
                .map(|window| &bytes[window[0]..window[1]])
                .collect();

            let raw_results = self
                .db
                .rocksdb()
                .batched_multi_get_cf(&cf, key_slices, false);

            let mapped = raw_results
                .into_iter()
                .map(|r| match r {
                    Ok(Some(slice)) => Ok(Some(pinned_to_bytes(self.db.clone(), slice))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(Error::Rocksdb(e)),
                })
                .collect();
            Ok(mapped)
        })
    }
}

impl AsRef<[u8]> for PinnedOwner {
    fn as_ref(&self) -> &[u8] {
        &self.slice
    }
}

/// Wrap a `DBPinnableSlice` plus its co-owned `Arc<Db>` in a
/// `bytes::Bytes` so callers do not have to reason about RocksDB
/// lifetimes themselves.
///
/// The `'a` lifetime on `DBPinnableSlice<'a>` is a `PhantomData<&'a
/// DB>` annotation only. The actual backing memory is either a
/// reference-counted block in the RocksDB block cache (kept alive
/// until the slice's `Drop` runs the registered cleanup) or a copied
/// buffer owned by the C++ `PinnableSlice` itself (freed when the
/// slice drops). Neither path requires a live `&DB` borrow; both
/// require only that the underlying DB allocation outlive the slice,
/// which the co-owned `Arc<Db>` guarantees. Drop order in
/// `PinnedOwner` (slice first, `Arc` second) ensures the cleanup
/// runs before the `Arc`'s last reference goes away.
fn pinned_to_bytes(db: Arc<Db>, slice: DBPinnableSlice<'_>) -> Bytes {
    // SAFETY: see the function-level doc comment. The lifetime is a
    // conservative phantom annotation; the actual memory pin is
    // sustained by the `Arc<Db>` co-owner held in `PinnedOwner`.
    let slice: DBPinnableSlice<'static> = unsafe { mem::transmute(slice) };
    Bytes::from_owner(PinnedOwner { slice, _db: db })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::DbOptions;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;

    /// Hand-rolled big-endian `u64` for tests.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct U64Be(u64);

    impl Encode for U64Be {
        fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
            buf.extend_from_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    impl Decode for U64Be {
        fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| DecodeError::msg("expected 8 bytes"))?;
            Ok(Self(u64::from_be_bytes(arr)))
        }
    }

    #[derive(Debug)]
    struct TestSchema {
        items: DbMap<U64Be, U64Be>,
    }

    impl Schema for TestSchema {
        fn cfs() -> Vec<(String, rocksdb::Options)> {
            vec![(String::from("items"), rocksdb::Options::default())]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                items: DbMap::new(db.clone(), "items")?,
            })
        }
    }

    /// Seed a `(key, value)` row through the underlying RocksDB so
    /// the read paths under test have something to find. Uses the
    /// `Encode` trait so the bytes match what `DbMap::get` will look
    /// for.
    fn seed(db: &Db, cf: &str, key: &U64Be, value: &U64Be) {
        let cf = db.cf_handle(cf).unwrap();
        let key_bytes = key.encode().unwrap();
        let value_bytes = value.encode().unwrap();
        db.rocksdb().put_cf(&cf, key_bytes, value_bytes).unwrap();
    }

    fn open() -> (TempDir, Arc<Db>, TestSchema) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        (dir, db, schema)
    }

    #[test]
    fn new_errors_on_unknown_cf() {
        let (_dir, db, _schema) = open();
        let err = DbMap::<U64Be, U64Be>::new(db, "missing").unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn get_returns_none_for_missing_key() {
        let (_dir, _db, schema) = open();
        assert!(schema.items.get(&U64Be(42)).unwrap().is_none());
    }

    #[test]
    fn get_returns_decoded_value() {
        let (_dir, db, schema) = open();
        seed(&db, "items", &U64Be(7), &U64Be(700));
        assert_eq!(schema.items.get(&U64Be(7)).unwrap(), Some(U64Be(700)));
    }

    #[test]
    fn get_raw_returns_none_for_missing_key() {
        let (_dir, _db, schema) = open();
        assert!(schema.items.get_raw(&U64Be(42)).unwrap().is_none());
    }

    #[test]
    fn get_raw_returns_value_bytes() {
        let (_dir, db, schema) = open();
        seed(&db, "items", &U64Be(7), &U64Be(700));
        let bytes = schema.items.get_raw(&U64Be(7)).unwrap().unwrap();
        assert_eq!(&bytes[..], &700u64.to_be_bytes());
    }

    #[test]
    fn get_raw_bytes_outlive_schema_drop() {
        let (_dir, db, schema) = open();
        seed(&db, "items", &U64Be(11), &U64Be(1100));
        let bytes = schema.items.get_raw(&U64Be(11)).unwrap().unwrap();
        // Drop the schema (and its DbMap clone of `Arc<Db>`); the
        // `Bytes` still co-owns the DB via `PinnedOwner`.
        drop(schema);
        assert_eq!(&bytes[..], &1100u64.to_be_bytes());
        // And `db` is still in scope, but the test passes whether or
        // not it is.
        drop(db);
    }

    #[test]
    fn multi_get_returns_decoded_values_per_key() {
        let (_dir, db, schema) = open();
        seed(&db, "items", &U64Be(1), &U64Be(10));
        seed(&db, "items", &U64Be(3), &U64Be(30));
        let keys = [U64Be(1), U64Be(2), U64Be(3)];
        let results = schema.items.multi_get(keys.iter()).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap(), &Some(U64Be(10)));
        assert_eq!(results[1].as_ref().unwrap(), &None);
        assert_eq!(results[2].as_ref().unwrap(), &Some(U64Be(30)));
    }

    #[test]
    fn multi_get_raw_returns_bytes_per_key() {
        let (_dir, db, schema) = open();
        seed(&db, "items", &U64Be(5), &U64Be(50));
        let keys = [U64Be(5), U64Be(6)];
        let results = schema.items.multi_get_raw(keys.iter()).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].as_ref().unwrap().as_ref().unwrap()[..],
            50u64.to_be_bytes(),
        );
        assert!(results[1].as_ref().unwrap().is_none());
    }

    #[test]
    fn multi_get_handles_empty_input() {
        let (_dir, _db, schema) = open();
        let keys: [U64Be; 0] = [];
        let results = schema.items.multi_get(keys.iter()).unwrap();
        assert!(results.is_empty());
    }
}
