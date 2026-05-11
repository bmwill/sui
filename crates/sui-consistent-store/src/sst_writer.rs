// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed builder for sorted SST files, ready for atomic ingestion
//! into a [`Db`](crate::Db) via
//! [`Db::ingest_files_cf`](crate::Db::ingest_files_cf).
//!
//! An [`SstWriter<K, V>`] wraps RocksDB's
//! [`SstFileWriter`](rocksdb::SstFileWriter) and accepts the same
//! typed keys and values as [`DbMap<K, V>`](crate::DbMap) and
//! [`Batch`](crate::Batch), encoded via the crate's
//! [`Encode`](crate::Encode) trait. The resulting `.sst` file can be
//! atomically ingested into a target column family.
//!
//! # Key-ordering contract
//!
//! Keys must be appended in strictly increasing comparator order
//! (byte-wise on the encoded representation). RocksDB returns an
//! error if this contract is violated, so the constraint is
//! detected, not silent; but recovery requires rebuilding the file
//! from scratch. Callers that produce keys from an unordered source
//! should buffer and sort by encoded bytes before driving the
//! writer.
//!
//! # Lifecycle
//!
//! 1. [`SstWriter::create`] opens the file for writing.
//! 2. [`put`](SstWriter::put) / [`merge`](SstWriter::merge) /
//!    [`delete`](SstWriter::delete) append entries.
//! 3. [`finish`](SstWriter::finish) finalizes the file and returns
//!    its path. Dropping the writer without calling `finish` leaves
//!    the file unfinalized on disk; the caller is responsible for
//!    cleaning up partial files.
//!
//! # Examples
//!
//! ```
//! use bytes::BufMut;
//! use sui_consistent_store::Encode;
//! use sui_consistent_store::error::EncodeError;
//! use sui_consistent_store::sst_writer::SstWriter;
//!
//! #[derive(Debug)]
//! struct U64Be(u64);
//!
//! impl Encode for U64Be {
//!     fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
//!         buf.put_slice(&self.0.to_be_bytes());
//!         Ok(())
//!     }
//! }
//!
//! let dir = tempfile::tempdir().unwrap();
//! let path = dir.path().join("items.sst");
//! let mut writer: SstWriter<U64Be, U64Be> =
//!     SstWriter::create(&path, rocksdb::Options::default()).unwrap();
//! writer.put(&U64Be(1), &U64Be(10)).unwrap();
//! writer.put(&U64Be(2), &U64Be(20)).unwrap();
//! let finalized = writer.finish().unwrap();
//! assert_eq!(finalized, path);
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::path::PathBuf;

use crate::Encode;
use crate::encode_buf::with_encode_buf;
use crate::error::Error;

/// A typed builder for a single sorted SST file.
///
/// `K` and `V` parameterize the file's key and value types; both
/// must implement [`Encode`] at the call sites that write entries.
/// The writer owns the [`rocksdb::Options`] used to construct the
/// underlying [`rocksdb::SstFileWriter`] so that callers do not have
/// to manage its lifetime themselves; field-declaration order in the
/// struct is load-bearing for that ownership claim (see safety
/// comments inside [`create`](Self::create)).
pub struct SstWriter<K, V> {
    // Field order is load-bearing: `inner` drops before `_options`.
    // The `'static` lifetime in `SstFileWriter<'static>` is extended
    // from a `&'_ Options` borrow via `mem::transmute`; soundness
    // depends on `_options` outliving `inner`.
    inner: rocksdb::SstFileWriter<'static>,
    _options: Box<rocksdb::Options>,
    path: PathBuf,
    _data: PhantomData<fn(K) -> V>,
}

impl<K, V> SstWriter<K, V> {
    /// Open a new SST file at `path` for writing, using `options`
    /// for table-format and compression settings.
    ///
    /// `options` should typically be the same per-CF options the
    /// target column family was opened with (or at least
    /// compatible: identical comparator, compression family, table
    /// format), so the produced SST can be ingested without
    /// surprise. Merge operators on `options` are *not* required
    /// here — the SST encodes merge operands directly and the
    /// reader-side operator is applied at read or compaction time
    /// after ingestion.
    ///
    /// The parent directory must exist. If a file already lives at
    /// `path`, RocksDB overwrites it; callers that need
    /// already-exists detection must check ahead of the call.
    pub fn create(path: impl AsRef<Path>, options: rocksdb::Options) -> Result<Self, Error> {
        let options = Box::new(options);
        // SAFETY: `SstFileWriter<'a>` borrows `&'a Options`. The
        // transmute to `'static` is sound because the boxed
        // `_options` is declared *after* `inner` in this struct,
        // so on drop `inner` is destroyed before `_options` is
        // freed. No public API exposes the contained writer beyond
        // the struct's lifetime, so the synthetic `'static` borrow
        // never outlives the real `_options` it points into.
        let writer: rocksdb::SstFileWriter<'static> =
            unsafe { std::mem::transmute(rocksdb::SstFileWriter::create(&options)) };

        let path: PathBuf = path.as_ref().to_path_buf();
        writer.open(&path)?;

        Ok(Self {
            inner: writer,
            _options: options,
            path,
            _data: PhantomData,
        })
    }

    /// Append a typed put.
    ///
    /// Encodes `key` and `value` into a thread-local scratch buffer
    /// and forwards the byte slices to RocksDB; RocksDB copies the
    /// bytes synchronously, so the scratch buffer is free for reuse
    /// on return.
    ///
    /// `key` must sort strictly after every key previously appended
    /// to this writer, by byte-wise comparison of the encoded
    /// representation. RocksDB returns
    /// [`Error::Rocksdb`](crate::error::Error::Rocksdb) if the
    /// contract is violated.
    pub fn put(&mut self, key: &K, value: &V) -> Result<(), Error>
    where
        K: Encode,
        V: Encode,
    {
        with_encode_buf(|buf| -> Result<(), Error> {
            key.encode_into(buf)?;
            let k_end = buf.len();
            value.encode_into(buf)?;
            let bytes = buf.as_slice();
            self.inner.put(&bytes[..k_end], &bytes[k_end..])?;
            Ok(())
        })
    }

    /// Append a typed merge operand.
    ///
    /// The encoded `operand` bytes are recorded in the SST as a
    /// merge entry. When the file is ingested and the target column
    /// family's merge operator runs at the next read or compaction,
    /// the operand is combined with any existing value at `key` and
    /// with any other merges (ingested or otherwise) staged against
    /// the same key.
    ///
    /// Same ordering contract as [`put`](Self::put).
    pub fn merge(&mut self, key: &K, operand: &V) -> Result<(), Error>
    where
        K: Encode,
        V: Encode,
    {
        with_encode_buf(|buf| -> Result<(), Error> {
            key.encode_into(buf)?;
            let k_end = buf.len();
            operand.encode_into(buf)?;
            let bytes = buf.as_slice();
            self.inner.merge(&bytes[..k_end], &bytes[k_end..])?;
            Ok(())
        })
    }

    /// Append a typed delete tombstone.
    ///
    /// The deletion is recorded in the SST and applied to any
    /// existing value at `key` in the target column family when the
    /// file is ingested.
    ///
    /// Same ordering contract as [`put`](Self::put).
    pub fn delete(&mut self, key: &K) -> Result<(), Error>
    where
        K: Encode,
    {
        with_encode_buf(|buf| -> Result<(), Error> {
            key.encode_into(buf)?;
            self.inner.delete(buf.as_slice())?;
            Ok(())
        })
    }

    /// Current on-disk size of the SST file in bytes.
    ///
    /// Useful for deciding when to roll over to a new SST during a
    /// long-running restore.
    pub fn file_size(&self) -> u64 {
        self.inner.file_size()
    }

    /// The path the SST is being written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Finalize the SST and return its path.
    ///
    /// After this call the file is closed, durable, and ready for
    /// ingestion via
    /// [`Db::ingest_files_cf`](crate::Db::ingest_files_cf). The
    /// returned path is the one originally passed to
    /// [`create`](Self::create).
    pub fn finish(mut self) -> Result<PathBuf, Error> {
        self.inner.finish()?;
        Ok(self.path.clone())
    }
}

impl<K, V> fmt::Debug for SstWriter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `rocksdb::SstFileWriter` does not implement Debug; report
        // the public, useful surface.
        f.debug_struct("SstWriter")
            .field("path", &self.path)
            .field("file_size", &self.inner.file_size())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use bytes::BufMut;
    use tempfile::TempDir;

    use super::*;
    use crate::error::EncodeError;

    /// Hand-rolled big-endian `u64` so the tests do not pull in a
    /// serialization dependency. The byte representation matches
    /// the comparator order RocksDB uses by default, which is what
    /// [`SstWriter`] requires.
    #[derive(Debug, Clone, Copy)]
    struct U64Be(u64);

    impl Encode for U64Be {
        fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
            buf.put_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    /// Type whose encode always fails. Used to assert that the
    /// encode-error path on a writer is reachable.
    #[derive(Debug)]
    struct AlwaysFails;

    impl Encode for AlwaysFails {
        fn encode_into<B: BufMut>(&self, _: &mut B) -> Result<(), EncodeError> {
            Err(EncodeError::msg("always fails"))
        }
    }

    fn writer_at(dir: &TempDir, name: &str) -> SstWriter<U64Be, U64Be> {
        SstWriter::create(dir.path().join(name), rocksdb::Options::default()).unwrap()
    }

    #[test]
    fn create_writes_a_file_visible_on_finish() {
        let dir = TempDir::new().unwrap();
        let mut w = writer_at(&dir, "ok.sst");
        w.put(&U64Be(1), &U64Be(10)).unwrap();
        w.put(&U64Be(2), &U64Be(20)).unwrap();
        let path = w.finish().unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }

    #[test]
    fn finish_returns_the_original_path() {
        let dir = TempDir::new().unwrap();
        let expected = dir.path().join("named.sst");
        let mut w: SstWriter<U64Be, U64Be> =
            SstWriter::create(&expected, rocksdb::Options::default()).unwrap();
        w.put(&U64Be(1), &U64Be(10)).unwrap();
        let returned = w.finish().unwrap();
        assert_eq!(returned, expected);
    }

    #[test]
    fn create_fails_when_parent_directory_missing() {
        let dir = TempDir::new().unwrap();
        let missing_parent = dir.path().join("does_not_exist").join("out.sst");
        let err = SstWriter::<U64Be, U64Be>::create(missing_parent, rocksdb::Options::default())
            .expect_err("opening into a missing directory should fail");
        assert!(matches!(err, Error::Rocksdb(_)));
    }

    #[test]
    fn create_overwrites_existing_file() {
        // RocksDB's SstFileWriter::open truncates rather than
        // refusing. The wrapper inherits that behavior; this test
        // pins it as the documented contract.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("overwrite.sst");
        std::fs::write(&path, b"prior contents").unwrap();
        let mut w: SstWriter<U64Be, U64Be> =
            SstWriter::create(&path, rocksdb::Options::default()).unwrap();
        w.put(&U64Be(1), &U64Be(10)).unwrap();
        let returned = w.finish().unwrap();
        assert_eq!(returned, path);
        let bytes = std::fs::read(&path).unwrap();
        // Not the prior contents (truncated and rewritten).
        assert_ne!(bytes, b"prior contents");
    }

    #[test]
    fn out_of_order_put_returns_rocksdb_error() {
        let dir = TempDir::new().unwrap();
        let mut w = writer_at(&dir, "ooo.sst");
        w.put(&U64Be(2), &U64Be(20)).unwrap();
        let err = w
            .put(&U64Be(1), &U64Be(10))
            .expect_err("put with smaller key must fail");
        assert!(matches!(err, Error::Rocksdb(_)));
    }

    #[test]
    fn put_with_equal_key_returns_rocksdb_error() {
        // RocksDB requires *strictly* increasing keys, so two puts
        // at the same key are also rejected.
        let dir = TempDir::new().unwrap();
        let mut w = writer_at(&dir, "eq.sst");
        w.put(&U64Be(1), &U64Be(10)).unwrap();
        let err = w
            .put(&U64Be(1), &U64Be(11))
            .expect_err("put with equal key must fail");
        assert!(matches!(err, Error::Rocksdb(_)));
    }

    #[test]
    fn finish_on_empty_writer_returns_rocksdb_error() {
        // RocksDB does not allow finalizing an SST with no entries.
        // Surface that as a Rocksdb error rather than silently
        // producing an unloadable file.
        let dir = TempDir::new().unwrap();
        let w = writer_at(&dir, "empty.sst");
        let err = w.finish().expect_err("empty SST must fail to finish");
        assert!(matches!(err, Error::Rocksdb(_)));
    }

    #[test]
    fn merge_and_delete_record_without_error() {
        let dir = TempDir::new().unwrap();
        let mut w = writer_at(&dir, "ops.sst");
        // Strictly increasing keys across put / merge / delete.
        w.put(&U64Be(1), &U64Be(10)).unwrap();
        w.merge(&U64Be(2), &U64Be(20)).unwrap();
        w.delete(&U64Be(3)).unwrap();
        let path = w.finish().unwrap();
        assert!(path.exists());
    }

    #[test]
    fn file_size_grows_after_appends() {
        let dir = TempDir::new().unwrap();
        let mut w = writer_at(&dir, "grow.sst");
        let initial = w.file_size();
        for i in 1..=64u64 {
            w.put(&U64Be(i), &U64Be(i * 10)).unwrap();
        }
        assert!(
            w.file_size() >= initial,
            "file_size should not shrink across appends",
        );
        let _ = w.finish().unwrap();
    }

    #[test]
    fn put_propagates_encode_error_for_key() {
        let dir = TempDir::new().unwrap();
        let mut w: SstWriter<AlwaysFails, U64Be> =
            SstWriter::create(dir.path().join("ke.sst"), rocksdb::Options::default()).unwrap();
        let err = w
            .put(&AlwaysFails, &U64Be(1))
            .expect_err("encode error on key should surface");
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn put_propagates_encode_error_for_value() {
        let dir = TempDir::new().unwrap();
        let mut w: SstWriter<U64Be, AlwaysFails> =
            SstWriter::create(dir.path().join("ve.sst"), rocksdb::Options::default()).unwrap();
        let err = w
            .put(&U64Be(1), &AlwaysFails)
            .expect_err("encode error on value should surface");
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn merge_propagates_encode_error_for_operand() {
        let dir = TempDir::new().unwrap();
        let mut w: SstWriter<U64Be, AlwaysFails> =
            SstWriter::create(dir.path().join("me.sst"), rocksdb::Options::default()).unwrap();
        let err = w
            .merge(&U64Be(1), &AlwaysFails)
            .expect_err("encode error on operand should surface");
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn delete_propagates_encode_error_for_key() {
        let dir = TempDir::new().unwrap();
        let mut w: SstWriter<AlwaysFails, U64Be> =
            SstWriter::create(dir.path().join("de.sst"), rocksdb::Options::default()).unwrap();
        let err = w
            .delete(&AlwaysFails)
            .expect_err("encode error on key should surface");
        assert!(matches!(err, Error::Encode(_)));
    }

    #[test]
    fn dropping_without_finish_does_not_panic() {
        // Reachable by, e.g., a `?` returning early. Exercise the
        // drop path so any FFI-level surprises surface.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("abandoned.sst");
        {
            let mut w: SstWriter<U64Be, U64Be> =
                SstWriter::create(&path, rocksdb::Options::default()).unwrap();
            w.put(&U64Be(1), &U64Be(10)).unwrap();
            // Drop without calling finish.
        }
        // The file path may or may not exist on disk depending on
        // RocksDB's flushing behavior. The contract is only that
        // we don't crash; assertions about disk state would be
        // implementation-coupled.
    }
}
