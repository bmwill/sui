// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format parsers for the formal-snapshot file layout.
//!
//! A formal snapshot is a directory tree served by an object store
//! ([`object_store`] backends or HTTP) with this shape:
//!
//! ```text
//! MANIFEST                  -- JSON root manifest listing available epochs
//! epoch_<E>/
//!   _SUCCESS                -- presence marker (zero bytes) indicating
//!                              the epoch's snapshot is complete
//!   MANIFEST                -- binary epoch manifest enumerating files
//!   <bucket>_<partition>.obj  -- live-objects file (per partition)
//!   <bucket>_<partition>.ref  -- reference (digest) file (per partition)
//! ```
//!
//! The on-disk format here is deliberately compatible with the
//! existing `sui-indexer-alt-consistent-store` layout: snapshots
//! produced for that crate are readable by this one and vice versa.
//!
//! # Parsers
//!
//! - [`RootManifest::read`] parses the top-level `MANIFEST` as
//!   JSON.
//! - [`EpochManifest::read`] parses the per-epoch `MANIFEST` as a
//!   `[u32 magic][BCS body][Sha3_256 digest]` blob.
//! - [`LiveObjectsFile::read`] parses an `<bucket>_<partition>.obj`
//!   file, applying decompression and yielding [`Object`]s.
//!
//! All parsers are pure (no I/O); higher layers fetch bytes via
//! [`Storage`](crate::storage::Storage) and pass them in.

use std::io::Cursor;
use std::io::Read;
use std::io::Seek as _;
use std::io::SeekFrom;

use anyhow::Context as _;
use anyhow::ensure;
use fastcrypto::hash::HashFunction;
use fastcrypto::hash::Sha3_256;
use integer_encoding::VarIntReader as _;
use serde::Deserialize;
use serde::Serialize;
use sui_storage::blob::Blob;
use sui_storage::blob::BlobEncoding;
use sui_types::base_types::ObjectID;
use sui_types::base_types::SequenceNumber;
use sui_types::object::Object;
use zstd::stream::read::Decoder;

/// Magic number at the start of the epoch manifest. Identifies the
/// file as an epoch manifest before BCS decoding is attempted.
const EPOCH_MANIFEST_MAGIC: u32 = 0x00C0FFEE;

/// Magic number at the start of each live-objects file. Identifies
/// the file as the expected format before varint-record decoding
/// begins.
const OBJECT_FILE_MAGIC: u32 = 0x00B7EC75;

/// Length in bytes of the SHA3-256 digest appended to the epoch
/// manifest for integrity validation.
const DIGEST_LEN: usize = Sha3_256::OUTPUT_SIZE;

/// JSON root manifest at the top of the snapshot store, listing
/// every epoch for which a complete snapshot is available.
#[derive(Debug, Deserialize)]
pub struct RootManifest {
    available_epochs: Vec<u64>,
}

/// Per-epoch manifest, enumerating every file that makes up the
/// epoch's snapshot.
///
/// The serialized representation is `[u32 magic][BCS body][Sha3_256
/// digest]`; the digest covers the magic-plus-BCS-body prefix and
/// is verified by [`read`](Self::read). `Serialize` is derived to
/// support producing test fixtures via `bcs::to_bytes`; production
/// consumers only need the deserialize path.
#[derive(Debug, Deserialize, Serialize)]
pub enum EpochManifest {
    /// Version 1 of the epoch manifest. The discriminant is encoded
    /// in the BCS tagged-union framing.
    V1(EpochManifestV1),
}

/// Body of a version-1 [`EpochManifest`].
#[derive(Debug, Deserialize, Serialize)]
pub struct EpochManifestV1 {
    /// Format version sentinel. Kept for forward compatibility but
    /// not validated against `EpochManifest::V1`'s implicit tag.
    pub version: u8,
    /// Length of an `ObjectID` in bytes. Informational; not used
    /// by the parsers in this crate.
    pub address_length: u64,
    /// One entry per file in the epoch's snapshot directory.
    pub metadata: Vec<FileMetadata>,
    /// The epoch number this manifest is for.
    pub epoch: u64,
}

/// Metadata for a single file inside an epoch's snapshot.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FileMetadata {
    pub file_type: FileType,
    /// Producer-side bucket index. The pair `(bucket, partition)`
    /// uniquely identifies a file inside an epoch.
    pub bucket: u32,
    /// Producer-side partition index within `bucket`.
    pub partition: u32,
    pub compression: FileCompression,
    /// SHA3-256 of the file body, used by the producer to detect
    /// corruption. This crate does not currently re-verify the
    /// digest at read time — the consuming protocol (HTTPS / object
    /// stores) handles integrity in transit.
    pub digest: [u8; DIGEST_LEN],
}

/// Kind of file inside the snapshot.
#[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum FileType {
    /// `.obj` file — a varint-framed stream of `LiveObject`s.
    Object = 0,
    /// `.ref` file — a stream of `(ObjectID, SequenceNumber, digest)`
    /// references. Not consumed by the restore path; present for
    /// the producer's other use cases.
    Reference,
}

/// Compression applied to a file body.
#[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum FileCompression {
    /// Bytes are stored as-is.
    None = 0,
    /// Bytes are zstd-compressed.
    Zstd,
}

/// A live-objects (`.obj`) file's contents after parsing: the
/// `(bucket, partition)` from its [`FileMetadata`] plus the
/// decoded [`Object`]s.
///
/// Wrapped objects (the `LiveObject::Wrapped` variant in the wire
/// format) are dropped at parse time; only the `Normal` variant is
/// retained.
#[derive(Debug)]
pub struct LiveObjectsFile {
    pub bucket: u32,
    pub partition: u32,
    pub objects: Vec<Object>,
}

/// The wire-level discriminated union for one entry in a `.obj`
/// file. The reference variant carries only the identity of a
/// wrapped object; restore does not need its content.
#[derive(Deserialize, Debug)]
enum LiveObject {
    Normal(Object),
    Wrapped(ObjectKey),
}

/// Identifier for a wrapped object inside an `.obj` file. The
/// fields are deserialized to consume the wire-format bytes; the
/// values themselves are used only by trace-level logging in
/// [`LiveObjectsFile::read`].
#[derive(Deserialize, Debug)]
struct ObjectKey(ObjectID, SequenceNumber);

impl RootManifest {
    /// Parse a JSON-encoded root manifest.
    pub fn read(data: &[u8]) -> anyhow::Result<Self> {
        serde_json::from_slice(data).context("Failed to parse root manifest")
    }

    /// The highest epoch number listed in the manifest, or `None`
    /// if the manifest is empty.
    pub fn latest(&self) -> Option<u64> {
        self.available_epochs.iter().copied().max()
    }

    /// Whether the manifest advertises a snapshot for `epoch`.
    pub fn contains(&self, epoch: u64) -> bool {
        self.available_epochs.contains(&epoch)
    }
}

impl EpochManifest {
    /// Parse and validate an epoch-manifest blob.
    ///
    /// Verifies the leading magic number, the BCS-encoded body, and
    /// the trailing SHA3-256 digest. The digest covers the magic
    /// plus the BCS body (everything except the digest itself).
    pub fn read(data: &[u8]) -> anyhow::Result<Self> {
        const MAGIC_LEN: usize = size_of_val(&EPOCH_MANIFEST_MAGIC);

        ensure!(
            data.len() >= MAGIC_LEN + DIGEST_LEN,
            "Epoch manifest too short",
        );

        let mut cursor = Cursor::new(data);

        // Check magic number.
        let mut magic = [0u8; MAGIC_LEN];
        cursor.read_exact(&mut magic)?;
        ensure!(
            u32::from_be_bytes(magic) == EPOCH_MANIFEST_MAGIC,
            "Not an epoch manifest",
        );

        // Read and verify the trailing digest.
        cursor.seek(SeekFrom::End(-(DIGEST_LEN as i64)))?;
        let end = cursor.position() as usize;
        let mut digest = [0u8; DIGEST_LEN];
        cursor.read_exact(&mut digest)?;

        let mut hasher = Sha3_256::new();
        hasher.update(&data[..end]);
        ensure!(
            hasher.finalize().digest == digest,
            "Epoch manifest digest mismatch",
        );

        // Deserialize the BCS body.
        bcs::from_bytes(&data[MAGIC_LEN..end]).context("Failed to deserialize epoch manifest")
    }

    /// The file list this manifest carries.
    pub fn metadata(&self) -> &[FileMetadata] {
        match self {
            EpochManifest::V1(m) => &m.metadata,
        }
    }

    /// The epoch number this manifest is for.
    pub fn epoch(&self) -> u64 {
        match self {
            EpochManifest::V1(m) => m.epoch,
        }
    }
}

impl FileCompression {
    /// A `Read` adapter that delivers the file's logical bytes,
    /// transparently decompressing if necessary.
    fn reader<'a>(self, data: &'a [u8]) -> anyhow::Result<Box<dyn Read + 'a>> {
        Ok(match self {
            FileCompression::None => Box::new(Cursor::new(data)),
            FileCompression::Zstd => Box::new(Decoder::new(Cursor::new(data))?),
        })
    }
}

impl LiveObjectsFile {
    /// Parse a `.obj` file body, applying the metadata's
    /// compression and dropping wrapped-object entries.
    ///
    /// The producer writes records as
    /// `[varint length][u8 encoding][length bytes]` triples
    /// terminated by a zero-length record (or EOF). Each record
    /// decodes via [`sui_storage::blob::Blob`] into a `LiveObject`.
    pub fn read(bytes: &[u8], metadata: &FileMetadata) -> anyhow::Result<Self> {
        const MAGIC_LEN: usize = size_of_val(&OBJECT_FILE_MAGIC);

        let mut read = metadata.compression.reader(bytes)?;

        let mut magic = [0u8; MAGIC_LEN];
        read.read_exact(&mut magic)?;
        ensure!(
            u32::from_be_bytes(magic) == OBJECT_FILE_MAGIC,
            "Not an object file",
        );

        let mut objects = vec![];
        while let Ok(len) = read.read_varint::<u64>() {
            if len == 0 {
                break;
            }

            let mut encoding_byte = [0u8; 1];
            read.read_exact(&mut encoding_byte)?;
            let encoding = BlobEncoding::try_from(encoding_byte[0]).with_context(|| {
                format!("Invalid encoding in object file: {}", encoding_byte[0])
            })?;

            let mut data = vec![0u8; len as usize];
            read.read_exact(&mut data)?;

            let object = Blob { data, encoding }
                .decode()
                .context("Failed to decode object from blob")?;

            match object {
                LiveObject::Normal(o) => objects.push(o),
                LiveObject::Wrapped(ObjectKey(id, version)) => {
                    // Restore does not need wrapped objects (only
                    // a reference back to where the wrapper lives,
                    // which is the wrapper's own entry); the
                    // pattern bind keeps the producer-side fields
                    // referenced for the dead-code linter.
                    tracing::trace!(
                        object_id = %id,
                        version = ?version,
                        "skipping wrapped object",
                    );
                }
            }
        }

        Ok(Self {
            bucket: metadata.bucket,
            partition: metadata.partition,
            objects,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_epoch_manifest_blob(epoch: u64) -> Vec<u8> {
        let manifest = EpochManifest::V1(EpochManifestV1 {
            version: 1,
            address_length: ObjectID::LENGTH as u64,
            metadata: vec![FileMetadata {
                file_type: FileType::Object,
                bucket: 0,
                partition: 1,
                compression: FileCompression::None,
                digest: [0u8; DIGEST_LEN],
            }],
            epoch,
        });

        let mut out = Vec::new();
        out.extend_from_slice(&EPOCH_MANIFEST_MAGIC.to_be_bytes());
        let body = bcs::to_bytes(&manifest).unwrap();
        out.extend_from_slice(&body);
        let mut hasher = Sha3_256::new();
        hasher.update(&out);
        out.extend_from_slice(&hasher.finalize().digest);
        out
    }

    #[test]
    fn root_manifest_round_trip() {
        let json = r#"{"available_epochs": [1, 7, 4]}"#;
        let m = RootManifest::read(json.as_bytes()).unwrap();
        assert_eq!(m.latest(), Some(7));
        assert!(m.contains(4));
        assert!(!m.contains(2));
    }

    #[test]
    fn root_manifest_empty_returns_no_latest() {
        let json = r#"{"available_epochs": []}"#;
        let m = RootManifest::read(json.as_bytes()).unwrap();
        assert_eq!(m.latest(), None);
        assert!(!m.contains(0));
    }

    #[test]
    fn root_manifest_invalid_json_errors() {
        let err = RootManifest::read(b"not json").unwrap_err();
        assert!(format!("{err:#}").contains("root manifest"));
    }

    #[test]
    fn epoch_manifest_round_trip() {
        let blob = build_epoch_manifest_blob(42);
        let m = EpochManifest::read(&blob).unwrap();
        assert_eq!(m.epoch(), 42);
        assert_eq!(m.metadata().len(), 1);
        assert_eq!(m.metadata()[0].bucket, 0);
        assert_eq!(m.metadata()[0].partition, 1);
        assert_eq!(m.metadata()[0].file_type, FileType::Object);
    }

    #[test]
    fn epoch_manifest_rejects_short_input() {
        let err = EpochManifest::read(&[0u8; 4]).unwrap_err();
        assert!(format!("{err:#}").contains("too short"));
    }

    #[test]
    fn epoch_manifest_rejects_bad_magic() {
        let mut blob = build_epoch_manifest_blob(1);
        // Overwrite the magic number.
        blob[0..4].copy_from_slice(&0u32.to_be_bytes());
        // Re-digest so we don't fail the digest check first.
        let end = blob.len() - DIGEST_LEN;
        let mut hasher = Sha3_256::new();
        hasher.update(&blob[..end]);
        blob[end..].copy_from_slice(&hasher.finalize().digest);
        let err = EpochManifest::read(&blob).unwrap_err();
        assert!(format!("{err:#}").contains("Not an epoch manifest"));
    }

    #[test]
    fn epoch_manifest_rejects_digest_mismatch() {
        let mut blob = build_epoch_manifest_blob(1);
        // Corrupt the digest.
        let last = blob.len() - 1;
        blob[last] ^= 0xFF;
        let err = EpochManifest::read(&blob).unwrap_err();
        assert!(format!("{err:#}").contains("digest mismatch"));
    }

    #[test]
    fn file_compression_none_reader_passes_bytes_through() {
        let bytes = b"hello";
        let mut out = Vec::new();
        FileCompression::None
            .reader(bytes)
            .unwrap()
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, bytes);
    }

    #[test]
    fn file_compression_zstd_reader_decompresses() {
        let original = b"compress me, compress me, compress me";
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut out = Vec::new();
        FileCompression::Zstd
            .reader(&compressed)
            .unwrap()
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn live_objects_file_empty_is_legal() {
        // Magic plus a single zero-varint terminator yields an empty
        // objects list.
        let mut buf = Vec::new();
        buf.extend_from_slice(&OBJECT_FILE_MAGIC.to_be_bytes());
        // varint 0 is one byte.
        buf.push(0);
        let metadata = FileMetadata {
            file_type: FileType::Object,
            bucket: 3,
            partition: 7,
            compression: FileCompression::None,
            digest: [0u8; DIGEST_LEN],
        };
        let parsed = LiveObjectsFile::read(&buf, &metadata).unwrap();
        assert_eq!(parsed.bucket, 3);
        assert_eq!(parsed.partition, 7);
        assert!(parsed.objects.is_empty());
    }

    #[test]
    fn live_objects_file_rejects_bad_magic() {
        let buf = [0u8; 4];
        let metadata = FileMetadata {
            file_type: FileType::Object,
            bucket: 0,
            partition: 0,
            compression: FileCompression::None,
            digest: [0u8; DIGEST_LEN],
        };
        let err = LiveObjectsFile::read(&buf, &metadata).unwrap_err();
        assert!(format!("{err:#}").contains("Not an object file"));
    }
}
