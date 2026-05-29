// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`FormalSnapshot`] — an opened handle on a formal snapshot
//! served by a [`Storage`] backend.
//!
//! Use [`FormalSnapshot::open`] to connect to a snapshot source,
//! validate that the requested (or latest) epoch is present, and
//! pre-fetch the epoch manifest. The returned handle then exposes:
//!
//! - [`epoch`](FormalSnapshot::epoch): the epoch this handle is
//!   anchored at.
//! - [`partition_metadata`](FormalSnapshot::partition_metadata):
//!   iterator of [`FileMetadata`] entries for each
//!   `.obj` (live-objects) file the snapshot contains.
//! - [`partition_id`](FormalSnapshot::partition_id): canonical
//!   8-byte encoding `(bucket, partition)` used as the opaque
//!   partition identifier in [`RestoreState::InProgress`] entries.
//! - [`fetch_partition`](FormalSnapshot::fetch_partition):
//!   downloads and parses a single partition's `.obj` file into a
//!   [`LiveObjectsFile`].
//!
//! Higher layers can iterate `partition_metadata`, fetch each in
//! whatever parallelism strategy fits their environment, and feed
//! the resulting objects to a [`RestoreRunner`](crate::RestoreRunner).

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::ensure;
use bytes::Bytes;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use object_store::path::Path;
use sui_consistent_store::Restore;
use tracing::info;

use crate::RestoreRunner;
use crate::snapshot_format::EpochManifest;
use crate::snapshot_format::FileMetadata;
use crate::snapshot_format::FileType;
use crate::snapshot_format::LiveObjectsFile;
use crate::snapshot_format::RootManifest;
use crate::storage::Storage;

/// An opened handle on a formal snapshot at a specific epoch.
///
/// Cloning is cheap: the storage backend lives behind an
/// [`Arc<dyn Storage + …>`] and the pre-fetched epoch manifest is
/// itself shared via the inner [`Arc`].
pub struct FormalSnapshot {
    /// The source of files related to the formal snapshot.
    source: Arc<dyn Storage + Send + Sync + 'static>,

    /// The epoch this snapshot is anchored at.
    epoch: u64,

    /// Pre-fetched per-epoch manifest. Cached at open time so
    /// every partition fetch does not re-download it.
    manifest: Arc<EpochManifest>,
}

impl Clone for FormalSnapshot {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            epoch: self.epoch,
            manifest: self.manifest.clone(),
        }
    }
}

impl std::fmt::Debug for FormalSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FormalSnapshot")
            .field("epoch", &self.epoch)
            .field("partitions", &self.manifest.metadata().len())
            .finish_non_exhaustive()
    }
}

impl FormalSnapshot {
    /// Open a snapshot at `epoch` (or the latest available epoch
    /// if `epoch` is `None`).
    ///
    /// Validates that:
    ///
    /// 1. The root `MANIFEST` lists the requested epoch.
    /// 2. The `epoch_<E>/_SUCCESS` marker file is present (the
    ///    producer writes it last, after every per-partition file
    ///    has been finalized).
    /// 3. The per-epoch `MANIFEST` parses and its digest matches.
    ///
    /// On success the per-epoch manifest is held in memory; later
    /// `fetch_partition` calls do not re-download it.
    pub async fn open(
        source: Arc<dyn Storage + Send + Sync + 'static>,
        epoch: Option<u64>,
    ) -> anyhow::Result<Self> {
        // 1. Read the root manifest.
        let root_bytes = source
            .get(Path::from("MANIFEST"))
            .await
            .context("Failed to fetch root manifest")?;
        let root = RootManifest::read(&root_bytes)?;

        // 2. Pick an epoch (caller-specified or latest available).
        let epoch = epoch
            .or_else(|| root.latest())
            .context("No epochs available in the snapshot store")?;
        ensure!(
            root.contains(epoch),
            "Requested epoch {epoch} is not available in the snapshot store",
        );

        // 3. Confirm the epoch's snapshot is complete.
        source
            .get(Path::from(format!("epoch_{epoch}/_SUCCESS")))
            .await
            .with_context(|| format!("Snapshot for epoch {epoch} is not marked complete"))?;

        // 4. Fetch and validate the per-epoch manifest.
        let manifest_bytes = source
            .get(Path::from(format!("epoch_{epoch}/MANIFEST")))
            .await
            .context("Failed to fetch epoch manifest")?;
        let manifest = EpochManifest::read(&manifest_bytes)?;

        info!(
            epoch,
            partitions = manifest.metadata().len(),
            "Connected to valid formal snapshot",
        );

        Ok(Self {
            source,
            epoch,
            manifest: Arc::new(manifest),
        })
    }

    /// The epoch this snapshot is anchored at.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Iterator over `(bucket, partition)` file metadata for every
    /// `.obj` (live-objects) entry in the snapshot.
    ///
    /// Reference (`.ref`) files are filtered out — they are not
    /// consumed by the restore path. The order follows the
    /// per-epoch manifest's order, which is the order the producer
    /// wrote files; consumers should not rely on a particular sort.
    pub fn partition_metadata(&self) -> impl Iterator<Item = &FileMetadata> + '_ {
        self.manifest
            .metadata()
            .iter()
            .filter(|m| matches!(m.file_type, FileType::Object))
    }

    /// Canonical 8-byte encoding of a partition's identifier,
    /// suitable as the opaque partition-id bytes used by
    /// [`RestoreRunner`] and persisted in the `__restore` CF's
    /// [`RestoreState::InProgress`](sui_consistent_store::RestoreState::InProgress)
    /// entry.
    ///
    /// Layout: `[bucket u32 big-endian][partition u32 big-endian]`.
    pub fn partition_id(metadata: &FileMetadata) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&metadata.bucket.to_be_bytes());
        out[4..].copy_from_slice(&metadata.partition.to_be_bytes());
        out
    }

    /// Fetch a single partition's `.obj` file and parse it into a
    /// [`LiveObjectsFile`].
    ///
    /// Path layout: `epoch_<E>/<bucket>_<partition>.obj`.
    pub async fn fetch_partition(
        &self,
        metadata: &FileMetadata,
    ) -> anyhow::Result<LiveObjectsFile> {
        let bytes = self.fetch_file(metadata).await?;
        LiveObjectsFile::read(&bytes, metadata).with_context(|| {
            format!(
                "Failed to parse partition {}_{}",
                metadata.bucket, metadata.partition
            )
        })
    }

    /// Fetch the raw bytes of a `.obj` or `.ref` file described by
    /// `metadata`. Exposed for callers that want to drive their own
    /// parse loop (or for tests that just want to inspect bytes).
    pub async fn fetch_file(&self, metadata: &FileMetadata) -> anyhow::Result<Bytes> {
        let ext = match metadata.file_type {
            FileType::Object => "obj",
            FileType::Reference => "ref",
        };
        let path = format!(
            "epoch_{}/{}_{}.{ext}",
            self.epoch, metadata.bucket, metadata.partition,
        );
        self.source
            .get(Path::from(path.clone()))
            .await
            .with_context(|| format!("Failed to fetch file: {path}"))
    }
}

/// Drive a single-pipeline restore from `snapshot` through
/// `runner`.
///
/// Calls [`runner.begin()`](RestoreRunner::begin) to recover any
/// previously-completed partitions, iterates the snapshot's
/// partition metadata, and for each partition not in the skip
/// set:
///
/// 1. Fetches the partition's `.obj` file via the snapshot's
///    [`Storage`] backend (concurrently up to
///    `partition_concurrency`).
/// 2. Hands the parsed objects to
///    [`runner.process_shard`](RestoreRunner::process_shard) in a
///    blocking task — `process_shard` is sync because the per-shard
///    work is CPU-bound (folding into the accumulator and encoding
///    into a [`Batch`](sui_consistent_store::Batch)), not I/O.
///
/// On success, calls [`runner.finish()`](RestoreRunner::finish) to
/// transition the pipeline to
/// [`RestoreState::Complete`](sui_consistent_store::RestoreState::Complete).
///
/// The runner is wrapped in [`Arc`] because the per-partition
/// futures spawn blocking tasks that may outlive the immediate
/// `await` frame.
pub async fn restore_pipeline_from_formal_snapshot<P: Restore>(
    runner: Arc<RestoreRunner<P>>,
    snapshot: Arc<FormalSnapshot>,
    partition_concurrency: usize,
) -> anyhow::Result<()> {
    let skip = runner.begin()?;

    let to_process: Vec<FileMetadata> = snapshot
        .partition_metadata()
        .filter(|&m| !skip.contains(FormalSnapshot::partition_id(m).as_slice()))
        .cloned()
        .collect();
    let total = to_process.len();

    info!(
        pipeline = P::NAME,
        partitions = total,
        skipped = skip.len(),
        "Restore plan",
    );

    futures::stream::iter(to_process.into_iter())
        .map(|meta| {
            let runner = runner.clone();
            let snapshot = snapshot.clone();
            async move {
                let partition_id = FormalSnapshot::partition_id(&meta);
                let parsed = snapshot.fetch_partition(&meta).await.with_context(|| {
                    format!(
                        "Failed to fetch partition {}_{}",
                        meta.bucket, meta.partition,
                    )
                })?;
                // `process_shard` is CPU-bound (folding + batch
                // encoding) and sync; run it on a blocking pool so
                // the async executor stays responsive.
                tokio::task::spawn_blocking(move || {
                    runner.process_shard(&partition_id, parsed.objects.into_iter().map(Ok))
                })
                .await
                .context("process_shard worker task panicked")??;
                anyhow::Ok(())
            }
        })
        .buffer_unordered(partition_concurrency)
        .try_collect::<Vec<_>>()
        .await?;

    runner.finish()?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_fixture {
    //! Helpers for building on-disk formal-snapshot fixtures in
    //! tests. Exposed at the crate level so the formal-snapshot
    //! driver's tests (added in a follow-up commit) can reuse them.

    use std::io::Write as _;
    use std::path::Path;

    use fastcrypto::hash::HashFunction;
    use fastcrypto::hash::Sha3_256;
    use sui_storage::blob::Blob;
    use sui_storage::blob::BlobEncoding;
    use sui_types::object::Object;

    use crate::snapshot_format::EpochManifest;
    use crate::snapshot_format::EpochManifestV1;
    use crate::snapshot_format::FileCompression;
    use crate::snapshot_format::FileMetadata;
    use crate::snapshot_format::FileType;

    const EPOCH_MANIFEST_MAGIC: u32 = 0x00C0FFEE;
    const OBJECT_FILE_MAGIC: u32 = 0x00B7EC75;
    const DIGEST_LEN: usize = Sha3_256::OUTPUT_SIZE;

    /// Describe one partition fixture: its `(bucket, partition)`
    /// pair and the objects it should contain.
    pub struct PartitionFixture {
        pub bucket: u32,
        pub partition: u32,
        pub objects: Vec<Object>,
    }

    /// Build a snapshot directory tree at `root` containing exactly
    /// `epoch`, with the supplied partitions.
    pub fn build_snapshot(root: &Path, epoch: u64, partitions: Vec<PartitionFixture>) {
        // Top-level MANIFEST listing available epochs.
        let root_manifest = format!(r#"{{"available_epochs": [{epoch}]}}"#);
        std::fs::write(root.join("MANIFEST"), root_manifest).unwrap();

        let epoch_dir = root.join(format!("epoch_{epoch}"));
        std::fs::create_dir_all(&epoch_dir).unwrap();

        // _SUCCESS marker (empty file).
        std::fs::write(epoch_dir.join("_SUCCESS"), b"").unwrap();

        // One .obj file per partition.
        let mut metadata = Vec::new();
        for p in &partitions {
            let obj_path = epoch_dir.join(format!("{}_{}.obj", p.bucket, p.partition));
            let body = write_object_file(&p.objects);
            std::fs::write(&obj_path, &body).unwrap();
            metadata.push(FileMetadata {
                file_type: FileType::Object,
                bucket: p.bucket,
                partition: p.partition,
                compression: FileCompression::None,
                digest: [0u8; DIGEST_LEN],
            });
        }

        // Per-epoch MANIFEST: magic + bcs(body) + digest.
        let manifest = EpochManifest::V1(EpochManifestV1 {
            version: 1,
            address_length: 32,
            metadata,
            epoch,
        });
        let mut out = Vec::new();
        out.extend_from_slice(&EPOCH_MANIFEST_MAGIC.to_be_bytes());
        out.extend_from_slice(&bcs::to_bytes(&manifest).unwrap());
        let mut hasher = Sha3_256::new();
        hasher.update(&out);
        out.extend_from_slice(&hasher.finalize().digest);
        std::fs::write(epoch_dir.join("MANIFEST"), &out).unwrap();
    }

    /// Serialize objects into a `.obj` file body. Uses the
    /// uncompressed (`FileCompression::None`) variant of the wire
    /// format. Each emitted record encodes a `Normal` `LiveObject`
    /// variant; wrapped objects are not produced by these fixtures.
    fn write_object_file(objects: &[Object]) -> Vec<u8> {
        // Local `LiveObject` mirroring the production wire-level
        // type, but only with the `Normal` variant — the BCS
        // discriminant for `Normal` is `0` in both, which is what
        // the reader expects.
        #[derive(serde::Serialize)]
        enum LiveObject<'a> {
            Normal(&'a Object),
        }

        let mut out = Vec::new();
        out.extend_from_slice(&OBJECT_FILE_MAGIC.to_be_bytes());

        // Each record is `varint(len) | u8(encoding) | bytes(len)`,
        // and a zero-length record terminates the stream.
        for o in objects {
            let live = LiveObject::Normal(o);
            let blob = Blob::encode(&live, BlobEncoding::Bcs).unwrap();
            integer_encoding::VarIntWriter::write_varint(&mut out, blob.data.len() as u64).unwrap();
            out.write_all(&[blob.encoding.into()]).unwrap();
            out.write_all(&blob.data).unwrap();
        }
        // Zero-length record terminator.
        integer_encoding::VarIntWriter::write_varint(&mut out, 0u64).unwrap();
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::local::LocalFileSystem;
    use sui_types::base_types::ObjectID;
    use sui_types::object::Object;
    use tempfile::TempDir;

    use super::test_fixture::PartitionFixture;
    use super::test_fixture::build_snapshot;
    use super::*;
    use crate::snapshot_format::FileCompression;

    fn obj(id: u8) -> Object {
        Object::immutable_with_id_for_testing(ObjectID::from_single_byte(id))
    }

    /// Open the snapshot fixture under `root` as a `FormalSnapshot`
    /// with the given epoch override.
    async fn open_local_snapshot(root: &std::path::Path, epoch: Option<u64>) -> FormalSnapshot {
        let store: Arc<dyn Storage + Send + Sync + 'static> =
            Arc::new(LocalFileSystem::new_with_prefix(root).unwrap());
        FormalSnapshot::open(store, epoch).await.unwrap()
    }

    #[tokio::test]
    async fn open_validates_epoch_and_returns_handle() {
        let dir = TempDir::new().unwrap();
        build_snapshot(
            dir.path(),
            42,
            vec![PartitionFixture {
                bucket: 0,
                partition: 0,
                objects: vec![obj(1)],
            }],
        );
        let snap = open_local_snapshot(dir.path(), Some(42)).await;
        assert_eq!(snap.epoch(), 42);
    }

    #[tokio::test]
    async fn open_defaults_to_latest_epoch_when_none_specified() {
        let dir = TempDir::new().unwrap();
        build_snapshot(
            dir.path(),
            123,
            vec![PartitionFixture {
                bucket: 0,
                partition: 0,
                objects: vec![],
            }],
        );
        let snap = open_local_snapshot(dir.path(), None).await;
        assert_eq!(snap.epoch(), 123);
    }

    #[tokio::test]
    async fn open_refuses_missing_epoch() {
        let dir = TempDir::new().unwrap();
        build_snapshot(dir.path(), 1, vec![]);
        let store: Arc<dyn Storage + Send + Sync + 'static> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let err = FormalSnapshot::open(store, Some(2)).await.unwrap_err();
        assert!(format!("{err:#}").contains("is not available"));
    }

    #[tokio::test]
    async fn open_refuses_incomplete_epoch_without_success_marker() {
        let dir = TempDir::new().unwrap();
        build_snapshot(dir.path(), 1, vec![]);
        // Delete the _SUCCESS marker.
        std::fs::remove_file(dir.path().join("epoch_1/_SUCCESS")).unwrap();
        let store: Arc<dyn Storage + Send + Sync + 'static> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let err = FormalSnapshot::open(store, Some(1)).await.unwrap_err();
        assert!(format!("{err:#}").contains("not marked complete"));
    }

    #[tokio::test]
    async fn partition_metadata_yields_only_object_files() {
        let dir = TempDir::new().unwrap();
        build_snapshot(
            dir.path(),
            5,
            vec![
                PartitionFixture {
                    bucket: 0,
                    partition: 0,
                    objects: vec![obj(1)],
                },
                PartitionFixture {
                    bucket: 0,
                    partition: 1,
                    objects: vec![obj(2)],
                },
            ],
        );
        let snap = open_local_snapshot(dir.path(), Some(5)).await;
        let count = snap.partition_metadata().count();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn fetch_partition_returns_objects_we_wrote() {
        let dir = TempDir::new().unwrap();
        let objects = vec![obj(1), obj(2), obj(3)];
        let expected_ids: Vec<ObjectID> = objects.iter().map(|o| o.id()).collect();
        build_snapshot(
            dir.path(),
            5,
            vec![PartitionFixture {
                bucket: 7,
                partition: 9,
                objects,
            }],
        );

        let snap = open_local_snapshot(dir.path(), Some(5)).await;
        let meta = snap.partition_metadata().next().unwrap().clone();
        assert_eq!(meta.bucket, 7);
        assert_eq!(meta.partition, 9);

        let parsed = snap.fetch_partition(&meta).await.unwrap();
        assert_eq!(parsed.bucket, 7);
        assert_eq!(parsed.partition, 9);
        let got_ids: Vec<ObjectID> = parsed.objects.iter().map(|o| o.id()).collect();
        assert_eq!(got_ids, expected_ids);
    }

    #[test]
    fn partition_id_is_canonical_8_bytes() {
        let meta = FileMetadata {
            file_type: FileType::Object,
            bucket: 0x0A0B0C0D,
            partition: 0x01020304,
            compression: FileCompression::None,
            digest: [0u8; 32],
        };
        let id = FormalSnapshot::partition_id(&meta);
        assert_eq!(id, [0x0A, 0x0B, 0x0C, 0x0D, 0x01, 0x02, 0x03, 0x04]);
    }

    /// End-to-end exercise of [`restore_pipeline_from_formal_snapshot`]:
    /// build an on-disk snapshot, open it, run a pipeline through
    /// the runner, then read back the pipeline's CF and confirm
    /// every object landed.
    mod driver {
        use std::collections::BTreeMap;
        use std::sync::Arc;

        use bytes::Buf;
        use bytes::BufMut;
        use object_store::local::LocalFileSystem;
        use sui_types::base_types::ObjectID;
        use sui_types::object::Object;
        use tempfile::TempDir;

        use super::super::*;
        use super::PartitionFixture;
        use super::build_snapshot;
        use super::obj;
        use sui_consistent_store::Batch;
        use sui_consistent_store::Db;
        use sui_consistent_store::DbMap;
        use sui_consistent_store::DbOptions;
        use sui_consistent_store::Decode;
        use sui_consistent_store::Encode;
        use sui_consistent_store::FrameworkSchema;
        use sui_consistent_store::PipelineTaskKey;
        use sui_consistent_store::RestoreState;
        use sui_consistent_store::Schema;
        use sui_consistent_store::error::DecodeError;
        use sui_consistent_store::error::EncodeError;
        use sui_consistent_store::error::OpenError;

        use crate::snapshot_format::FileCompression;

        /// Test helper: read the persisted `RestoreState` for
        /// `pipeline` via the auto-registered framework schema.
        fn read_restore_state(db: &Db, pipeline: &str) -> Option<RestoreState> {
            db.framework()
                .restore
                .get(&PipelineTaskKey::new(pipeline))
                .unwrap()
        }

        /// Test helper: write `state` for `pipeline` (commit
        /// immediately).
        fn write_restore_state(db: &Db, pipeline: &str, state: &RestoreState) {
            let fw = FrameworkSchema::new(db.clone());
            let mut batch = db.batch();
            batch
                .put(&fw.restore, &PipelineTaskKey::new(pipeline), state)
                .unwrap();
            batch.commit().unwrap();
        }

        /// Big-endian `ObjectID` key newtype.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct ObjectIdKey([u8; ObjectID::LENGTH]);

        impl ObjectIdKey {
            fn new(id: ObjectID) -> Self {
                Self(id.into_bytes())
            }
        }

        impl Encode for ObjectIdKey {
            fn encode_into<B: BufMut>(&self, buf: &mut B) -> Result<(), EncodeError> {
                buf.put_slice(&self.0);
                Ok(())
            }
        }

        impl Decode for ObjectIdKey {
            fn decode<B: Buf>(buf: &mut B) -> Result<Self, DecodeError> {
                if buf.remaining() != ObjectID::LENGTH {
                    return Err(DecodeError::msg("unexpected ObjectIdKey length"));
                }
                let mut id = [0u8; ObjectID::LENGTH];
                buf.copy_to_slice(&mut id);
                Ok(Self(id))
            }
        }

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

        #[derive(Debug)]
        struct VersionsSchema {
            versions: DbMap<ObjectIdKey, U64Be>,
        }

        impl Schema for VersionsSchema {
            fn cfs(
                base_options: &sui_consistent_store::rocksdb::Options,
            ) -> Vec<sui_consistent_store::CfDescriptor> {
                vec![sui_consistent_store::CfDescriptor::new(
                    "versions",
                    base_options.clone(),
                )]
            }

            fn open(db: &Db) -> Result<Self, OpenError> {
                Ok(Self {
                    versions: DbMap::new(db.clone(), "versions")?,
                })
            }
        }

        struct VersionsPipeline;

        impl Restore for VersionsPipeline {
            const NAME: &'static str = "versions";
            type Schema = VersionsSchema;
            type Batch = BTreeMap<ObjectID, u64>;

            fn restore(
                &self,
                accumulator: &mut Self::Batch,
                object: &Object,
            ) -> anyhow::Result<()> {
                accumulator
                    .entry(object.id())
                    .and_modify(|hi| {
                        if object.version().value() > *hi {
                            *hi = object.version().value();
                        }
                    })
                    .or_insert(object.version().value());
                Ok(())
            }

            fn commit(
                &self,
                schema: &Self::Schema,
                batch: &Self::Batch,
                write_batch: &mut Batch,
            ) -> anyhow::Result<usize> {
                for (id, v) in batch {
                    write_batch.put(&schema.versions, &ObjectIdKey::new(*id), &U64Be(*v))?;
                }
                Ok(batch.len())
            }
        }

        #[tokio::test]
        async fn restore_pipeline_end_to_end() {
            // Build a snapshot with two partitions covering objects
            // 1..=3 and 4..=6.
            let snapshot_dir = TempDir::new().unwrap();
            let db_dir = TempDir::new().unwrap();

            build_snapshot(
                snapshot_dir.path(),
                42,
                vec![
                    PartitionFixture {
                        bucket: 0,
                        partition: 0,
                        objects: vec![obj(1), obj(2), obj(3)],
                    },
                    PartitionFixture {
                        bucket: 0,
                        partition: 1,
                        objects: vec![obj(4), obj(5), obj(6)],
                    },
                ],
            );

            // Open the DB and the snapshot.
            let (db, schema) =
                Db::open::<VersionsSchema>(db_dir.path(), DbOptions::default()).unwrap();
            let schema = Arc::new(schema);
            let snapshot_store: Arc<dyn Storage + Send + Sync + 'static> =
                Arc::new(LocalFileSystem::new_with_prefix(snapshot_dir.path()).unwrap());
            let snapshot = Arc::new(
                FormalSnapshot::open(snapshot_store, Some(42))
                    .await
                    .unwrap(),
            );

            // Drive the restore.
            let runner = Arc::new(RestoreRunner::new(
                db.clone(),
                Arc::new(VersionsPipeline),
                schema.clone(),
                snapshot.epoch(),
            ));
            restore_pipeline_from_formal_snapshot(runner, snapshot, 2)
                .await
                .unwrap();

            // Every object the snapshot fixture wrote should be in
            // the pipeline's CF.
            for id in [1u8, 2, 3, 4, 5, 6] {
                let obj_id = ObjectID::from_single_byte(id);
                let val = schema.versions.get(&ObjectIdKey::new(obj_id)).unwrap();
                assert!(val.is_some(), "object {id} missing");
            }

            // Restore state is Complete at the snapshot's epoch.
            match read_restore_state(&db, "versions") {
                Some(RestoreState::Complete { restored_at }) => {
                    assert_eq!(restored_at, 42);
                }
                other => panic!("expected Complete, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn restore_skips_already_completed_partitions_on_resume() {
            // Build a snapshot, run it once, then "lose" one
            // partition's data by clearing the CF and rerunning —
            // the rerun should skip the partition that was marked
            // complete and write nothing for it.
            let snapshot_dir = TempDir::new().unwrap();
            let db_dir = TempDir::new().unwrap();

            build_snapshot(
                snapshot_dir.path(),
                7,
                vec![
                    PartitionFixture {
                        bucket: 0,
                        partition: 0,
                        objects: vec![obj(1)],
                    },
                    PartitionFixture {
                        bucket: 0,
                        partition: 1,
                        objects: vec![obj(2)],
                    },
                ],
            );

            let (db, schema) =
                Db::open::<VersionsSchema>(db_dir.path(), DbOptions::default()).unwrap();
            let schema = Arc::new(schema);
            let snapshot_store: Arc<dyn Storage + Send + Sync + 'static> =
                Arc::new(LocalFileSystem::new_with_prefix(snapshot_dir.path()).unwrap());

            // First run: only mark partition 0 as complete (simulate
            // a crash after partition 0 ingested but before
            // partition 1).
            let mut done = std::collections::BTreeSet::new();
            let p0 = FormalSnapshot::partition_id(&FileMetadata {
                file_type: FileType::Object,
                bucket: 0,
                partition: 0,
                compression: FileCompression::None,
                digest: [0u8; 32],
            });
            done.insert(p0.to_vec());
            write_restore_state(
                &db,
                "versions",
                &RestoreState::InProgress {
                    target_checkpoint: 7,
                    partitions_complete: done,
                },
            );

            // Resume the restore. Only partition 1 should be
            // fetched and ingested.
            let snapshot = Arc::new(FormalSnapshot::open(snapshot_store, Some(7)).await.unwrap());
            let runner = Arc::new(RestoreRunner::new(
                db.clone(),
                Arc::new(VersionsPipeline),
                schema.clone(),
                snapshot.epoch(),
            ));
            restore_pipeline_from_formal_snapshot(runner, snapshot, 1)
                .await
                .unwrap();

            // Object 1 is from partition 0 (skipped) — should NOT
            // appear in the CF.
            assert!(
                schema
                    .versions
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(1)))
                    .unwrap()
                    .is_none(),
            );
            // Object 2 is from partition 1 (processed) — should
            // appear.
            assert!(
                schema
                    .versions
                    .get(&ObjectIdKey::new(ObjectID::from_single_byte(2)))
                    .unwrap()
                    .is_some(),
            );
        }
    }
}
