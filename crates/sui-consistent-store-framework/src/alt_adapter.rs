// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`AltAdapter`] — the glue that lets a
//! [`sui_consistent_store::Pipeline`] plug into the indexer-alt
//! framework as a [`Processor`] + [`sequential::Handler`].
//!
//! The framework drives a pipeline through three calls in
//! sequence:
//!
//! 1. [`Processor::process`] takes a fetched
//!    [`Checkpoint`](sui_indexer_alt_framework::types::full_checkpoint_content::Checkpoint),
//!    converts it to [`CheckpointData`] (the form the consumer
//!    pipeline takes), and forwards to
//!    [`Pipeline::process`](sui_consistent_store::Pipeline::process).
//! 2. [`sequential::Handler::batch`] folds the resulting
//!    [`Pipeline::Value`] entries into the pipeline's typed
//!    accumulator across one or more checkpoints.
//! 3. [`sequential::Handler::commit`] drains the accumulator into
//!    the [`Connection`](crate::Connection)'s pending [`Batch`] via
//!    [`Pipeline::commit`](sui_consistent_store::Pipeline::commit).
//!    The framework's `SequentialStore::transaction` then writes
//!    the watermark and commits atomically — both the pipeline's
//!    data and the watermark advance land together.
//!
//! The adapter holds the pipeline by value and is constructed via
//! [`AltAdapter::new`]. It is `Send + Sync + 'static`, which the
//! framework requires.

use std::sync::Arc;

use async_trait::async_trait;
use sui_consistent_store::Pipeline;
use sui_indexer_alt_framework::pipeline::Processor;
use sui_indexer_alt_framework::pipeline::sequential;
use sui_indexer_alt_framework::types::full_checkpoint_content::Checkpoint;
use sui_indexer_alt_framework::types::full_checkpoint_content::CheckpointData;

use crate::Connection;
use crate::Store;

/// Wraps a [`Pipeline`] and supplies the
/// [`Processor`] + [`sequential::Handler`] impls the indexer-alt
/// framework's `sequential::pipeline` driver expects.
///
/// Construct via [`AltAdapter::new`]; pass the resulting value to
/// the framework's indexer registration API.
pub struct AltAdapter<P>(P);

impl<P> AltAdapter<P> {
    /// Wrap a pipeline.
    pub fn new(pipeline: P) -> Self {
        Self(pipeline)
    }

    /// Borrow the wrapped pipeline.
    pub fn pipeline(&self) -> &P {
        &self.0
    }
}

#[async_trait]
impl<P: Pipeline> Processor for AltAdapter<P> {
    const NAME: &'static str = P::NAME;

    type Value = P::Value;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        // The framework feeds us the newer `Checkpoint` type; the
        // consumer pipeline takes the canonical `CheckpointData`.
        // Convert via the `From` impl. The conversion materializes
        // per-tx input/output object lists from the effects, which
        // is the work the per-checkpoint cost is dominated by; it
        // is unavoidable when bridging the two type families.
        let cp_data: CheckpointData = (**checkpoint).clone().into();
        self.0.process(&cp_data)
    }
}

#[async_trait]
impl<P: Pipeline> sequential::Handler for AltAdapter<P> {
    type Store = Store<P::Schema>;
    type Batch = P::Batch;

    // Forward the pipeline's tuning constants to the framework's
    // collector / committer. The pipeline's
    // `MAX_BATCH_CHECKPOINTS` becomes the framework's; the other
    // two stay at the framework's defaults today since
    // `Pipeline` does not yet expose them.
    const MAX_BATCH_CHECKPOINTS: usize = P::MAX_BATCH_CHECKPOINTS;

    fn batch(&self, batch: &mut Self::Batch, values: std::vec::IntoIter<Self::Value>) {
        self.0.batch(batch, values);
    }

    async fn commit<'a>(
        &self,
        batch: &Self::Batch,
        conn: &mut Connection<'a, P::Schema>,
    ) -> anyhow::Result<usize> {
        // The pipeline writes through `&mut Batch`; our
        // `Connection` exposes the pending batch as a public field
        // so we can hand it to the pipeline without going through
        // `&conn` (which would conflict with `conn.store`'s
        // immutable borrow).
        self.0.commit(conn.store.schema(), batch, &mut conn.batch)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Buf;
    use bytes::BufMut;
    use scoped_futures::ScopedFutureExt;
    use sui_consistent_store::Batch;
    use sui_consistent_store::Db;
    use sui_consistent_store::DbMap;
    use sui_consistent_store::DbOptions;
    use sui_consistent_store::Decode;
    use sui_consistent_store::Encode;
    use sui_consistent_store::Schema;
    use sui_consistent_store::error::DecodeError;
    use sui_consistent_store::error::EncodeError;
    use sui_consistent_store::error::OpenError;
    use sui_consistent_store::rocksdb;
    use sui_indexer_alt_framework::types::base_types::ObjectID;
    use sui_indexer_alt_framework::types::full_checkpoint_content::Checkpoint as FwCheckpoint;
    use sui_indexer_alt_framework::types::object::Object;
    use sui_indexer_alt_framework::types::test_checkpoint_data_builder::TestCheckpointBuilder;
    use sui_indexer_alt_framework_store_traits::CommitterWatermark;
    use sui_indexer_alt_framework_store_traits::Connection as _;
    use sui_indexer_alt_framework_store_traits::SequentialStore;
    use sui_indexer_alt_framework_store_traits::Store as _;
    use tempfile::TempDir;

    use super::*;
    use crate::FrameworkSchema;

    // --- Pipeline scaffolding ---

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
                return Err(DecodeError::msg("unexpected length"));
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
        fn cfs(base_options: &rocksdb::Options) -> Vec<sui_consistent_store::CfDescriptor> {
            vec![sui_consistent_store::CfDescriptor::new(
                "versions",
                base_options.clone(),
            )]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

    #[derive(Debug)]
    struct Combined {
        framework: FrameworkSchema,
        user: VersionsSchema,
    }

    impl Schema for Combined {
        fn cfs(base_options: &rocksdb::Options) -> Vec<sui_consistent_store::CfDescriptor> {
            let mut cfs = FrameworkSchema::cfs(base_options);
            cfs.extend(VersionsSchema::cfs(base_options));
            cfs
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                framework: FrameworkSchema::open(db)?,
                user: VersionsSchema::open(db)?,
            })
        }
    }

    /// Test pipeline: tracks the highest version seen per object.
    struct VersionsPipeline;

    impl Pipeline for VersionsPipeline {
        const NAME: &'static str = "versions";
        type Schema = VersionsSchema;
        type Value = (ObjectID, u64);
        type Batch = BTreeMap<ObjectID, u64>;

        fn restore(&self, _: &mut Self::Batch, _: &Object) -> anyhow::Result<()> {
            Ok(())
        }

        fn process(&self, checkpoint: &CheckpointData) -> anyhow::Result<Vec<Self::Value>> {
            let mut out = vec![];
            for tx in &checkpoint.transactions {
                for o in &tx.output_objects {
                    out.push((o.id(), o.version().value()));
                }
            }
            Ok(out)
        }

        fn batch(&self, acc: &mut Self::Batch, values: std::vec::IntoIter<Self::Value>) {
            for (id, v) in values {
                acc.entry(id)
                    .and_modify(|hi| {
                        if v > *hi {
                            *hi = v;
                        }
                    })
                    .or_insert(v);
            }
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

    fn setup() -> (TempDir, Store<VersionsSchema>) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<Combined>(dir.path(), DbOptions::default()).unwrap();
        let store = Store::new(db, Arc::new(schema.framework), Arc::new(schema.user));
        (dir, store)
    }

    fn build_checkpoint(seq: u64) -> Arc<FwCheckpoint> {
        Arc::new(
            TestCheckpointBuilder::new(seq)
                .start_transaction(0)
                .create_owned_object(1)
                .finish_transaction()
                .build_checkpoint(),
        )
    }

    /// Compile-time check that the adapter satisfies the
    /// `sequential::Handler` bound the framework requires. (That
    /// bound transitively implies `Processor`, so checking the
    /// handler alone is sufficient.)
    fn _assert_traits<P: Pipeline>(adapter: AltAdapter<P>) -> impl sequential::Handler {
        adapter
    }

    #[tokio::test]
    async fn name_is_propagated_from_pipeline() {
        // Compile-time: const propagation through the trait.
        assert_eq!(
            <AltAdapter<VersionsPipeline> as Processor>::NAME,
            "versions"
        );
    }

    #[tokio::test]
    async fn process_converts_checkpoint_and_forwards() {
        let adapter = AltAdapter::new(VersionsPipeline);
        let cp = build_checkpoint(1);
        let values = adapter.process(&cp).await.unwrap();
        // At least one output object (the newly created one;
        // builders typically also emit a gas object).
        assert!(!values.is_empty());
    }

    #[tokio::test]
    async fn batch_folds_via_pipeline() {
        let adapter = AltAdapter::new(VersionsPipeline);
        let mut acc = BTreeMap::new();
        let id = ObjectID::from_single_byte(7);
        sequential::Handler::batch(
            &adapter,
            &mut acc,
            vec![(id, 1), (id, 5), (id, 3)].into_iter(),
        );
        assert_eq!(acc.get(&id), Some(&5));
    }

    #[tokio::test]
    async fn handler_commit_writes_through_pipeline() {
        let (_dir, store) = setup();
        let adapter = AltAdapter::new(VersionsPipeline);

        // Build an accumulator and commit through a transaction —
        // exercises the framework's `SequentialStore::transaction`
        // plus the `Handler::commit` wiring.
        let id = ObjectID::from_single_byte(3);
        let mut acc: BTreeMap<ObjectID, u64> = BTreeMap::new();
        acc.insert(id, 42u64);

        store
            .transaction(move |c| {
                async move {
                    sequential::Handler::commit(&adapter, &acc, c).await?;
                    c.set_committer_watermark(
                        VersionsPipeline::NAME,
                        CommitterWatermark::new_for_testing(1),
                    )
                    .await?;
                    Ok::<(), anyhow::Error>(())
                }
                .scope_boxed()
            })
            .await
            .unwrap();

        // Pipeline's CF reflects the write.
        assert_eq!(
            store.schema().versions.get(&ObjectIdKey::new(id)).unwrap(),
            Some(U64Be(42)),
        );
        // Watermark advanced.
        let mut conn = store.connect().await.unwrap();
        let w = conn
            .committer_watermark(VersionsPipeline::NAME)
            .await
            .unwrap();
        assert_eq!(w.unwrap().checkpoint_hi_inclusive, 1);
    }

    #[tokio::test]
    async fn end_to_end_process_batch_commit_through_transaction() {
        // Exercises the full Processor → Handler pipeline that the
        // framework would drive: process the framework's checkpoint
        // type, fold values into the accumulator, commit through a
        // transaction, observe the data in the pipeline's CF.
        let (_dir, store) = setup();
        let adapter = AltAdapter::new(VersionsPipeline);
        let cp = build_checkpoint(7);

        let values = adapter.process(&cp).await.unwrap();
        let mut acc = BTreeMap::new();
        sequential::Handler::batch(&adapter, &mut acc, values.into_iter());
        assert!(!acc.is_empty());

        store
            .transaction(move |c| {
                async move {
                    sequential::Handler::commit(&adapter, &acc, c).await?;
                    c.set_committer_watermark(
                        VersionsPipeline::NAME,
                        CommitterWatermark::new_for_testing(7),
                    )
                    .await?;
                    Ok::<(), anyhow::Error>(())
                }
                .scope_boxed()
            })
            .await
            .unwrap();

        // Every output object in the checkpoint has a CF row.
        let rows = store.schema().versions.iter(..).unwrap().count();
        assert!(rows > 0);
    }
}
