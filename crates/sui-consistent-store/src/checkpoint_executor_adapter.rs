// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`CheckpointExecutorAdapter`] — the validator-side tip-mode
//! driver that bridges `sui-core`'s checkpoint executor to a set of
//! [`Pipeline`] implementations.
//!
//! The shape mirrors today's
//! [`sui_core::rpc_index::RpcIndexStore`]: a single object holds
//! every pipeline, accepts checkpoints out-of-order via
//! [`index_checkpoint`](CheckpointExecutorAdapter::index_checkpoint),
//! stages each per-checkpoint result in a `BTreeMap` keyed by
//! checkpoint sequence number, and commits them in order via
//! [`commit_update_for_checkpoint`](CheckpointExecutorAdapter::commit_update_for_checkpoint).
//! All pipelines plus a single
//! [`Batch::commit`](crate::Batch::commit) land atomically — the
//! validator's checkpoint executor advances its watermark only
//! after the commit returns successfully.
//!
//! # Pipeline heterogeneity
//!
//! Each [`Pipeline`] declares its own
//! [`Self::Schema`](crate::Pipeline::Schema),
//! [`Self::Value`](crate::Pipeline::Value), and
//! [`Self::Batch`](crate::Pipeline::Batch) associated types.
//! Storing many in one adapter requires type erasure: this module's
//! private [`TipPipeline`] trait fixes the surface (process a
//! checkpoint into an opaque accumulator, drain an opaque
//! accumulator into a write batch) and a generic
//! [`PipelineAdapter<P>`] supplies a blanket
//! [`TipPipeline`](TipPipeline) impl for any [`Pipeline`].
//! Internally, accumulators travel as `Box<dyn Any + Send>` and
//! down-cast back to the pipeline's typed `Self::Batch` at commit
//! time. The down-cast can only fail on a programmer error
//! (mis-wired pipeline order), and surfaces as a clear panic.
//!
//! # Construction
//!
//! Build via [`CheckpointExecutorAdapter::builder`], which returns
//! an [`AdapterBuilder`] that accepts pipelines through
//! [`add_pipeline`](AdapterBuilder::add_pipeline):
//!
//! ```ignore
//! let adapter = CheckpointExecutorAdapter::builder(db.clone())
//!     .add_pipeline(Arc::new(BalancesPipeline), Arc::new(schema.clone()))
//!     .add_pipeline(Arc::new(ObjectByOwnerPipeline), Arc::new(schema.clone()))
//!     .build();
//! ```
//!
//! The [`Db`] handle is `Clone`, so callers pass it by value
//! (`db.clone()`) rather than wrapping it in an outer `Arc`.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Context as _;
use parking_lot::Mutex;
use sui_types::full_checkpoint_content::Checkpoint;

use crate::Batch;
use crate::Db;
use crate::Pipeline;

/// Tip-mode adapter wrapping a fixed list of [`Pipeline`]s.
///
/// See the module documentation for the staging model. The
/// adapter is `Send + Sync` so the validator can share it across
/// the checkpoint executor's worker pool.
pub struct CheckpointExecutorAdapter {
    db: Db,
    pipelines: Vec<Arc<dyn TipPipeline>>,
    pending: Mutex<BTreeMap<u64, Vec<Box<dyn Any + Send>>>>,
}

/// Type-erased view of one [`Pipeline`] sufficient for the tip
/// path's needs.
///
/// Implementations must guarantee that the `Box<dyn Any>` returned
/// by [`process_into_batch`](TipPipeline::process_into_batch) is
/// the *same* concrete type as the one
/// [`commit_batch`](TipPipeline::commit_batch) downcasts to.
trait TipPipeline: Send + Sync + 'static {
    /// The pipeline's name, used for logs and error context.
    fn name(&self) -> &'static str;

    /// Process a checkpoint into the pipeline's typed accumulator,
    /// returned boxed as `dyn Any` for storage in the staging map.
    fn process_into_batch(
        &self,
        checkpoint: &Checkpoint,
    ) -> anyhow::Result<Box<dyn Any + Send>>;

    /// Drain the staged accumulator (originally returned by
    /// [`process_into_batch`](Self::process_into_batch)) into
    /// `write_batch`. Returns the row count for metrics.
    fn commit_batch(
        &self,
        batch: Box<dyn Any + Send>,
        write_batch: &mut Batch,
    ) -> anyhow::Result<usize>;
}

/// Blanket [`TipPipeline`] adapter for any concrete [`Pipeline`].
///
/// Holds the pipeline impl and an [`Arc`] of its schema so commits
/// have everything they need without taking borrowed references
/// across the type-erasure boundary.
struct PipelineAdapter<P: Pipeline> {
    pipeline: Arc<P>,
    schema: Arc<P::Schema>,
}

impl<P: Pipeline> TipPipeline for PipelineAdapter<P> {
    fn name(&self) -> &'static str {
        P::NAME
    }

    fn process_into_batch(
        &self,
        checkpoint: &Checkpoint,
    ) -> anyhow::Result<Box<dyn Any + Send>> {
        let values = self
            .pipeline
            .process(checkpoint)
            .with_context(|| format!("pipeline {} process failed", P::NAME))?;
        let mut acc = <P::Batch as Default>::default();
        self.pipeline.batch(&mut acc, values.into_iter());
        Ok(Box::new(acc))
    }

    fn commit_batch(
        &self,
        batch: Box<dyn Any + Send>,
        write_batch: &mut Batch,
    ) -> anyhow::Result<usize> {
        // `expect` here is appropriate: the only way a downcast
        // fails is if the adapter wired pipelines and their
        // accumulators in conflicting orders, which is a
        // programmer error caught at the first commit.
        let acc: Box<P::Batch> = batch
            .downcast::<P::Batch>()
            .expect("PipelineAdapter::commit_batch downcast — pipeline order mismatch");
        self.pipeline
            .commit(&self.schema, &acc, write_batch)
            .with_context(|| format!("pipeline {} commit failed", P::NAME))
    }
}

/// Builder for [`CheckpointExecutorAdapter`]. Accepts pipelines
/// via [`add_pipeline`](Self::add_pipeline) and finalizes via
/// [`build`](Self::build).
pub struct AdapterBuilder {
    db: Db,
    pipelines: Vec<Arc<dyn TipPipeline>>,
}

impl AdapterBuilder {
    /// Register a pipeline against the adapter.
    ///
    /// `pipeline` is the pipeline implementation; `schema` is the
    /// portion of the database it reads from / writes to. Both
    /// are wrapped in [`Arc`] because each pipeline impl is shared
    /// across every checkpoint indexed; cloning the `Arc` is the
    /// per-call cost.
    ///
    /// Pipelines commit in registration order. Order does not
    /// matter for correctness since each pipeline owns disjoint
    /// CFs, but it does set the order in which commits land in the
    /// shared [`Batch`].
    pub fn add_pipeline<P: Pipeline>(mut self, pipeline: Arc<P>, schema: Arc<P::Schema>) -> Self {
        self.pipelines
            .push(Arc::new(PipelineAdapter { pipeline, schema }));
        self
    }

    /// Finalize the builder into a [`CheckpointExecutorAdapter`].
    pub fn build(self) -> CheckpointExecutorAdapter {
        CheckpointExecutorAdapter {
            db: self.db,
            pipelines: self.pipelines,
            pending: Mutex::new(BTreeMap::new()),
        }
    }
}

impl CheckpointExecutorAdapter {
    /// Start building an adapter for `db`.
    pub fn builder(db: Db) -> AdapterBuilder {
        AdapterBuilder {
            db,
            pipelines: Vec::new(),
        }
    }

    /// Process `checkpoint` through every registered pipeline and
    /// stage the per-pipeline accumulators under the checkpoint's
    /// sequence number.
    ///
    /// May be called out-of-order across checkpoints — the
    /// staging map keys by sequence number and
    /// [`commit_update_for_checkpoint`](Self::commit_update_for_checkpoint)
    /// looks up the right entry. Calling with the same sequence
    /// number twice replaces the prior staged work; this matches
    /// `rpc_index`'s current shape.
    ///
    /// Errors from any pipeline's
    /// [`process`](crate::Pipeline::process) abort the call before
    /// any staging happens — there is no partial-stage state.
    pub fn index_checkpoint(&self, checkpoint: &Checkpoint) -> anyhow::Result<()> {
        let seq = checkpoint.summary.sequence_number;

        // Build every pipeline's accumulator before touching the
        // pending map, so a mid-way failure does not leave
        // partial state.
        let mut staged = Vec::with_capacity(self.pipelines.len());
        for pipeline in &self.pipelines {
            staged.push(pipeline.process_into_batch(checkpoint).with_context(|| {
                format!("indexing checkpoint {seq} for pipeline {}", pipeline.name())
            })?);
        }

        self.pending.lock().insert(seq, staged);
        Ok(())
    }

    /// Commit the staged updates for checkpoint `seq` atomically.
    ///
    /// Pops the per-pipeline accumulators for `seq` from the
    /// staging map, drains each into a shared [`Batch`], and
    /// commits. Returns the total row count across all pipelines
    /// for metrics.
    ///
    /// Returns an error if no staging entry exists for `seq`,
    /// either because [`index_checkpoint`](Self::index_checkpoint)
    /// was never called for that sequence number or because a
    /// prior call to this method already committed it.
    pub fn commit_update_for_checkpoint(&self, seq: u64) -> anyhow::Result<usize> {
        let staged = self
            .pending
            .lock()
            .remove(&seq)
            .with_context(|| format!("no staged updates for checkpoint {seq}"))?;

        // `staged` and `self.pipelines` are built in lockstep by
        // `index_checkpoint` (one accumulator per pipeline, in
        // registration order), so their lengths always match.
        debug_assert_eq!(
            self.pipelines.len(),
            staged.len(),
            "pipeline / staged length invariant"
        );
        let mut write_batch = self.db.batch();
        let mut total = 0usize;
        for (i, acc) in staged.into_iter().enumerate() {
            let pipeline = &self.pipelines[i];
            total += pipeline
                .commit_batch(acc, &mut write_batch)
                .with_context(|| {
                    format!(
                        "committing checkpoint {seq} for pipeline {}",
                        pipeline.name()
                    )
                })?;
        }
        write_batch.commit()?;
        Ok(total)
    }

    /// Number of checkpoints currently staged but not yet
    /// committed. Exposed for metrics and tests.
    pub fn pending_checkpoint_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// The lowest staged checkpoint sequence number, or `None` if
    /// no checkpoints are staged. Exposed so the validator's
    /// checkpoint executor can sanity-check that
    /// [`commit_update_for_checkpoint`](Self::commit_update_for_checkpoint)
    /// is called in monotonic order.
    pub fn next_staged_checkpoint(&self) -> Option<u64> {
        self.pending.lock().keys().next().copied()
    }

    /// Discard any staged update for `seq` without committing.
    /// Useful when the executor decides not to advance through a
    /// checkpoint (a reorg, for example).
    pub fn discard_staged_checkpoint(&self, seq: u64) -> bool {
        self.pending.lock().remove(&seq).is_some()
    }
}

impl std::fmt::Debug for CheckpointExecutorAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointExecutorAdapter")
            .field("pipelines", &self.pipelines.len())
            .field("pending", &self.pending.lock().len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Buf;
    use bytes::BufMut;
    use sui_types::base_types::ObjectID;
    use sui_types::object::Object;
    use sui_types::test_checkpoint_data_builder::TestCheckpointBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::DbMap;
    use crate::DbOptions;
    use crate::Decode;
    use crate::Encode;
    use crate::Schema;
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    use crate::error::OpenError;

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

    /// Two-CF schema used by the heterogeneous-pipeline tests.
    #[derive(Debug)]
    struct TestSchema {
        versions: DbMap<ObjectIdKey, U64Be>,
        counts: DbMap<ObjectIdKey, U64Be>,
    }

    impl Schema for TestSchema {
        fn cfs(base_options: &rocksdb::Options) -> Vec<crate::CfDescriptor> {
            vec![
                crate::CfDescriptor::new("versions", base_options.clone()),
                crate::CfDescriptor::new("counts", base_options.clone()),
            ]
        }

        fn open(db: &Db) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
                counts: DbMap::new(db.clone(), "counts")?,
            })
        }
    }

    /// Pipeline that records the highest version for each output
    /// object id.
    struct VersionsPipeline;

    impl Pipeline for VersionsPipeline {
        const NAME: &'static str = "versions";
        type Schema = TestSchema;
        type Value = (ObjectID, u64);
        type Batch = BTreeMap<ObjectID, u64>;

        fn restore(&self, _: &mut Self::Batch, _: &Object) -> anyhow::Result<()> {
            Ok(())
        }

        fn process(&self, checkpoint: &Checkpoint) -> anyhow::Result<Vec<Self::Value>> {
            let mut out = vec![];
            for tx in &checkpoint.transactions {
                for object in tx.output_objects(&checkpoint.object_set) {
                    out.push((object.id(), object.version().value()));
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

    /// Pipeline that counts how many output objects appear per id
    /// across a checkpoint.
    struct CountsPipeline;

    impl Pipeline for CountsPipeline {
        const NAME: &'static str = "counts";
        type Schema = TestSchema;
        type Value = ObjectID;
        type Batch = BTreeMap<ObjectID, u64>;

        fn restore(&self, _: &mut Self::Batch, _: &Object) -> anyhow::Result<()> {
            Ok(())
        }

        fn process(&self, checkpoint: &Checkpoint) -> anyhow::Result<Vec<Self::Value>> {
            let mut out = vec![];
            for tx in &checkpoint.transactions {
                for object in tx.output_objects(&checkpoint.object_set) {
                    out.push(object.id());
                }
            }
            Ok(out)
        }

        fn batch(&self, acc: &mut Self::Batch, values: std::vec::IntoIter<Self::Value>) {
            for id in values {
                *acc.entry(id).or_insert(0) += 1;
            }
        }

        fn commit(
            &self,
            schema: &Self::Schema,
            batch: &Self::Batch,
            write_batch: &mut Batch,
        ) -> anyhow::Result<usize> {
            for (id, c) in batch {
                write_batch.put(&schema.counts, &ObjectIdKey::new(*id), &U64Be(*c))?;
            }
            Ok(batch.len())
        }
    }

    /// Build a fresh database, schema, and a two-pipeline adapter.
    fn setup() -> (TempDir, Db, Arc<TestSchema>, CheckpointExecutorAdapter) {
        let dir = TempDir::new().unwrap();
        let (db, schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let schema = Arc::new(schema);
        let adapter = CheckpointExecutorAdapter::builder(db.clone())
            .add_pipeline(Arc::new(VersionsPipeline), schema.clone())
            .add_pipeline(Arc::new(CountsPipeline), schema.clone())
            .build();
        (dir, db, schema, adapter)
    }

    /// Build a single-checkpoint `Checkpoint` that creates one
    /// owned object with the given id.
    fn checkpoint_with_object(checkpoint_seq: u64, object_idx: u64) -> Checkpoint {
        TestCheckpointBuilder::new(checkpoint_seq)
            .start_transaction(0)
            .create_owned_object(object_idx)
            .finish_transaction()
            .build_checkpoint()
    }

    #[test]
    fn index_then_commit_writes_through_all_pipelines() {
        let (_dir, _db, schema, adapter) = setup();
        let cp = checkpoint_with_object(1, 7);
        // Each checkpoint produces multiple output objects (the
        // explicitly-created one plus producer-side bookkeeping
        // like the gas object); rely on cardinality bounds rather
        // than an exact count.
        let expected_objects = cp
            .transactions
            .iter()
            .flat_map(|tx| tx.output_objects(&cp.object_set))
            .count();
        adapter.index_checkpoint(&cp).unwrap();
        assert_eq!(adapter.pending_checkpoint_count(), 1);
        assert_eq!(adapter.next_staged_checkpoint(), Some(1));

        let rows = adapter.commit_update_for_checkpoint(1).unwrap();
        // Two pipelines, each writing one row per distinct id.
        assert_eq!(rows, expected_objects * 2);
        assert_eq!(adapter.pending_checkpoint_count(), 0);

        // Both pipelines wrote something.
        assert_eq!(schema.versions.iter(..).unwrap().count(), expected_objects);
        assert_eq!(schema.counts.iter(..).unwrap().count(), expected_objects);
    }

    #[test]
    fn index_out_of_order_commit_in_order_works() {
        let (_dir, _db, schema, adapter) = setup();
        let cp1 = checkpoint_with_object(1, 10);
        let cp2 = checkpoint_with_object(2, 20);
        let cp3 = checkpoint_with_object(3, 30);
        // Each checkpoint creates a distinct object id, plus a gas
        // object that the test builder shares across checkpoints
        // (so its appearance in `output_objects` for cp1, cp2, cp3
        // is the same id reused).
        let all_ids: std::collections::BTreeSet<ObjectID> = [&cp1, &cp2, &cp3]
            .iter()
            .flat_map(|cp| {
                cp.transactions
                    .iter()
                    .flat_map(|tx| tx.output_objects(&cp.object_set))
            })
            .map(|o| o.id())
            .collect();

        // Index out of order: 1, 3, 2.
        adapter.index_checkpoint(&cp1).unwrap();
        adapter.index_checkpoint(&cp3).unwrap();
        adapter.index_checkpoint(&cp2).unwrap();
        assert_eq!(adapter.pending_checkpoint_count(), 3);
        assert_eq!(adapter.next_staged_checkpoint(), Some(1));

        adapter.commit_update_for_checkpoint(1).unwrap();
        adapter.commit_update_for_checkpoint(2).unwrap();
        adapter.commit_update_for_checkpoint(3).unwrap();
        assert_eq!(adapter.pending_checkpoint_count(), 0);

        // Every distinct id observed across the three checkpoints
        // ends up as a row in the counts CF.
        assert_eq!(schema.counts.iter(..).unwrap().count(), all_ids.len());
    }

    #[test]
    fn commit_unknown_checkpoint_returns_error() {
        let (_dir, _db, _schema, adapter) = setup();
        let err = adapter.commit_update_for_checkpoint(42).unwrap_err();
        assert!(format!("{err:#}").contains("no staged updates"));
    }

    #[test]
    fn commit_clears_the_staging_entry() {
        let (_dir, _db, _schema, adapter) = setup();
        adapter
            .index_checkpoint(&checkpoint_with_object(5, 1))
            .unwrap();
        adapter.commit_update_for_checkpoint(5).unwrap();
        // A second commit at the same seq has nothing to drain.
        let err = adapter.commit_update_for_checkpoint(5).unwrap_err();
        assert!(format!("{err:#}").contains("no staged updates"));
    }

    #[test]
    fn discard_removes_staged_checkpoint() {
        let (_dir, _db, _schema, adapter) = setup();
        adapter
            .index_checkpoint(&checkpoint_with_object(9, 1))
            .unwrap();
        assert!(adapter.discard_staged_checkpoint(9));
        assert_eq!(adapter.pending_checkpoint_count(), 0);
        assert!(!adapter.discard_staged_checkpoint(9));
    }

    #[test]
    fn index_same_seq_replaces_prior_stage() {
        // Matches `rpc_index`'s current behavior: indexing twice
        // for the same seq overwrites the earlier stage. Only the
        // second checkpoint's output objects appear in the CF.
        let (_dir, _db, schema, adapter) = setup();
        let cp1 = checkpoint_with_object(1, 1);
        let cp2 = checkpoint_with_object(1, 2);
        let second_ids: std::collections::BTreeSet<ObjectID> = cp2
            .transactions
            .iter()
            .flat_map(|tx| tx.output_objects(&cp2.object_set))
            .map(|o| o.id())
            .collect();
        let first_only_id = cp1
            .transactions
            .iter()
            .flat_map(|tx| tx.output_objects(&cp1.object_set))
            .map(|o| o.id())
            .find(|id| !second_ids.contains(id));

        adapter.index_checkpoint(&cp1).unwrap();
        adapter.index_checkpoint(&cp2).unwrap();
        assert_eq!(adapter.pending_checkpoint_count(), 1);
        adapter.commit_update_for_checkpoint(1).unwrap();

        let persisted_ids: std::collections::BTreeSet<ObjectID> = schema
            .counts
            .iter(..)
            .unwrap()
            .map(|r| {
                let (k, _) = r.unwrap();
                ObjectID::new(k.0)
            })
            .collect();
        // The second checkpoint's ids are present.
        for id in &second_ids {
            assert!(
                persisted_ids.contains(id),
                "second-checkpoint id {id} should be persisted",
            );
        }
        // An id unique to the first checkpoint is not.
        if let Some(id) = first_only_id {
            assert!(
                !persisted_ids.contains(&id),
                "first-only id {id} should have been displaced",
            );
        }
    }

    #[test]
    fn empty_adapter_can_index_and_commit() {
        // No pipelines registered — index_checkpoint stages an empty
        // Vec, commit_update_for_checkpoint commits an empty batch.
        let dir = TempDir::new().unwrap();
        let (db, _schema) = Db::open::<TestSchema>(dir.path(), DbOptions::default()).unwrap();
        let adapter = CheckpointExecutorAdapter::builder(db).build();

        adapter
            .index_checkpoint(&checkpoint_with_object(1, 1))
            .unwrap();
        let rows = adapter.commit_update_for_checkpoint(1).unwrap();
        assert_eq!(rows, 0);
    }
}
