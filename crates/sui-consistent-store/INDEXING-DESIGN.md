# Indexing and restore design

This document records the design for the next layer above the
foundational `sui-consistent-store` primitive: a unified per-pipeline
trait for indexing logic, plus the four driver shells that consume it
(two restore drivers, two tip drivers).

`PLAN.md` tracks the foundational crate's roadmap and is the parent
document. This file is scoped to the indexing-pipeline layer; once the
shape stabilizes the relevant material folds into module documentation
and this file is removed.

## Goals

- One trait, written once per pipeline, that drives all four flows:
  - Restore from a formal snapshot (indexer-side).
  - Restore from a live-object-set iterator over the validator's
    perpetual store (validator-side).
  - Tip-of-chain indexing under `sui-indexer-alt-framework`.
  - Tip-of-chain indexing under `sui-core`'s checkpoint executor.
- Restore optimized for speed: build sorted SST files in worker
  threads, atomically ingest at the bottom level, no compaction
  backlog at the boundary into tip mode.
- No reopen between restore mode and tip mode. Only mutable RocksDB
  options change at the boundary.
- Resumable restore: a process death partway through restore must
  not require redoing already-ingested partitions.
- Mirror the existing `sui-indexer-alt-framework` `Processor` +
  `sequential::Handler` shapes so the framework adapter is a thin
  glue layer rather than a re-derivation.

## Non-goals (initial version)

- Concurrent (out-of-order) tip pipelines. v1 supports sequential
  tip pipelines only, matching the existing
  `sui-indexer-alt-consistent-store` story and the validator's
  in-order commit constraint.
- Restore from sources other than `Object`. The indexer's formal
  snapshot and the validator's perpetual store both surface
  `Object`s; nothing concrete demands a richer input record yet.
  We can promote the input to an associated type later.
- Macros for deriving `Pipeline` impls. Hand-written for v1.
- Watermark coordination across pipelines (the `Synchronizer`
  analog). Lives in the future framework adapter crate
  (`sui-consistent-store-framework`); not in scope here.

## Findings that drive the design

### Existing trait shapes are very close

`sui-indexer-alt-consistent-store`'s pipelines already implement
three traits in parallel against the framework:

```rust
// sui-indexer-alt-framework
trait Processor {
    const NAME: &'static str;
    type Value;
    async fn process(&self, &Arc<Checkpoint>) -> Result<Vec<Self::Value>>;
}

trait sequential::Handler: Processor {
    type Store;
    type Batch: Default;
    const MAX_BATCH_CHECKPOINTS: usize;
    fn batch(&self, &mut Self::Batch, std::vec::IntoIter<Self::Value>);
    async fn commit(&self, &Self::Batch, &mut Connection) -> Result<usize>;
}

// sui-indexer-alt-consistent-store
trait Restore<S: Schema>: Processor {
    const FANOUT: usize = 10;
    fn restore(schema: &S, &Object, &mut WriteBatch) -> Result<()>;
}
```

The validator's `rpc_index::index_checkpoint(&CheckpointData, _)` is
structurally a single fused `process` + `batch` + `commit`. (The
`LayoutResolver` parameter on today's signature is unused — see
`rpc_index.rs:640`.) Aligning the two means the validator adapter
can drive the same per-pipeline trait the framework adapter drives.

### `SstFileWriter` supports merge

`rust-rocksdb`'s `SstFileWriter` exposes `put`, `merge`, and `delete`
([docs](https://docs.rs/rocksdb/latest/rocksdb/struct.SstFileWriter.html)).
Keys must be inserted in strict comparator-sorted order within a
single SST. When such an SST is ingested, RocksDB's merge operator
applies at read time the same way it does for ordinary writes. So
schemas with merge operators (`balances`, `coin_index`) work cleanly
via SST ingest, with two implementation options the pipeline author
chooses:

1. Pre-fold merges in memory (per partition, into a `BTreeMap`) and
   emit one `put` per key. The right choice when the merge operator
   is a fold whose value at restore time is fully known — which
   matches `Balances`, since restore observes every coin once.
2. Emit raw merges into the SST, sorted by key. RocksDB applies them
   at read/compaction. The right choice when the partition's merges
   should accumulate against state from another partition or another
   pipeline.

The trait surface accommodates both because `restore` writes through a
`Batch` (the typed wrapper exposed by `sui-consistent-store`); the
driver decides whether `Batch` is backed by an `SstFileWriter` or by
an in-memory accumulator.

### The "no reopen" constraint rules out `prepare_for_bulk_load`

`Options::prepare_for_bulk_load` sets `num_levels = 2`, which is
immutable post-open. Reverting to normal levelled compaction requires
a close/reopen.

What *is* runtime-mutable via `db.set_options_cf` (per [advanced_options.h](https://github.com/facebook/rocksdb/blob/main/include/rocksdb/advanced_options.h)):

- `disable_auto_compactions`
- `level0_file_num_compaction_trigger`,
  `level0_slowdown_writes_trigger`, `level0_stop_writes_trigger`
- `write_buffer_size`, `max_write_buffer_number`
- `soft_pending_compaction_bytes_limit`,
  `hard_pending_compaction_bytes_limit`
- `target_file_size_base`, `target_file_size_multiplier`

`unordered_write` is *not* runtime-mutable. So the design accepts that
restore is somewhat slower than today's `rpc_index` (which opens with
`unordered_write=true` and reopens) in exchange for not reopening.

`IngestExternalFile` plus runtime-mutable compaction toggles is
sufficient: ingested SSTs land at the bottom level when the CF is
empty, bypassing memtable, WAL, and L0 entirely. There's no L0
backlog at the boundary into tip mode, so write stalls due to
compaction are not a concern.

## The unified `Pipeline` trait

```rust
/// A single indexing pipeline. Owns a slice of the schema and
/// supplies four functions: `restore` for per-object restore writes,
/// `process` for per-checkpoint extraction at tip, `batch` for
/// folding values across checkpoints into an accumulator, and
/// `commit` for atomic write.
///
/// Implementations are usually unit structs and the trait methods
/// are essentially functions; pass `&self` for symmetry with
/// `sui-indexer-alt-framework` and so configuration that varies
/// per-instance can live on the struct.
pub trait Pipeline: Send + Sync + 'static {
    /// Identifies the pipeline in logs, metrics, and persisted
    /// per-pipeline progress markers.
    const NAME: &'static str;

    /// The portion of the schema this pipeline reads from / writes
    /// to. Typically a borrow of the full schema or a sub-projection.
    type Schema: Send + Sync;

    /// The type produced by `process` and consumed by `batch`.
    /// Equivalent to `Processor::Value` in the framework today.
    type Value: Send + Sync + 'static;

    /// The accumulator into which `batch` folds values. The driver
    /// decides how much to fold (per-checkpoint or across many).
    type Batch: Default + Send + Sync + 'static;

    /// How much restore-side parallelism the driver should use for
    /// this pipeline. Both restore drivers honor it.
    const RESTORE_FANOUT: usize = 10;

    /// Maximum number of checkpoints to fold into a single tip-mode
    /// commit. Honored by the framework driver; the validator
    /// driver always uses 1.
    const MAX_BATCH_CHECKPOINTS: usize = 5 * 60;

    /// Fold updates derived from a single live object into the
    /// per-shard accumulator. Drivers parallelize across input
    /// objects up to `RESTORE_FANOUT`. Each worker owns its own
    /// accumulator; the driver feeds it to `commit` when the
    /// shard is done.
    ///
    /// Accumulator-style rather than writing through a `Batch`
    /// directly: SST ingestion requires at most one operation per
    /// key per shard's commit, so folding by key in the
    /// accumulator (which `commit` then drains into the write
    /// target) makes that invariant a property of the pipeline's
    /// data structure rather than a runtime check.
    fn restore(
        &self,
        accumulator: &mut Self::Batch,
        object: &Object,
    ) -> Result<(), Error>;

    /// Extract a checkpoint into rows. Pure function; called from
    /// worker threads under both drivers.
    fn process(
        &self,
        checkpoint: &CheckpointData,
    ) -> Result<Vec<Self::Value>, Error>;

    /// Fold a checkpoint's values into the running accumulator.
    /// `values` are presented in checkpoint order; the same `Batch`
    /// may receive values from many consecutive checkpoints before
    /// `commit`.
    fn batch(
        &self,
        batch: &mut Self::Batch,
        values: std::vec::IntoIter<Self::Value>,
    );

    /// Apply the folded accumulator to `write_batch`, atomically
    /// alongside any other state the driver decides to write
    /// (e.g. a watermark row). Returns the number of rows written
    /// for metrics.
    fn commit(
        &self,
        schema: &Self::Schema,
        batch: &Self::Batch,
        write_batch: &mut Batch,
    ) -> Result<usize, Error>;
}

```

A pipeline whose `process` is naturally async (none of the existing
ones are) wraps the trait in a `BoxFuture`-returning adapter; making
`process` itself async would force everyone into `async-trait`
boilerplate without benefit.

`process` takes just `&CheckpointData` rather than a context struct
that bundles a `LayoutResolver`. Today's `rpc_index::index_checkpoint`
threads a `LayoutResolver` through but the parameter is bound to
`_resolver` and never used (`rpc_index.rs:640`); the indexer-alt
pipelines never had one. Adding context fields when a real consumer
needs them is cheap; carrying an unused one forever is not.

### Why this shape over alternatives

- **Single trait vs. several supertraits.** The framework today
  splits `Processor`/`Handler`/`Restore`. Splitting clarifies which
  methods are "tip" vs "restore", but at the cost of three impl
  blocks per pipeline and a constraint chain a future framework
  adapter has to navigate. A single trait is friendlier to the
  validator-side adapter (which doesn't care about the split) and to
  schema authors. The framework adapter blanket-implements the
  framework's `Processor` and `sequential::Handler` for any
  `T: Pipeline`, recovering the split at the framework boundary.
- **Sync `process` and `commit`.** The validator's checkpoint
  executor is sync; the framework's per-checkpoint code is async but
  the body is pure. Keeping the trait sync avoids `async-trait` macro
  costs and an `Arc<Self>` requirement; the framework adapter wraps
  with `Box::pin(async move { ... })` if needed.
- **`Batch` typed via the foundational crate, not `rocksdb::WriteBatch`
  directly.** Keeps the trait surface free of raw rocksdb types and
  lets the SST-ingest driver substitute its own `Batch` impl that
  records writes into an `SstFileWriter` instead of a memtable batch.

## The four drivers

Each driver is a thin shell around a `Pipeline` impl. None of them
duplicates indexing logic; they only sequence calls to `restore` /
`process` / `batch` / `commit` and supply different write contexts.

### 1. `RestoreFromFormalSnapshot` (indexer-side)

Lives in `sui-consistent-store` itself. Mirrors the existing
`sui-indexer-alt-consistent-store::restore::Restorer` flow:

- `object_store`-backed source streams partitioned `LiveObjects`.
- Per-pipeline `mpsc::Receiver<Arc<LiveObjects>>` receives partitions.
- For each partition, spawn up to `RESTORE_FANOUT` workers that build
  a per-partition `SstFileWriter` (one per CF the pipeline writes
  to), call `Pipeline::restore` per object, and finalize.
- Once all partitions are written, atomically ingest all SSTs for
  the pipeline via `Db::ingest_files(...)`.
- On finalize, mark the pipeline complete in the persisted progress
  marker (see "Resumable restore" below).

Speed knobs: number of in-flight partitions, SST target file size,
workers per partition.

### 2. `RestoreFromPerpetualStore` (validator-side)

Lives in `sui-core`. Mirrors today's `par_index_live_object_set`:

- Partition the `ObjectID` space into `RESTORE_FANOUT * N` shards
  (the `1 << 5 = 32` choice today is fine as a default).
- Each shard wraps `perpetual_tables.range_iter_live_object_set(...)`
  as an `Iterator<Item = Result<Object, _>>` and hands it to the
  shared restore-runner exported by `sui-consistent-store`.

The driver lives in `sui-core` rather than in `sui-consistent-store`
because `AuthorityStore::perpetual_tables` is a heavy dep tree we
don't want in the foundational crate's default build. The
restore-runner accepts a generic `IntoIterator<Item = Object>` per
shard, so the validator-side wrapper is a few dozen lines of glue.

Both restore drivers share an internal helper inside
`sui-consistent-store` that owns the "open SSTs in a staging dir,
write through them, finalize, ingest, mark complete" lifecycle;
the only thing they differ on is the object stream.

### 3. `IndexerAltAdapter` (framework crate, future)

Lives in `sui-consistent-store-framework`. For each `T: Pipeline`,
provide:

```rust
impl<T: Pipeline> sui_indexer_alt_framework::pipeline::Processor for AltAdapter<T> { ... }
impl<T: Pipeline> sui_indexer_alt_framework::pipeline::sequential::Handler for AltAdapter<T> { ... }
```

so that `framework::Indexer::sequential_pipeline::<AltAdapter<MyPipeline>>(...)`
just works. The adapter also owns the `Synchronizer` analog that
coordinates `Db::take_snapshot` calls across pipelines at stride
boundaries.

### 4. `CheckpointExecutorAdapter` (validator-side)

A small helper struct (~100 lines) lives in `sui-consistent-store`,
wrapping a fixed list of pipelines and exposing the surface the
checkpoint executor uses today on `RpcIndexStore`:

```rust
impl<P: PipelineSet> CheckpointExecutorAdapter<P> {
    pub fn index_checkpoint(&self, checkpoint: &CheckpointData);
    pub fn commit_update_for_checkpoint(&self, seq: u64) -> Result<(), Error>;
}
```

(The existing call site in `checkpoint_executor/mod.rs:728` passes a
`&mut dyn LayoutResolver` that the index discards; the migration to
this adapter drops the unused argument.)

`PipelineSet` is a tuple-of-pipelines trait (or a `Vec<Box<dyn …>>` if
type-erasure is preferred). `index_checkpoint` calls `process` +
`batch` for each pipeline and stages a `Batch` keyed by `seq` in a
`Mutex<BTreeMap<u64, Batch>>`. `commit_update_for_checkpoint` pops the
batch and calls `commit` atomically. Per-pipeline watermarks are
written into the same atomic batch.

## Resumable restore

Restore is one-shot: there is no resume from an arbitrary mid-restore
point at object granularity. But process death between partitions
should not waste work.

Approach: the foundational `sui-consistent-store` crate gains a
single internal CF (`__restore`) keyed by pipeline name, holding a
small typed enum:

```rust
pub(crate) enum RestoreState {
    InProgress { partitions_complete: BTreeSet<PartitionId>, target_checkpoint: u64 },
    Complete { restored_at: u64 },
}
```

The restore driver writes `partitions_complete` after each
`ingest_external_file_cf` call, atomically with whatever
per-pipeline marker the consumer cares about. On startup, the driver
queries the state and skips already-ingested partitions.

`PartitionId` is the formal snapshot's partition ID for driver 1, or
the `ObjectID`-range shard ID for driver 2. The driver decides; the
state CF only stores it as opaque bytes.

This is purely additive; pipelines that don't restore at all simply
have no entry. The framework adapter and the checkpoint executor
adapter both check this state on startup and refuse to begin tip
indexing while any pipeline is still `InProgress`.

## RocksDB strategy

The foundational `sui-consistent-store` crate gains:

- `pub struct SstWriter<K, V>` — a typed wrapper over
  `rocksdb::SstFileWriter`. Methods:
  - `put(key: &K, value: &V) -> Result<(), Error>`
  - `merge(key: &K, operand: &V) -> Result<(), Error>`
  - `delete(key: &K) -> Result<(), Error>`
  - `finish(self) -> Result<PathBuf, Error>`

  Constraint: keys must be inserted in encoded-byte order. The
  encoding traits' contract is that the comparator on bytes matches
  the comparator on values (the tuple-prefix story documented in
  PLAN.md), so callers can sort their domain values before encoding
  and the resulting byte order is the right one.

- `pub fn Db::ingest_files_cf(&self, cf: &str, paths: Vec<PathBuf>)`
  — ingests one CF's SSTs atomically with sane defaults
  (`move_files=true`, `snapshot_consistency=true`,
  `allow_blocking_flush=true`). Multiple CFs are ingested by calling
  this multiple times; ordering doesn't matter for atomicity within
  a CF, and the per-CF call is what `IngestExternalFile` is scoped
  to anyway.

- `pub fn Db::set_restore_options_cf(&self, cf: &str)` and
  `pub fn Db::set_tip_options_cf(&self, cf: &str)` — toggle the
  mutable compaction knobs documented above. Restore mode raises L0
  triggers and disables auto-compaction; tip mode restores defaults
  (or per-CF values supplied at open time, captured for reversal).

The "captured for reversal" requires a small bookkeeping struct
inside `Db` that remembers each CF's tip-mode option set; both
toggles are runtime-mutable so no reopen is needed. The schema's
`Schema::cfs` already returns per-CF `rocksdb::Options`, so the tip
defaults are knowable at open time.

The restore driver wraps this in a high-level sequence:

1. `Db::set_restore_options_cf` for each CF the pipeline writes to.
2. Build SSTs.
3. `Db::ingest_files_cf` per CF.
4. `Db::set_tip_options_cf` for each CF.
5. Update the `__restore` CF's `RestoreState`.

## Crate layering

Two crates, growing outward:

1. **`sui-consistent-store`** (this crate, exists today). Holds
   *both* the foundational RocksDB primitive and the
   indexing-pipeline layer:
   - Foundational primitive: `Db`, `DbMap`, `Schema`, `Batch`,
     `SnapshotHandle`, plus the new `SstWriter`,
     `Db::ingest_files_cf`, and the
     `set_restore_options_cf` / `set_tip_options_cf` toggles.
   - Indexing layer: the `Pipeline` trait, the formal-snapshot fetch
     path (donor:
     `sui-indexer-alt-consistent-store::restore::formal_snapshot`),
     the shared restore-runner that the two restore drivers feed
     objects into, and the validator-side
     `CheckpointExecutorAdapter`.

   Workspace dependencies: `sui-types` (for `Object` and
   `CheckpointData`) and `sui-storage` (for the formal-snapshot
   wire format's `Blob` codec; replaceable later if we want to
   slim the dep). Crates.io: `object_store`, `tempfile`,
   `tracing`, `bytes`, `parking_lot`, etc. as today.

   Excluded from the crate (and from its dep tree):
   - `sui-indexer-alt-framework`. The framework adapter lives in
     a sibling crate (below).
   - `sui-storage::AuthorityStore` / `sui-core` types. The
     validator-side perpetual-store restore driver lives in
     `sui-core` itself and feeds objects into the shared
     restore-runner via a generic stream parameter.

2. **`sui-consistent-store-framework`** (future, named in PLAN.md).
   Implements `sui-indexer-alt-framework`'s store traits and
   bridges `Pipeline` → framework's `Processor` + `Handler`. Hosts
   the cross-pipeline snapshot synchronizer.

We considered splitting the indexing layer into a third
`sui-consistent-store-pipeline` crate to keep the foundational
primitive on crates.io alone. We decided against it: there is one
real consumer of the foundational primitive in isolation today
(none, in fact — both consumers want the indexing layer too), and
collapsing the layers means fewer Cargo.toml manifests, fewer
re-exports, and one fewer crate to publish. If a future consumer
genuinely wants the foundational primitive without `sui-types`, we
can split then.

## Open trait shape decisions to confirm before implementing

1. **`Pipeline` taking `&self` vs. associated functions.** Today's
   `Restore::restore` is an associated function (no `&self`); today's
   `Processor::process` takes `&self`. I've chosen `&self`
   consistently above so per-instance config (e.g. fanout overrides
   from a config file, a logger handle) can live on the struct.
   Confirm or override.
2. **`Batch` as the foundational typed wrapper, not `rocksdb::WriteBatch`.**
   This is a minor break from today's `Restore` trait (which exposes
   a raw `rocksdb::WriteBatch`). Every pipeline impl moves from
   `schema.balances.merge(&key, delta, batch)?` (typed_store-like
   open-coded merge) to `batch.merge(&schema.balances, &key, &delta)?`
   (the foundational crate's typed `Batch::merge`). I think this is
   strictly an improvement; flag if not.
3. **`SstWriter<K, V>` typed-or-untyped.** Typed matches the rest of
   the crate's surface; the trade-off is that the same SST-ingest
   driver can't write to multiple CFs with different `K, V` through
   a single dyn-friendly type. The driver will need an opaque
   `Box<dyn SstWriterErased>` or generic over the pipeline's CF set.
   I'd default to typed and let the driver hold a small enum or a
   per-CF `SstWriter` map.
4. **`sui-storage` dependency.** Folding the formal-snapshot fetch
   path in pulls `sui-storage` (for the `Blob` codec). It's not a
   light dep, but it's the same one the existing
   `sui-indexer-alt-consistent-store` carries today, and the
   formal-snapshot file format is genuinely defined in those terms.
   If we want to slim it later, we can vendor the few hundred lines
   of `Blob` we use; for now, take the dep.

## Implementation roadmap

Each step lands as a single atomic commit. Each commit must build,
pass `cargo fmt`, `cargo xclippy`, and `cargo nextest run` for the
crate.

1. **Foundational RocksDB additions.**
   1. `SstWriter<K, V>` typed wrapper plus tests (round-trip a small
      sorted SST, ingest into a fresh DB, read back).
   2. `Db::ingest_files_cf` plus tests (ingest into empty CF, ingest
      into populated CF, ingest with overlap, ingest with merge
      operator).
   3. `Db::set_restore_options_cf` / `set_tip_options_cf` plus tests
      (toggle, write, toggle back, observe no stalls; the per-CF
      tip-defaults bookkeeping struct).
2. **`sui-types` integration and the `Pipeline` trait.**
   1. Add `sui-types` to `Cargo.toml`. Define the `Pipeline` trait
      in a new `pipeline` module.
   2. A tiny in-crate test pipeline that exercises all four methods
      end-to-end against an in-memory `Db`. No restore drivers yet —
      the test calls `Pipeline::restore` directly.
   3. The `__restore` internal CF and `RestoreState` typed accessor.
3. **Shared restore-runner.** Generic over an
   `IntoIterator<Item = Object>` per shard. Owns the "open SSTs,
   write through them, finalize, ingest, mark complete" lifecycle.
   Tests with a synthetic in-memory object stream.
4. **Formal-snapshot restore driver.** Donor code from
   `sui-indexer-alt-consistent-store::restore::{formal_snapshot,
   format, storage, broadcaster}`, adapted onto the shared
   restore-runner. Adds the `sui-storage` dep. Tests with a small
   pre-built fixture.
5. **`CheckpointExecutorAdapter`.** Wraps a fixed list of pipelines,
   exposes `index_checkpoint` / `commit_update_for_checkpoint`. Smoke
   test that mirrors `rpc_index`'s shape.
6. **Validator-side perpetual-store restore driver.** Lives in
   `sui-core`. Donor code from `par_index_live_object_set`, adapted
   to feed the consolidated crate's restore-runner via a generic
   shard iterator. Smoke test with a fake `AuthorityStore`.
7. **`sui-consistent-store-framework` crate.** Implements
   `Processor` + `sequential::Handler` for
   `AltAdapter<T: Pipeline>`. Hosts the snapshot `Synchronizer`.
   End-to-end test: spin up a small framework `Indexer` with two
   pipelines, ingest a synthetic checkpoint stream, observe correct
   watermarks and a stride-aligned snapshot.
8. **Migration-of-one.** Port one existing pipeline (probably
   `Balances`, smallest schema, exercises both `merge` and a
   `BTreeMap` accumulator) onto the new trait, in both consumers.
   The actual cutover of the rest is out of scope for this design.
