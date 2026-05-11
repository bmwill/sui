# sui-consistent-store

## Overview

`sui-consistent-store` is a foundational, type-safe wrapper around RocksDB
intended to back Sui's on-disk indexes — both the validator's RPC index
(today: `sui-core::rpc_index`) and the indexer's consistent store (today:
`sui-indexer-alt-consistent-store`). It is a peer to neither: existing
implementations are expected to migrate onto it over time.

This document records the design decisions made for the initial version
of the crate. It is a planning artifact; once the design stabilizes, the
relevant material will fold into module-level documentation and this
file will be removed.

## Goals

- Pluggable per-type encoding for keys and values; no hardcoded bincode
  or bcs.
- Strong static typing for keys and values at the API surface.
- In-memory point-in-time snapshots of the database, addressable by an
  externally supplied monotonic checkpoint number, for serving consistent
  reads.
- Forward, reverse, and prefix iteration over typed column families.
- Zero-copy reads where possible, exposed through `bytes::Bytes` so that
  callers do not have to manage RocksDB lifetimes themselves.
- *(Future)* Optional, opt-in filesystem-level checkpoints for
  backup and offline use; not implemented in v1 because the
  existing `sui-indexer-alt-consistent-store` does not use them
  either, and a real consumer can drive the design when it
  surfaces.
- A plain, hand-written schema model — schemas are ordinary Rust structs
  of typed `DbMap<K, V>` fields.
- Modest, considered dependency footprint. The foundational primitives
  stay on crates.io alone, but the indexing-pipeline layer takes a
  dependency on `sui-types` (for `Object` and `CheckpointData`) so a
  unified `Pipeline` trait can drive both restore and tip flows from
  a single impl. See `INDEXING-DESIGN.md`.

## Non-goals (initial version)

- Macros for deriving schemas. Schemas are hand-written; we will revisit
  derives only if the boilerplate proves painful at scale.
- Static prefix-of-key safety. Prefix iteration accepts any
  `&impl Encode`; the contract that the prefix's encoding is a byte
  prefix of the full key's encoding is documented, not enforced.
- Concurrent (out-of-order) commit pipelines via the framework adapter.
  v1 of the framework crate will support sequential pipelines only,
  which is the only model compatible with the cross-pipeline snapshot
  coordination used today.
- Built-in helpers for the validator's "stage batch keyed by checkpoint,
  commit later in order" pattern. Consumers stage as they see fit.
- A built-in watermark schema. The framework adapter (separate crate)
  owns watermark and restoration bookkeeping.

## Design decisions

### 1. Foundational primitive

`sui-consistent-store` is a foundational primitive. Existing on-disk
indexes (`sui-core::rpc_index`,
`sui-indexer-alt-consistent-store`) are expected to migrate onto it.
The crate has no knowledge of the indexer-alt framework or of
validator execution; those concerns live in a sibling crate
(`sui-consistent-store-framework`, future) or in consumers. The
crate *does* know about Sui-level types (`Object` and
`CheckpointData`) at the indexing-pipeline layer, and about the
formal-snapshot file format on the restore side. See
`INDEXING-DESIGN.md` for the full layering rationale.

### 2. Encoding bound to the type

Encoding is a property of the Rust type, not of the column family. Each
key and value type implements `Encode` and (typically) `Decode`. Schemas
are expected to use bespoke wrapper newtypes for keys and values rather
than reusing arbitrary domain types. Two motivations:

- Migration and evolution. Introducing a new wrapper type alongside the
  old one (with a different on-disk representation) is a non-disruptive
  way to evolve a schema. Dual-write, then cut over.
- It forces an explicit decision about the on-disk representation at
  the point each new key or value type is added.

The cost is more boilerplate at schema-definition time, which we accept.

### 3. Custom `Encode` / `Decode` traits

The crate defines its own minimal encoding traits rather than sitting on
top of `bincode` 2.x or `serde`. The shape:

```rust
pub trait Encode {
    fn encode_into<B: bytes::BufMut>(&self, buf: &mut B) -> Result<(), EncodeError>;
}

pub trait Decode: Sized {
    fn decode(bytes: &[u8]) -> Result<Self, DecodeError>;
}
```

Encode is fallible. Some encodings (for instance, tuple-style fixed-int
encodings applied to types not statically known to fit) can fail; the
cost of returning `Result` is negligible. The `BufMut` parameter is
load-bearing: it exposes only `put_*` methods, so the append-only
contract is enforced by the type system rather than by trust.

`EncodeError` and `DecodeError` are crate-defined boxed-inner structs
that carry a free-form message and an optional source error; the
top-level `Error` enum has variants for each plus `Rocksdb`,
`MissingColumnFamily`, and a defensive `Internal` for invariant
violations.

A `DecodeBorrowed<'a>` trait was scoped out of the v1 surface; the
zero-copy path is reachable today through `DbMap::get_raw` returning
`bytes::Bytes`. We can revisit if a schema has a real need for typed
borrowed views.

### 4. In-memory snapshot model

Consistent reads are served from in-memory `rocksdb::Snapshot`s held in
a bounded `BTreeMap<u64, (Arc<Snapshot>, Watermark)>` keyed by an
externally supplied monotonic checkpoint number. This matches the
existing `sui-indexer-alt-consistent-store` design and inherits its
trade-offs:

- The buffer is empty after restart; readers cannot serve historical
  checkpoints across process boundaries from in-memory snapshots alone.
- A small, bounded number of long-lived snapshots is fine; many or
  unbounded growth pressures compaction.
- The `Db`'s public methods take `&self` and synchronize with a
  `parking_lot::RwLock` on the snapshot buffer. Snapshots'
  `rocksdb::Snapshot<'_>` borrows are extended to `'static` via
  `mem::transmute`; soundness rests on field declaration order
  (the `RwLock<BTreeMap<…, Arc<SnapshotEntry>>>` is declared *before*
  `inner: rocksdb::DB`, so every retained snapshot drops before the
  DB) and on `SnapshotHandle` field order (entry before the co-owned
  `Arc<Db>`). Both `unsafe` sites carry safety comments deriving the
  argument from these invariants.

RocksDB also exposes a filesystem-level checkpoint mechanism via
`rocksdb::checkpoint::Checkpoint`, which produces a hard-linked copy
of the live SSTs that can be opened as a standalone read-only or
secondary database. We considered exposing this as
`Db::create_checkpoint(path)` but skipped it for v1: the existing
`sui-indexer-alt-consistent-store` does not use it, and adding it
without a real consumer would freeze a design (path layout,
manifest handling, error semantics) we can shape better with one in
hand. The crate stays focused on in-memory snapshots until a
filesystem-checkpoint use case surfaces.

### 5. Crate split

Two crates:

- **`sui-consistent-store`** (this crate). The foundational primitive
  *and* the indexing-pipeline layer (the `Pipeline` trait, the two
  restore drivers, the formal-snapshot fetch path, and the
  validator's checkpoint-executor adapter). No
  `sui-indexer-alt-framework` dependency. Workspace dependencies
  are limited to `sui-types` and `sui-storage` (for the formal
  snapshot wire format). The validator-side perpetual-store
  restore driver lives in `sui-core` and feeds objects into this
  crate's restore runner via a small stream trait.
- **`sui-consistent-store-framework`** (future). Implements the
  `sui-indexer-alt-framework-store-traits` interfaces and contains
  the cross-pipeline snapshot coordinator (analogous to today's
  `sui-indexer-alt-consistent-store::store::synchronizer`).

We will build and stabilize the core crate before starting the
framework crate. See `INDEXING-DESIGN.md` for the indexing-layer
plan.

### 6. Zero-copy reads

The pinned-slice path in `rust-rocksdb` exposes `DBPinnableSlice<'a>`
whose `'a` is a `PhantomData<&'a DB>` annotation only. The actual
backing memory is either:

- A reference-counted block in the RocksDB block cache, kept alive
  until the slice's `Drop` runs the registered `Cache::Release`
  cleanup, or
- A copied buffer owned by the C++ `PinnableSlice` allocation itself
  (the `PinSelf` path: memtable hits, merge results, wide-column
  values), freed when the slice's Drop runs
  `rocksdb_pinnableslice_destroy`.

Neither path requires a live `&DB` borrow; both require only that the
DB allocation outlives the slice. We co-own `Arc<DB>` with each pinned
slice and lifetime-extend via `transmute`, then wrap
`(Arc<DB>, DBPinnableSlice<'static>)` in
`bytes::Bytes::from_owner(...)`. Drop order in the wrapping struct is
load-bearing: slice first, `Arc` second.

The crate's read API exposes:

- `DbMap::get(&K) -> Result<Option<V>>`. Owned decode. Implemented
  internally via `get_pinned` to avoid the redundant copy that
  `DB::get`'s standard path performs.
- `DbMap::get_raw(&K) -> Result<Option<bytes::Bytes>>`. Raw bytes,
  decoded by the caller. Lifetime-free at the API surface.

Iteration uses `DBRawIterator` (the borrowed-slice iterator) so that
keys and values are read zero-copy from the iterator's internal buffer
before being decoded into owned Rust types.

A pinned-slice cache caveat applies: each outstanding `Bytes` clone of
a block-cache-backed value pins an LRU handle. Long-lived or unbounded
pins can drive `block_cache.pinned-usage` past
`block_cache.capacity()`. We will document this on `get_raw` and
recommend short scopes for the returned `Bytes`.

### 7. Hand-written schemas, parameterized by reader

A schema is a struct of `DbMap<K, V, R>` fields, parameterized by a
[`Reader`](crate::Reader) (defaulted to [`Live`](crate::Live)). Two
trait impls per schema:

```rust
pub trait Schema: Sized {
    fn cfs(base_options: &rocksdb::Options) -> Vec<(&'static str, rocksdb::Options)>;
    fn open(db: &Arc<Db>) -> Result<Self, OpenError>;
}

pub trait SchemaAtSnapshot {
    type At<'s> where Self: 's;
    fn at<'s>(&'s self, snap: &'s SnapshotHandle) -> Self::At<'s>;
}
```

`Schema` is implemented for `MySchema<Live>` and supplies the column
families and the live constructor. `SchemaAtSnapshot` is the
companion that re-binds each field via [`DbMap::at`](crate::DbMap::at)
and produces `MySchema<Snapshot<'s>>`. Authors who never read at a
snapshot can skip the `SchemaAtSnapshot` impl. We will revisit a
`#[derive(Schema)]` macro only if the hand-written form becomes
painful at scale.

### 8. Documented prefix iteration, no static safety

`DbMap::iter_prefix` accepts `&impl Encode` and seeds RocksDB iterator
bounds with the encoded prefix. The contract that the encoded prefix is
a byte prefix of every encoded key it should match is documented, not
enforced. In practice, schemas that encode compound keys with a
prefix-preserving encoding (tuples under bincode big-endian fixed-int,
length-tagged byte concatenation, and similar) and pass tuple prefixes
(`&(owner,)` to match `&(owner, type, id)`) get the right behavior. The
crate's iteration documentation will spell this out with a worked
example.

## Architecture

### Module layout (planned)

- `error` — the top-level `Error` enum, plus `EncodeError`,
  `DecodeError`, and `OpenError`.
- `encode` — the `Encode`, `Decode`, and `DecodeBorrowed` traits.
- `db` — `Db` (the wrapped RocksDB), `DbOptions` opening configuration,
  and the internal self-referential `Inner` that holds the
  `rocksdb::DB` and the snapshot buffer.
- `schema` — the `Schema` trait.
- `map` — `DbMap<K, V>`, the typed column-family handle. Point reads
  (`get`, `get_raw`, `multi_get`, `multi_get_raw`).
- `batch` — `Batch`, the typed atomic-write builder.
- `iter` — typed forward, reverse, and prefix iterators built on
  `DBRawIterator`.
- `snapshot` — the `Snapshot` handle, plus `take_snapshot`,
  `at_snapshot`, and `snapshots_range` on `Db`. Reads from a `Snapshot`
  re-use the same `DbMap` API surface.
- `checkpoint` — filesystem-level `Db::create_checkpoint(path)`.

### Type sketch

```rust
pub struct Db { /* RwLock<BTreeMap<u64, Arc<SnapshotEntry>>> + rocksdb::DB. */ }

pub trait Reader: private::Sealed {
    fn db(&self) -> &Arc<Db>;
    fn read_options(&self) -> rocksdb::ReadOptions;
}
pub struct Live { db: Arc<Db> }
pub struct Snapshot<'s> { handle: &'s SnapshotHandle }

pub struct DbMap<K, V, R: Reader = Live> {
    reader: R,
    cf_name: Box<str>,
    _data: PhantomData<fn(K) -> V>,
}

impl<K, V> DbMap<K, V, Live> {
    pub fn new(db: Arc<Db>, cf_name: impl Into<Box<str>>) -> Result<Self, OpenError>;
}

impl<K, V, R: Reader> DbMap<K, V, R> {
    /// Re-bind at a captured snapshot.
    pub fn at<'s>(&self, snap: &'s SnapshotHandle) -> DbMap<K, V, Snapshot<'s>>;

    /// Reads route through `self.reader.read_options()` and `db()`.
    pub fn get(&self, key: &K) -> Result<Option<V>> where K: Encode, V: Decode;
    pub fn get_raw(&self, key: &K) -> Result<Option<Bytes>> where K: Encode;
    pub fn multi_get<'k, I>(&self, keys: I) -> Result<Vec<Result<Option<V>>>>
    where K: Encode + 'k, V: Decode, I: IntoIterator<Item = &'k K>;
    pub fn iter(&self, range: impl RangeBounds<K>) -> Result<Iter<'_, K, V>>;
    pub fn iter_prefix(&self, prefix: &impl Encode) -> Result<Iter<'_, K, V>>;
    pub fn iter_rev(&self, range: impl RangeBounds<K>) -> Result<RevIter<'_, K, V>>;
    pub fn contains_key(&self, key: &K) -> Result<bool> where K: Encode;
}

pub struct Batch { /* Wraps rocksdb::WriteBatch + Arc<Db>. */ }

impl Batch {
    /// Live-only: snapshot-bound projections are statically read-only.
    pub fn put<K: Encode, V: Encode>(
        &mut self,
        map: &DbMap<K, V, Live>,
        key: &K,
        value: &V,
    ) -> Result<&mut Self>;
    pub fn delete<K: Encode, V>(
        &mut self,
        map: &DbMap<K, V, Live>,
        key: &K,
    ) -> Result<&mut Self>;
    pub fn merge<K: Encode, V: Encode>(
        &mut self,
        map: &DbMap<K, V, Live>,
        key: &K,
        operand: &V,
    ) -> Result<&mut Self>;
    pub fn commit(self) -> Result<()>;
}
```

## Implementation roadmap

Each step lands as a single atomic commit. Each commit must build, pass
`cargo fmt --check` (with the project's import-granularity and
doc-comment formatting flags), pass `cargo clippy`, pass
`cargo nextest run`, and pass `cargo test --doc`, all scoped to this
crate.

1. **(this commit)** Empty crate skeleton, `PLAN.md`, and workspace
   integration. The crate compiles to an empty `lib.rs`.
2. **Encoding traits and errors.** `Encode`, `Decode`, `DecodeBorrowed`,
   `EncodeError`, and `DecodeError`. Tests on a couple of trivial
   implementations defined in the test module.
3. **`Db` core and the `Schema` trait.** Open and drop, schema-driven
   column-family registration, the internal self-referential structure,
   and `DbOptions` configuration. Tests open and close an empty schema.
4. **`DbMap<K, V>` point reads.** `get`, `get_raw`, `multi_get`, and
   `multi_get_raw`. Internal reads route through `get_pinned`.
5. **`Batch`.** Typed `put` and `delete`, plus atomic commit.
6. **Iteration.** Forward, reverse, and prefix iterators built on
   `DBRawIterator`.
7. **In-memory snapshots.** The snapshot buffer, `take_snapshot`,
   `at_snapshot`, `snapshots_range`, and snapshot-bound reads and
   iterators.
8. **Merge operators.** Typed `Batch::merge`. Schema authors install
   merge operators via the per-CF `rocksdb::Options` they return
   from [`Schema::cfs`](crate::Schema::cfs) (already supported);
   this commit closes the loop by exposing a typed entry point that
   actually triggers the operator.

Within each step we will add module-level documentation and rustdoc
with at least one worked example for every public-facing API.

## Testing strategy

- Each step adds unit tests against a small in-crate test schema. We
  define a couple of trivial key and value types (for example,
  `U64Key`, `BytesValue`) that implement the encoding traits manually
  with hand-rolled byte representations. This avoids pulling in a
  serialization dependency (bcs, bincode, or serde) just for tests.
- Integration tests exercise multi-step flows (schema open, batch
  write, snapshot, snapshot read, snapshot drop).
- `tempfile::TempDir` provisions per-test on-disk databases.
- We do not run under the simulator (no `cargo simtest` here); the
  crate is plain synchronous Rust.

## Open questions and future work

- Whether `DbMap::get_raw` needs a complementary borrowed-decode
  helper (`DbMap::get_borrowed::<R: DecodeBorrowed<'_>>`) or whether
  byte-returning plus caller-side decode is good enough in practice.
- Concurrent (out-of-order) framework store integration. Deferred to
  the framework crate; not part of this crate's surface.
- Compaction filters and column-family-specific options. The
  `Schema::cfs()` method returns full `rocksdb::Options` per column
  family, so this is consumer-driven; we may add convenience builders
  later.
- Metrics. Per-CF runtime metrics land via `Db::cf_metrics`
  returning a populated `RocksMetrics`; consumers map this into
  Prometheus or similar. We will revisit if patterns emerge that
  warrant a richer surface.
