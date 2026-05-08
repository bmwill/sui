# Future design considerations

This file captures design questions we've considered but deferred.
They're worth revisiting when (a) a real consumer's needs make the
trade-offs concrete, or (b) the foundational crate's surface
stabilizes enough that a larger restructure is cheap relative to
later breakage.

PLAN.md tracks the active implementation plan; this file tracks
"things we want to remember to think about again."

## Default-snapshot reads

### Status: not implemented (Shape D below)

The crate exposes two reading surfaces today: `DbMap<K, V>` for
live-tip reads and `SnapshotHandle` / `SnapshotView<K, V>` for
snapshot-bound reads. A schema struct holds `DbMap` fields, so the
"easy" path is `schema.foo.get(&k)?` (live, no consistency
guarantee), and snapshot-bound reads require the longer
`db.at_snapshot(cp)?.view(&schema.foo).get(&k)?` form.

We've considered restructuring the API so consistent reads are the
default. The current decision is to leave the surface as it is and
let consumers (the framework crate, `rpc_index`, etc.) opt into
their own enforcement. This document records the alternatives we
weighed.

### Why we deferred

The foundational crate's job is primitives. Different consumers
diverge:

- The framework crate, which serves a `consistent-store`-style
  RPC at strict checkpoints, wants snapshots-by-default and would
  benefit from API-level enforcement.
- `rpc_index` doesn't care about consistency at all; it reads at
  the live tip from a single thread and would chafe against any
  forced snapshot construction.

A pluggable layer above us can satisfy both. Baking strict
defaults into the foundation would push the framework's policy
onto every other consumer.

### Shape A: rename live methods to make snapshot reads the path of least resistance

Keep the types as-is, but rename `DbMap`'s read methods with a
`live_` prefix (or move them onto a new `LiveDbMap` newtype that
`DbMap` doesn't expose by default). Snapshot reads via
`SnapshotView` keep clean names.

```rust
// Live-tip (now explicit):
schema.items.live_get(&k)?;
schema.items.live_iter(range)?;

// Snapshot (the recommended path):
let items = db.latest_snapshot().expect("snapshot taken").view(&schema.items);
items.get(&k)?;
items.iter(range)?;
```

**Pros:** small change. Cosmetic enforcement: nothing prevents the
`live_*` calls, but typing them makes the trade-off explicit at
every call site. `SnapshotView` carries the ergonomic burden.

**Cons:** doesn't actually prevent inconsistent reads —
discipline-only. And a consumer that wants live-tip reads for a
real reason (`rpc_index`) is now annotating every read.

**Cost:** API rename across `DbMap` plus a documentation pass; ~1
small commit.

### Shape B: strip `DbMap` of read methods; all reads go through readers

`DbMap<K, V>` becomes metadata only. All reads route through one
of two reader types:

- `SnapshotHandle` / `SnapshotView` — exists, snapshot-bound.
- `LiveReader` / `LiveView<K, V>` — new, live-tip reads.
  Constructed via `db.live()`, named to make the trade-off
  obvious.

```rust
// Snapshot:
let items = db.latest_snapshot().expect("...").view(&schema.items);
items.get(&k)?;

// Live (explicit):
let items = db.live().view(&schema.items);
items.get(&k)?;
```

A `Reader` trait could unify the two so generic code can be
written against either. GATs would let us also unify the iterator
return types — heavier but feasible.

**Pros:** the API enforces the choice. Both paths look identical
at the call site so switching is a one-line change. The framework
crate gets snapshot-by-default for free.

**Cons:** breaking change to `DbMap`. Every existing test, doc
example, and (eventually) consumer call site must be updated.
Adds the `LiveReader` and `LiveView` types and a `Reader` trait.
Slightly more verbose at every call site
(`db.live().view(&map).get(&k)?` vs today's `map.get(&k)?`).

**Cost:** medium-large restructure. Best done before the crate has
many consumers.

### Shape C: schema is parameterized by the reader

The schema's `open` returns typed handles that have already chosen
a reader. The same schema definition produces two types — say
`MySchema<Live>` and `MySchema<Snapshot>` — and reads against the
chosen reader are direct calls on the field.

```rust
struct MySchema<R: Reader> {
    items: TypedMap<R, U64Be, U64Be>,
}

// Open as live:
let (db, schema): (_, MySchema<Live>) = Db::open_live(path, opts)?;
schema.items.get(&k)?;

// Open as snapshot:
let snap = db.at_snapshot(cp)?;
let schema = snap.bind::<MySchema>()?; // MySchema<Snapshot>
schema.items.get(&k)?;
```

**Pros:** maximum ergonomics post-binding. Once a reader is chosen
and the schema is constructed, every subsequent read is plain
`schema.field.get(&k)?` and the type system enforces consistency
for the lifetime of that binding.

**Cons:** by far the biggest restructure. `DbMap` becomes generic
over the reader, schemas have a different construction shape,
write paths (`Batch`) need their own story, and the schema author
has to understand reader generics. Heavy investment for an
incremental ergonomic gain over Shape B.

**Cost:** large. Worth it only if a strong consumer use case
demands it.

### Schema ergonomics for snapshots (orthogonal to A/B/C)

Three smaller ideas surfaced alongside the bigger Shape choices.
None has been implemented; each is independently mergeable.

1. **`Schema::view_at(&self, &SnapshotHandle)` returning a
   parallel "view-of" struct.** Useful when reading against many
   maps from one snapshot. Costs a parallel type per schema (or a
   derive macro to generate it).
2. **`SnapshotHandle::with<S: Schema>(&schema)` returning a
   `SchemaView<'_, S>`.** Symmetric to (1). Avoids per-schema
   types if the view can be generic over the schema.
3. **`db.with_snapshot(cp, |snap, schema| { ... })`** — a
   closure-scoped reader pattern. Tidy for read-only handlers.
   Opinionated, but doesn't preclude the other entry points.

### Recommendation when revisiting

If a consumer asks for "snapshot by default" and we're still early
in adoption, **Shape B**. It's the cleanest hard-enforcement
outcome and `LiveReader` gives consumers like `rpc_index` an
explicit out without breaking semantics.

If we're past the point of breaking changes, **Shape A** plus
strong documentation is the cheapest nudge.

Shape C is on the table only if we discover that snapshot-bound
reads dominate enough call sites that the per-call `view(&map)`
indirection becomes a real burden — at which point a derive macro
generating per-schema view types might be a less invasive way to
reach the same destination.

The schema-ergonomic ideas (1)–(3) are useful regardless of A/B/C;
revisit when there's a concrete schema with many CFs that gets
read against from one snapshot frequently.
