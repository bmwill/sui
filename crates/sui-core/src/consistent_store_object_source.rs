// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`LiveObjectSource`] implementations bridging
//! [`AuthorityStore`](crate::authority::AuthorityStore) /
//! [`AuthorityPerpetualTables`] to
//! [`sui_consistent_store::object_source`]'s parallel restore
//! driver.
//!
//! The impls are thin wrappers around
//! [`AuthorityPerpetualTables::range_iter_live_object_set`]:
//! filter out wrapped entries, wrap each remaining [`Object`] in
//! `Ok(...)`, and forward the iterator. No copying beyond what the
//! underlying iterator already does.
//!
//! Nothing in this module wires these impls into a caller — the
//! driver is provided as a building block for the future migration
//! away from `rpc_index`. The existing
//! [`par_index_live_object_set`](crate::par_index_live_object_set)
//! path stays unchanged.

use sui_consistent_store::LiveObjectSource;
use sui_types::base_types::ObjectID;
use sui_types::object::Object;

use crate::authority::AuthorityStore;
use crate::authority::authority_store_tables::AuthorityPerpetualTables;
use crate::authority::authority_store_tables::LiveObject;

impl LiveObjectSource for AuthorityPerpetualTables {
    fn range<'a>(
        &'a self,
        start: ObjectID,
        end: ObjectID,
    ) -> Box<dyn Iterator<Item = anyhow::Result<Object>> + 'a> {
        Box::new(
            self.range_iter_live_object_set(Some(start), Some(end), false)
                .filter_map(|live| match live {
                    LiveObject::Normal(o) => Some(Ok(o)),
                    // `range_iter_live_object_set(false)` does not
                    // produce wrapped entries, but match
                    // exhaustively to keep the filter total.
                    LiveObject::Wrapped(_) => None,
                }),
        )
    }
}

/// Convenience impl forwarding through [`AuthorityStore`] to its
/// owned [`AuthorityPerpetualTables`].
///
/// Lets callers pass an `&AuthorityStore` directly without
/// reaching into the `perpetual_tables` field.
impl LiveObjectSource for AuthorityStore {
    fn range<'a>(
        &'a self,
        start: ObjectID,
        end: ObjectID,
    ) -> Box<dyn Iterator<Item = anyhow::Result<Object>> + 'a> {
        self.perpetual_tables.range(start, end)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use bytes::Buf;
    use bytes::BufMut;
    use sui_consistent_store::Batch;
    use sui_consistent_store::Db;
    use sui_consistent_store::DbMap;
    use sui_consistent_store::DbOptions;
    use sui_consistent_store::Decode;
    use sui_consistent_store::Encode;
    use sui_consistent_store::LiveObjectSource;
    use sui_consistent_store::Pipeline;
    use sui_consistent_store::RestoreRunner;
    use sui_consistent_store::RestoreState;
    use sui_consistent_store::Schema;
    use sui_consistent_store::error::DecodeError;
    use sui_consistent_store::error::EncodeError;
    use sui_consistent_store::error::OpenError;
    use sui_consistent_store::object_source::restore_pipeline_from_object_source;
    use sui_types::base_types::ObjectID;
    use tempfile::TempDir;

    use super::*;

    // ---- Pipeline scaffolding ----

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
        fn cfs(
            base_options: &sui_consistent_store::rocksdb::Options,
        ) -> Vec<(&'static str, sui_consistent_store::rocksdb::Options)> {
            vec![("versions", base_options.clone())]
        }

        fn open(db: &Arc<Db>) -> Result<Self, OpenError> {
            Ok(Self {
                versions: DbMap::new(db.clone(), "versions")?,
            })
        }
    }

    struct VersionsPipeline;

    impl Pipeline for VersionsPipeline {
        const NAME: &'static str = "versions";
        type Schema = VersionsSchema;
        type Value = (ObjectID, u64);
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

        fn process(
            &self,
            _: &sui_types::full_checkpoint_content::CheckpointData,
        ) -> anyhow::Result<Vec<Self::Value>> {
            Ok(vec![])
        }

        fn batch(&self, _: &mut Self::Batch, _: std::vec::IntoIter<Self::Value>) {}

        fn commit(
            &self,
            schema: &Self::Schema,
            batch: &Self::Batch,
            write_batch: &mut Batch,
        ) -> anyhow::Result<usize> {
            for (id, v) in batch {
                write_batch.put(
                    &schema.versions,
                    &ObjectIdKey::new(*id),
                    &U64Be(*v),
                )?;
            }
            Ok(batch.len())
        }
    }

    /// Open a fresh empty `AuthorityPerpetualTables` for testing.
    /// We bypass the full `AuthorityState` machinery — the trait
    /// impl only needs the perpetual tables' range iterator.
    fn open_perpetual_tables() -> (TempDir, AuthorityPerpetualTables) {
        let dir = TempDir::new().unwrap();
        let tables = AuthorityPerpetualTables::open(dir.path(), None, None);
        (dir, tables)
    }

    #[tokio::test]
    async fn empty_perpetual_tables_yields_no_objects() {
        let (_dir, tables) = open_perpetual_tables();
        let count = LiveObjectSource::range(&tables, ObjectID::ZERO, ObjectID::MAX).count();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn empty_perpetual_tables_drives_runner_to_complete_with_no_writes() {
        // End-to-end: open empty perpetual tables, run a restore
        // against them, confirm the runner transitions to Complete
        // and writes nothing.
        let (_perp_dir, tables) = open_perpetual_tables();

        let db_dir = TempDir::new().unwrap();
        let (db, schema) =
            Db::open::<VersionsSchema>(db_dir.path(), DbOptions::default()).unwrap();
        let schema = Arc::new(schema);
        let staging = TempDir::new().unwrap();
        let runner = Arc::new(RestoreRunner::new(
            db.clone(),
            Arc::new(VersionsPipeline),
            schema.clone(),
            7,
            staging.path().to_path_buf(),
            sui_consistent_store::rocksdb::Options::default(),
        ));

        restore_pipeline_from_object_source(runner, &tables, 3).unwrap();

        // No rows in the pipeline CF.
        let got_ids: BTreeSet<ObjectID> = schema
            .versions
            .iter(..)
            .unwrap()
            .map(|r| {
                let (k, _) = r.unwrap();
                ObjectID::new(k.0)
            })
            .collect();
        assert!(got_ids.is_empty());

        // Restore is marked Complete.
        match db.restore_state("versions").unwrap() {
            Some(RestoreState::Complete { restored_at }) => assert_eq!(restored_at, 7),
            other => panic!("expected Complete, got {other:?}"),
        }
    }
}
