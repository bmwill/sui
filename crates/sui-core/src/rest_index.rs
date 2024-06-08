// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use itertools::Itertools;
use move_core_types::language_storage::TypeTag;
use serde::{Deserialize, Serialize};
use sui_json_rpc_types::SuiObjectDataFilter;
use sui_types::base_types::ObjectInfo;
use sui_types::base_types::{
    ObjectDigest, ObjectID, SequenceNumber, SuiAddress, TransactionDigest,
};
use sui_types::dynamic_field::{self, DynamicFieldInfo};
use sui_types::error::{SuiError, SuiResult};
use sui_types::messages_checkpoint::CheckpointContents;
use sui_types::object::{Object, Owner};
use sui_types::storage::error::Error as StorageError;
use tracing::{debug, info, trace};
use typed_store::rocks::{
    default_db_options, read_size_from_env, DBBatch, DBMap, DBOptions, MetricConf,
};
use typed_store::traits::Map;
use typed_store::traits::{TableSummary, TypedStoreDebug};
use typed_store::TypedStoreError;
use typed_store_derive::DBMapUtils;

use crate::authority::authority_store_tables::LiveObject;
use crate::authority::AuthorityStore;
use crate::checkpoints::CheckpointStore;
use crate::state_accumulator::AccumulatorStore;

type OwnerIndexKey = (SuiAddress, ObjectID);
type DynamicFieldKey = (ObjectID, ObjectID);

#[derive(Debug)]
pub struct ObjectIndexChanges {
    pub deleted_owners: Vec<OwnerIndexKey>,
    pub deleted_dynamic_fields: Vec<DynamicFieldKey>,
    pub new_owners: Vec<(OwnerIndexKey, ObjectInfo)>,
    pub new_dynamic_fields: Vec<(DynamicFieldKey, DynamicFieldInfo)>,
}

#[derive(Clone, Copy, Serialize, Deserialize, Eq, PartialEq, Debug)]
pub struct TransactionInfo {
    checkpoint: u64,
    timestamp_ms: u64,
}

#[derive(DBMapUtils)]
struct IndexStoreTables {
    transactions: DBMap<TransactionDigest, TransactionInfo>,

    /// This is an index of object references to currently existing objects, indexed by the
    /// composite key of the SuiAddress of their owner and the object ID of the object.
    /// This composite index allows an efficient iterator to list all objected currently owned
    /// by a specific user, and their object reference.
    owner: DBMap<OwnerIndexKey, ObjectInfo>,
    // /// This is an index of object references to currently existing dynamic field object, indexed by the
    // /// composite key of the object ID of their parent and the object ID of the dynamic field object.
    // /// This composite index allows an efficient iterator to list all objects currently owned
    // /// by a specific object, and their object reference.
    // dynamic_field: DBMap<DynamicFieldKey, DynamicFieldInfo>,
}

impl IndexStoreTables {
    fn owner(&self) -> &DBMap<OwnerIndexKey, ObjectInfo> {
        &self.owner
    }

    fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }

    fn init(
        &mut self,
        authority_store: &AuthorityStore,
        checkpoint_store: &CheckpointStore,
    ) -> Result<(), StorageError> {
        info!("Initializing REST indexes");

        // Iterate through transactions/checkpoints that have yet to be pruned and fill out the
        // transactions index.
        if let Some(highest_executed_checkpint) =
            checkpoint_store.get_highest_executed_checkpoint_seq_number()?
        {
            let lowest_available_checkpoint =
                checkpoint_store.get_highest_pruned_checkpoint_seq_number()?;

            let mut batch = self.transactions.batch();

            for seq in lowest_available_checkpoint..=highest_executed_checkpint {
                let checkpoint = checkpoint_store
                    .get_checkpoint_by_sequence_number(seq)?
                    .ok_or_else(|| StorageError::missing(format!("missing checkpoint {seq}")))?;
                let contents = checkpoint_store
                    .get_checkpoint_contents(&checkpoint.content_digest)?
                    .ok_or_else(|| StorageError::missing(format!("missing checkpoint {seq}")))?;

                let info = TransactionInfo {
                    checkpoint: checkpoint.sequence_number,
                    timestamp_ms: checkpoint.timestamp_ms,
                };

                batch.insert_batch(
                    &self.transactions,
                    contents.iter().map(|digests| (digests.transaction, info)),
                )?;
            }

            batch.write()?;
        }

        // Iterate through live object set to initialize object-based indexes
        for object in authority_store
            .iter_live_object_set(false)
            .filter_map(LiveObject::to_normal)
        {
            let Owner::AddressOwner(owner) = object.owner else {
                continue;
            };

            let mut batch = self.owner.batch();

            // Owner Index
            let owner_key = (owner, object.id());
            let object_info = ObjectInfo::new(&object.compute_object_reference(), &object);

            batch.insert_batch(&self.owner, [(owner_key, object_info)])?;

            batch.write()?;
        }

        info!("Finished initializing REST indexes");

        Ok(())
    }

    fn prune(
        &self,
        checkpoint_contents_to_prune: &[CheckpointContents],
    ) -> Result<(), TypedStoreError> {
        let mut batch = self.transactions.batch();

        let transactions_to_prune = checkpoint_contents_to_prune
            .iter()
            .flat_map(|contents| contents.iter().map(|digests| digests.transaction));

        batch.delete_batch(&self.transactions, transactions_to_prune)?;

        batch.write()
    }
}

pub struct RestIndexStore {
    tables: IndexStoreTables,
}

impl RestIndexStore {
    pub fn new(
        path: &Path,
        authority_store: &AuthorityStore,
        checkpoint_store: &CheckpointStore,
    ) -> Self {
        let mut tables = IndexStoreTables::open_tables_read_write(
            path.into(),
            MetricConf::new("rest-index"),
            None,
            None,
        );

        // If the index tables are empty then we need to populate them
        if tables.is_empty() {
            tables.init(authority_store, checkpoint_store).unwrap();
        }

        Self { tables }
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn prune(
        &self,
        checkpoint_contents_to_prune: &[CheckpointContents],
    ) -> Result<(), TypedStoreError> {
        self.tables.prune(checkpoint_contents_to_prune)
    }

    pub fn get_owner_objects(
        &self,
        owner: SuiAddress,
        cursor: Option<ObjectID>,
        limit: usize,
        filter: Option<SuiObjectDataFilter>,
    ) -> SuiResult<Vec<ObjectInfo>> {
        let cursor = match cursor {
            Some(cursor) => cursor,
            None => ObjectID::ZERO,
        };
        Ok(self
            .get_owner_objects_iterator(owner, cursor, filter)?
            .take(limit)
            .collect())
    }

    /// starting_object_id can be used to implement pagination, where a client remembers the last
    /// object id of each page, and use it to query the next page.
    pub fn get_owner_objects_iterator(
        &self,
        owner: SuiAddress,
        starting_object_id: ObjectID,
        filter: Option<SuiObjectDataFilter>,
    ) -> SuiResult<impl Iterator<Item = ObjectInfo> + '_> {
        Ok(self
            .tables
            .owner()
            .unbounded_iter()
            // The object id 0 is the smallest possible
            .skip_to(&(owner, starting_object_id))?
            .skip(usize::from(starting_object_id != ObjectID::ZERO))
            .take_while(move |((address_owner, _), _)| address_owner == &owner)
            .filter(move |(_, o)| {
                if let Some(filter) = filter.as_ref() {
                    filter.matches(o)
                } else {
                    true
                }
            })
            .map(|(_, object_info)| object_info))
    }
}
