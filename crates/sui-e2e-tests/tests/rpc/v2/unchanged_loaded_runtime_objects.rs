// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use sui_macros::sim_test;
use sui_rpc::field::FieldMask;
use sui_rpc::field::FieldMaskUtil;
use sui_rpc::proto::sui::rpc::v2::GetCheckpointRequest;
use sui_rpc::proto::sui::rpc::v2::GetTransactionRequest;
use sui_types::transaction::CallArg;
use sui_types::transaction::ObjectArg;
use sui_types::transaction::TransactionData;
use sui_types::transaction::TransactionKind;
use sui_types::Identifier;
use test_cluster::TestClusterBuilder;

use crate::{stake_with_validator, transfer_coin};

#[sim_test]
async fn test_unchanged_loaded_runtime_objects() {
    use sui_types::programmable_transaction_builder::ProgrammableTransactionBuilder;

    let test_cluster = TestClusterBuilder::new().build().await;

    let _transaction_digest = transfer_coin(&test_cluster.wallet).await;
    let transaction_digest = stake_with_validator(&test_cluster).await;

    let mut client = sui_rpc::client::v2::Client::new(test_cluster.rpc_url()).unwrap();
    let t = client
        .ledger_client()
        .get_transaction(
            GetTransactionRequest::new(&transaction_digest)
                .with_read_mask(FieldMask::from_paths(["*"])),
        )
        .await
        .unwrap()
        .into_inner()
        .transaction
        .unwrap();

    assert!(t.effects().unchanged_loaded_runtime_objects().is_empty());

    let c = client
        .ledger_client()
        .get_checkpoint(
            GetCheckpointRequest::by_sequence_number(t.checkpoint())
                .with_read_mask(FieldMask::from_paths(["*"])),
        )
        .await
        .unwrap()
        .into_inner()
        .checkpoint
        .unwrap();

    assert!(!c
        .objects()
        .objects()
        .iter()
        .any(|o| o.object_type() == "package"));

    let address = test_cluster.get_address_0();
    let objects = client
        .state_client()
        .list_owned_objects(
            sui_rpc::proto::sui::rpc::v2::ListOwnedObjectsRequest::default()
                .with_owner(address.to_string())
                .with_read_mask(FieldMask::from_str("object_id,version,digest,object_type")),
        )
        .await
        .unwrap()
        .into_inner()
        .objects;

    let gas = &objects[0];

    let mut builder = ProgrammableTransactionBuilder::new();
    builder
        .move_call(
            "0x3".parse().unwrap(),
            Identifier::new("sui_system").unwrap(),
            Identifier::new("active_validator_voting_powers").unwrap(),
            vec![],
            vec![CallArg::Object(ObjectArg::SharedObject {
                id: "0x5".parse().unwrap(),
                initial_shared_version: 1.into(),
                mutable: false,
            })],
        )
        .unwrap();
    let ptb = builder.finish();
    let gas_data = sui_types::transaction::GasData {
        payment: vec![(
            gas.object_id().parse().unwrap(),
            gas.version().into(),
            gas.digest().parse().unwrap(),
        )],
        owner: address,
        price: 1000,
        budget: 100_000_000,
    };

    let kind = TransactionKind::ProgrammableTransaction(ptb);
    let tx_data = TransactionData::new_with_gas_data(kind, address, gas_data);

    let txn = test_cluster.wallet.sign_transaction(&tx_data).await;
    let transaction_digest = (*txn.digest()).into();

    // TODO: The data that we get back from execute_transaction isn't complete since we still need
    // to pipe through the info from the validators
    let _transaction = super::execute_transaction(&mut client, &txn).await;

    let t = client
        .ledger_client()
        .get_transaction(
            GetTransactionRequest::new(&transaction_digest)
                .with_read_mask(FieldMask::from_paths(["*"])),
        )
        .await
        .unwrap()
        .into_inner()
        .transaction
        .unwrap();

    assert_eq!(t.effects().unchanged_loaded_runtime_objects().len(), 1);
    assert_eq!(
        t.effects().unchanged_loaded_runtime_objects()[0].object_id(),
        "0x5b890eaf2abcfa2ab90b77b8e6f3d5d8609586c3e583baf3dccd5af17edf48d1"
    );

    assert_eq!(t.effects().unchanged_consensus_objects().len(), 1);
    assert_eq!(
        t.effects().unchanged_consensus_objects()[0].object_id(),
        "0x0000000000000000000000000000000000000000000000000000000000000005"
    );
}
