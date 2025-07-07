// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

use sui_macros::sim_test;
use sui_rpc::field::FieldMask;
use sui_rpc::proto::sui::rpc::v2beta2::get_checkpoint_request::CheckpointId;
use sui_rpc::proto::sui::rpc::v2beta2::ledger_service_client::LedgerServiceClient;
use sui_rpc::proto::sui::rpc::v2beta2::{
    Checkpoint, ExecutedTransaction, GetCheckpointRequest, Object, Transaction,
};
use sui_rpc::proto::sui::rpc::v2beta2::{GetEpochRequest, GetTransactionRequest};
use sui_rpc::Client;
use sui_sdk_types::ObjectId;
use test_cluster::TestClusterBuilder;

use crate::{stake_with_validator, transfer_coin};
use sui_rpc::{
    field::FieldMaskUtil,
    proto::sui::rpc::v2beta2::{
        simulate_transaction_request::TransactionChecks, Argument, Command, Input, MoveCall,
        ProgrammableTransaction, SimulateTransactionRequest,
    },
};

#[tokio::test]
async fn calculate_staking_rewards() {
    let test_cluster = TestClusterBuilder::new().with_epoch_duration_ms(3_000).build().await;

    let _transaction_digest = transfer_coin(&test_cluster.wallet).await;
    let transaction_digest = stake_with_validator(&test_cluster).await;

    // let mut client = LedgerServiceClient::connect(test_cluster.rpc_url().to_owned())
    //     .await
    //     .unwrap();
    let mut client = Client::new(test_cluster.rpc_url()).unwrap();

    let transaction = client
        .ledger_client()
        .get_transaction(GetTransactionRequest {
            digest: Some(transaction_digest.to_string()),
            read_mask: Some(FieldMask::from_str("*")),
        })
        .await
        .unwrap()
        .into_inner()
        .transaction
        .unwrap();

    let sender = transaction.transaction.as_ref().unwrap().sender();

    let staked_object = transaction
        .effects
        .as_ref()
        .unwrap()
        .changed_objects
        .iter()
        .find(|o| o.object_type().contains("StakedSui"))
        .unwrap();

    println!("sender: {}", sender);
    println!("StakedSui: {:#?}", staked_object);
    // println!("{:#?}", transaction);

    let id = ObjectId::from_str(staked_object.object_id()).unwrap();
    println!("id: {}", staked_object.object_id());
    println!("id: {id}");
    for _ in 0..5 {
        let now = std::time::Instant::now();
        let resp = client.get_delegated_stake(&id).await.unwrap();
        println!("{}", now.elapsed().as_millis());
        // let resp = client
        //     .live_data_client()
        //     .simulate_transaction(SimulateTransactionRequest {
        //         transaction: Some(Transaction {
        //             kind: Some(
        //                 ProgrammableTransaction {
        //                     inputs: vec![
        //                         Input {
        //                             object_id: Some("0x5".into()),
        //                             ..Default::default()
        //                         },
        //                         Input {
        //                             object_id: Some(staked_object.object_id().into()),
        //                             ..Default::default()
        //                         },
        //                     ],
        //                     commands: vec![Command::from(MoveCall {
        //                         package: Some("0x3".to_owned()),
        //                         module: Some("sui_system".to_owned()),
        //                         function: Some("calculate_rewards".to_owned()),
        //                         type_arguments: vec![],
        //                         arguments: vec![Argument::new_input(0), Argument::new_input(1)],
        //                     })],
        //                 }
        //                 .into(),
        //             ),
        //             sender: Some(sender.into()),
        //             ..Default::default()
        //         }),
        //         read_mask: Some(FieldMask::from_str("*")),
        //         checks: Some(TransactionChecks::Disabled as _),
        //         do_gas_selection: None,
        //     })
        //     .await
        //     .unwrap()
        //     .into_inner();
        // let resp = client
        //     .ledger_client()
        //     .get_epoch(GetEpochRequest {
        //         epoch: None,
        //         read_mask: Some(FieldMask::from_str("*")),
        //     })
        //     .await;
        println!("{:#?}", resp);

        test_cluster.wait_for_epoch(None).await;
    }
}

// let mut client = Client::new("http://localhost:9000").unwrap();

// println!("{:#?}", resp);
