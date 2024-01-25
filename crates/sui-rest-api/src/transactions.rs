// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use axum::extract::{Path, State};
use sui_types::digests::TransactionDigest;
use sui_types::effects::{TransactionEffects, TransactionEffectsAPI, TransactionEvents};
use sui_types::storage::ReadStore;
use sui_types::transaction::Transaction;
use tap::Pipe;

use crate::Result;
use crate::{accept::AcceptFormat, response::ResponseContent};

pub const GET_TRANSACTION_PATH: &str = "/transactions/:transaction";

//TODO fix output type
pub async fn get_transaction<S: ReadStore>(
    Path(transaction_digest): Path<TransactionDigest>,
    accept: AcceptFormat,
    State(state): State<S>,
) -> Result<ResponseContent<TransactionResponse>> {
    let transaction = (*state
        .get_transaction(&transaction_digest)?
        .ok_or(TransactionNotFoundError(transaction_digest))?)
    .clone()
    .into_inner();
    let effects = state
        .get_transaction_effects(&transaction_digest)?
        .ok_or(TransactionNotFoundError(transaction_digest))?;
    let events = if let Some(event_digest) = effects.events_digest() {
        state
            .get_events(event_digest)?
            .ok_or(TransactionNotFoundError(transaction_digest))?
            .pipe(Some)
    } else {
        None
    };

    let checkpoint = state.get_transaction_checkpoint(&transaction_digest)?;
    let timestamp_ms = if let Some(checkpoint) = checkpoint {
        state
            .get_checkpoint_by_sequence_number(checkpoint)?
            .map(|checkpoint| checkpoint.timestamp_ms)
    } else {
        None
    };

    let response = TransactionResponse {
        transaction,
        effects,
        events,
        checkpoint,
        timestamp_ms,
    };

    match accept {
        AcceptFormat::Json => ResponseContent::Json(response),
        AcceptFormat::Bcs => ResponseContent::Bcs(response),
    }
    .pipe(Ok)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TransactionResponse {
    transaction: Transaction,
    effects: TransactionEffects,
    events: Option<TransactionEvents>,
    checkpoint: Option<u64>,
    timestamp_ms: Option<u64>,
}

#[derive(Debug)]
pub struct TransactionNotFoundError(TransactionDigest);

impl std::fmt::Display for TransactionNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Transaction {} not found", self.0)
    }
}

impl std::error::Error for TransactionNotFoundError {}

impl From<TransactionNotFoundError> for crate::RestError {
    fn from(value: TransactionNotFoundError) -> Self {
        Self::new(axum::http::StatusCode::NOT_FOUND, value.to_string())
    }
}
