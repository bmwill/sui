use clap::Parser;
use std::path::Path;
use sui_data_ingestion_core::ReaderOptions;

use anyhow::Result;
use async_trait::async_trait;
use sui_data_ingestion_core::{setup_single_workflow_with_file_progress_store, Worker};
use sui_types::full_checkpoint_content::CheckpointData;

struct CustomWorker {
    dir: String,
}

#[async_trait]
impl Worker for CustomWorker {
    async fn process_checkpoint(&self, checkpoint: CheckpointData) -> Result<()> {
        let sequence_number = checkpoint.checkpoint_summary.sequence_number;
        let bcs = bcs::to_bytes(&checkpoint).unwrap();

        match bcs::from_bytes::<sui_sdk2::types::CheckpointData>(&bcs) {
            Ok(_) => {
                if (sequence_number % 10000) == 0 {
                    tracing::info!("processed checkpoint {sequence_number}");
                }
            }
            Err(e) => {
                tracing::error!("error deserializing checkpoint {sequence_number}: {e}");
                std::fs::write(format!("{}/{sequence_number}.checkpoint", self.dir), bcs)?;
                std::fs::write(format!("{}/{sequence_number}.err", self.dir), e.to_string())?;
            }
        }
        Ok(())
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(long, default_value_t = 10)]
    concurrency: usize,
    #[clap(long, default_value_t = 10)]
    batch_size: usize,
    #[clap(long, default_value = "mainnet")]
    network: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (_guard, _handle) = telemetry_subscribers::TelemetryConfig::new().init();

    let args = Args::parse();

    tracing::info!("starting scan with {args:#?}");

    let dir = format!("checkpoints/{}", args.network);
    std::fs::create_dir_all(&dir).unwrap();
    if !Path::new("progress").exists() {
        std::fs::write("progress", "{}").unwrap();
    }
    let (executor, term_sender) = setup_single_workflow_with_file_progress_store(
        CustomWorker {
            dir,
        },
        args.network.clone(),
        format!("https://checkpoints.{}.sui.io", args.network),
        "progress".into(),
        args.concurrency,
        Some(ReaderOptions {
            batch_size: args.batch_size,
            ..Default::default()
        }),
    )
    .await?;
    executor.await?;
    Ok(())
}
