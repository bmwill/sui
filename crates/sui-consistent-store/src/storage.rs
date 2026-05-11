// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! [`Storage`] — the minimal byte-fetching abstraction used by the
//! formal-snapshot restore path.
//!
//! The trait has one method: fetch the bytes at a path. Two impls
//! ship with the crate:
//!
//! - A blanket impl for any
//!   [`object_store::ObjectStore`](object_store::ObjectStore),
//!   covering S3, GCS, Azure, local filesystem, and HTTP via the
//!   `object_store` builders.
//! - An [`HttpStorage`] for raw HTTP endpoints, since `object_store`'s
//!   HTTP backend is opinionated about path layout and we sometimes
//!   want a thin reqwest-based fetcher.
//!
//! Higher layers
//! ([`FormalSnapshot`](crate::formal_snapshot::FormalSnapshot))
//! consume a `Arc<dyn Storage + Send + Sync + 'static>` so the
//! backend can be chosen at the calling crate's CLI layer.

use std::time::Duration;

use anyhow::Context as _;
use bytes::Bytes;
use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use url::Url;

/// Fetches byte blobs by path. Implemented for arbitrary
/// `object_store` backends and for raw HTTP via [`HttpStorage`].
#[async_trait::async_trait]
pub trait Storage {
    /// Fetch the bytes at `path`. Implementations should surface
    /// transport-level failures (4xx / 5xx, DNS, TLS) as
    /// [`anyhow::Error`] with sufficient context for logging.
    async fn get(&self, path: Path) -> anyhow::Result<Bytes>;
}

/// Common timeouts for snapshot fetches, threaded through both
/// [`HttpStorage`] and the `object_store` builders.
///
/// Both fields default to "no timeout" — appropriate for restores
/// of large snapshots where long-tailed individual fetches are
/// preferable to spurious retries. Production drivers may set
/// finite values via CLI / config.
#[derive(Clone, Debug, Default)]
pub struct StorageConnectionArgs {
    /// How long to wait for a single fetch to complete. `None` means
    /// no timeout.
    pub snapshot_timeout_ms: Option<u64>,

    /// How long to wait while establishing a TCP / TLS connection.
    /// `None` means no timeout.
    pub snapshot_connection_timeout_ms: Option<u64>,
}

/// A raw-HTTP [`Storage`] backed by `reqwest`.
///
/// Use this when the snapshot is served by a plain HTTP endpoint
/// that does not need the path-translation logic the `object_store`
/// HTTP backend imposes. Use the
/// [`object_store::http::HttpBuilder`] backend instead when the
/// endpoint is conventionally laid out for `object_store`.
pub struct HttpStorage {
    endpoint: Url,
    client: reqwest::Client,
}

impl HttpStorage {
    /// Construct a new [`HttpStorage`] rooted at `endpoint`.
    ///
    /// `endpoint` must be a URL with a trailing slash (or one will
    /// be appended implicitly by [`Url::join`]); subsequent
    /// [`get`](Self::get) calls resolve paths against it.
    pub fn new(endpoint: Url, args: StorageConnectionArgs) -> anyhow::Result<Self> {
        let mut builder = reqwest::ClientBuilder::new().https_only(false);

        if let Some(timeout) = args.snapshot_timeout_ms {
            builder = builder.timeout(Duration::from_millis(timeout));
        }
        if let Some(timeout) = args.snapshot_connection_timeout_ms {
            builder = builder.connect_timeout(Duration::from_millis(timeout));
        }

        Ok(Self {
            endpoint,
            client: builder
                .build()
                .context("Failed to build HTTP client for snapshot storage")?,
        })
    }
}

#[async_trait::async_trait]
impl Storage for HttpStorage {
    async fn get(&self, path: Path) -> anyhow::Result<Bytes> {
        let url = self
            .endpoint
            .join(path.as_ref())
            .with_context(|| format!("Bad path: {path}"))?;

        self.client
            .get(url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch: {path}"))?
            .error_for_status()
            .with_context(|| format!("Failed to fetch: {path}"))?
            .bytes()
            .await
            .with_context(|| format!("Failed to read bytes from: {path}"))
    }
}

#[async_trait::async_trait]
impl<S: ObjectStore> Storage for S {
    async fn get(&self, path: Path) -> anyhow::Result<Bytes> {
        ObjectStoreExt::get(self, &path)
            .await
            .with_context(|| format!("Failed to fetch: {path}"))?
            .bytes()
            .await
            .with_context(|| format!("Failed to read bytes from: {path}"))
    }
}

impl From<StorageConnectionArgs> for ClientOptions {
    fn from(args: StorageConnectionArgs) -> ClientOptions {
        let mut opts = ClientOptions::new();
        opts = if let Some(timeout) = args.snapshot_timeout_ms {
            opts.with_timeout(Duration::from_millis(timeout))
        } else {
            opts.with_timeout_disabled()
        };
        opts = if let Some(timeout) = args.snapshot_connection_timeout_ms {
            opts.with_connect_timeout(Duration::from_millis(timeout))
        } else {
            opts.with_connect_timeout_disabled()
        };
        opts
    }
}

#[cfg(test)]
mod tests {
    use object_store::local::LocalFileSystem;
    use tempfile::TempDir;

    use super::*;

    #[tokio::test]
    async fn object_store_storage_round_trip_via_local_filesystem() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("hello.txt"), b"world").unwrap();
        let store = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let got: Bytes = Storage::get(&store, Path::from("hello.txt")).await.unwrap();
        assert_eq!(&got[..], b"world");
    }

    #[tokio::test]
    async fn object_store_storage_missing_path_errors() {
        let dir = TempDir::new().unwrap();
        let store = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let err = Storage::get(&store, Path::from("nope.txt")).await.unwrap_err();
        assert!(format!("{err:#}").to_lowercase().contains("failed to fetch"));
    }

    #[test]
    fn client_options_from_args_with_timeouts() {
        // Smoke test — we cannot easily introspect ClientOptions's
        // configured values, but constructing it must not panic.
        let _opts: ClientOptions = StorageConnectionArgs {
            snapshot_timeout_ms: Some(1000),
            snapshot_connection_timeout_ms: Some(500),
        }
        .into();
    }

    #[test]
    fn client_options_from_args_disabled_timeouts() {
        let _opts: ClientOptions = StorageConnectionArgs::default().into();
    }
}
