// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process Relay-to-global-router consumer adapter.
//!
//! Prototype of the "future Relay-to-global-router adapter" deferred by the host module:
//! it stays inside this crate and consumes the crate-private subscription seam directly,
//! so it inherits the actor's exact-sequence handshake without needing the delivery
//! cursors a WAN transport would require.
//!
//! Each source is one colocated [`KvDcRelay`] host treated as one pool lane (the Relay's
//! `dc_id` names the pool). A source must materialize exactly one serving-endpoint actor;
//! multi-endpoint relays are rejected rather than silently merged, because one lane holds
//! one publisher lease. Every source gets its own single-lane [`GlobalCkfIndexer`]: with
//! endpoint-derived indexer domains, cross-pool grouping into shared lanes cannot be
//! assumed, and per-source indexers keep prefix depths comparable (same tokens, block
//! size, and hash format) without requiring it. Any stream fault — lag, closure, a
//! non-applied ingest outcome — tears down that source's generation and resubscribes with
//! a fresh lease epoch; queries observe the source as not ready in between.

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::cuckoo::{
    ConsumerInstanceId, GlobalCkfIndexer, GlobalCkfIngestOutcome, GlobalCkfManifest, LaneLease,
    PrefixSearchConfig,
};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::host::KvDcRelay;

const SOURCE_POLL_INTERVAL: Duration = Duration::from_millis(200);
const SOURCE_REBUILD_BACKOFF: Duration = Duration::from_millis(500);

/// One colocated Relay host consumed as one pool lane.
pub struct RelayConsumerSource {
    /// Pool label surfaced in query results (for the global router this is the
    /// pool's Dynamo namespace).
    pub label: String,
    pub relay: Arc<KvDcRelay>,
}

/// Per-source prefix-match result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsumerLaneMatch {
    pub label: String,
    /// False while the source is between generations (no snapshot installed yet,
    /// or torn down after a stream fault) or has no ready lane.
    pub ready: bool,
    /// Matched prefix depth in KV blocks. Zero when not ready.
    pub prefix_depth: u32,
}

struct SourceSlot {
    label: String,
    published: Arc<ArcSwapOption<GlobalCkfIndexer>>,
}

/// Queryable set of relay-fed pool lanes for a colocated global router.
pub struct KvDcRelayConsumer {
    slots: Vec<SourceSlot>,
    tasks: parking_lot::Mutex<Vec<JoinHandle<()>>>,
    cancel: CancellationToken,
}

impl KvDcRelayConsumer {
    /// Spawn one ingestion task per source. Sources become queryable independently as
    /// their snapshots install; `find_matches` reports each source's readiness.
    pub fn start(sources: Vec<RelayConsumerSource>, consumer_instance: u64) -> Self {
        let cancel = CancellationToken::new();
        let instance = ConsumerInstanceId::new(consumer_instance);
        let mut tasks = Vec::with_capacity(sources.len());
        let slots = sources
            .into_iter()
            .map(|source| {
                let published: Arc<ArcSwapOption<GlobalCkfIndexer>> =
                    Arc::new(ArcSwapOption::empty());
                tasks.push(tokio::spawn(run_source(
                    source.label.clone(),
                    source.relay,
                    instance,
                    published.clone(),
                    cancel.child_token(),
                )));
                SourceSlot {
                    label: source.label,
                    published,
                }
            })
            .collect();
        Self {
            slots,
            tasks: parking_lot::Mutex::new(tasks),
            cancel,
        }
    }

    /// Match a block-hash sequence against every source lane.
    ///
    /// Always returns one entry per source, in construction order, so callers can
    /// distinguish "pool has no matching prefix" from "pool's lane is not ready".
    pub fn find_matches(&self, sequence: &[LocalBlockHash]) -> Vec<ConsumerLaneMatch> {
        self.slots
            .iter()
            .map(|slot| {
                let guard = slot.published.load();
                let matched = guard.as_ref().and_then(|indexer| {
                    indexer
                        .find_prefix_matches(sequence)
                        .ok()
                        .and_then(|result| result.lanes()[0])
                });
                match matched {
                    Some(lane_match) => ConsumerLaneMatch {
                        label: slot.label.clone(),
                        ready: true,
                        prefix_depth: lane_match.prefix_depth(),
                    },
                    None => ConsumerLaneMatch {
                        label: slot.label.clone(),
                        ready: false,
                        prefix_depth: 0,
                    },
                }
            })
            .collect()
    }

    pub async fn shutdown(&self) {
        self.cancel.cancel();
        let tasks: Vec<_> = self.tasks.lock().drain(..).collect();
        for task in tasks {
            let _ = task.await;
        }
    }
}

/// Drive one source through subscribe → snapshot → delta generations until cancelled.
async fn run_source(
    label: String,
    relay: Arc<KvDcRelay>,
    instance: ConsumerInstanceId,
    published: Arc<ArcSwapOption<GlobalCkfIndexer>>,
    cancel: CancellationToken,
) {
    let mut epoch: u64 = 1;
    while !cancel.is_cancelled() {
        match run_source_generation(&label, &relay, instance, epoch, &published, &cancel).await {
            Ok(()) => return, // cancelled
            Err(error) => {
                tracing::warn!(
                    source = %label,
                    epoch,
                    %error,
                    "KV DC Relay consumer source generation ended; resubscribing"
                );
            }
        }
        published.store(None);
        epoch += 1;
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(SOURCE_REBUILD_BACKOFF) => {}
        }
    }
}

async fn run_source_generation(
    label: &str,
    relay: &KvDcRelay,
    instance: ConsumerInstanceId,
    epoch: u64,
    published: &ArcSwapOption<GlobalCkfIndexer>,
    cancel: &CancellationToken,
) -> anyhow::Result<()> {
    // Wait for the relay to materialize its sole serving-endpoint actor.
    let handle = loop {
        let mut actors = relay.actor_handles().await;
        match actors.len() {
            0 => {
                tokio::select! {
                    _ = cancel.cancelled() => return Ok(()),
                    _ = tokio::time::sleep(SOURCE_POLL_INTERVAL) => continue,
                }
            }
            1 => break actors.pop().expect("length checked").1,
            n => {
                anyhow::bail!(
                    "consumer source '{label}' requires exactly one materialized endpoint, \
                     found {n}; scope the relay with namespace_filter/endpoint_prefix"
                );
            }
        }
    };

    let lease = LaneLease::new(instance, 0, epoch);
    let subscription = handle
        .subscribe(lease)
        .await
        .map_err(|error| anyhow::anyhow!("subscribe failed: {error}"))?;
    let identity = subscription.snapshot.identity();
    let mut lanes = [None; dynamo_kv_router::indexer::cuckoo::CKF_LANE_COUNT];
    lanes[0] = Some(identity.pool_id());
    let manifest = GlobalCkfManifest::new(
        instance,
        identity.indexer_domain(),
        identity.format(),
        lanes,
    )?;
    let indexer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default())?;
    let mut ingestor = indexer.claim_lane(0)?;
    ingestor.assign(identity, lease)?;
    let outcome = ingestor.install_snapshot(&subscription.snapshot);
    let GlobalCkfIngestOutcome::SnapshotInstalled { sequence } = outcome else {
        anyhow::bail!("initial snapshot rejected: {outcome:?}");
    };
    tracing::info!(
        source = %label,
        epoch,
        sequence,
        pool_id = ?identity.pool_id(),
        "KV DC Relay consumer lane ready"
    );
    published.store(Some(Arc::new(indexer)));

    let mut deltas = subscription.deltas;
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            received = deltas.recv() => match received {
                Ok(delta) => match ingestor.apply_delta(&delta) {
                    GlobalCkfIngestOutcome::DeltaApplied { .. } => {}
                    GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate { .. } => {}
                    outcome => anyhow::bail!("delta not applied: {outcome:?}"),
                },
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    anyhow::bail!("delta stream lagged by {skipped} publications");
                }
                Err(broadcast::error::RecvError::Closed) => {
                    anyhow::bail!("delta stream closed (actor retired or replaced)");
                }
            },
        }
    }
}
