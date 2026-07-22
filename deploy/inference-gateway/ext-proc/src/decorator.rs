// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decorator pickers over an inner [`EndpointPicker`] — prototype for the
//! scorer-composition surface discussed in the NV co-design thread.
//!
//! [`CalibratedPicker`] wraps the stock `Router` and adds, without touching
//! the router API:
//! - a **shadow prefix index** per endpoint (self-computed chained block
//!   hashes over `PickResult.token_ids`, TTL-decayed) that produces a
//!   per-request *predicted* cached-token count at pick time;
//! - the **calibration join**: `on_request_complete`'s
//!   `usage.prompt_tokens_details.cached_tokens` (PR #11868) is compared with
//!   the prediction, the signed error is logged per request, and the shadow
//!   TTL is adjusted online (over-prediction → decay faster; under-prediction
//!   → decay slower);
//! - a **TTFT EMA** per endpoint fed by the pick→`on_prefill_complete` edge —
//!   the viability signal for a forfeit gate: when every endpoint's EMA
//!   exceeds `DYN_DECORATOR_TTFT_FORFEIT_MS`, `pick()` returns
//!   [`PickError::Saturated`] before consulting the inner router.
//!
//! Everything is env-gated by `DYN_DECORATOR=calibration`; when unset the
//! wrapper is a strict pass-through.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo, ResponseUsage};

const ENV_ENABLE: &str = "DYN_DECORATOR";
const ENV_TTL_SECS: &str = "DYN_DECORATOR_TTL_SECS";
const ENV_FORFEIT_TTFT_MS: &str = "DYN_DECORATOR_TTFT_FORFEIT_MS";
const ENV_RETRY_AFTER_SECS: &str = "DYN_SHED_RETRY_AFTER_SECS";
const ENV_BLOCK_SIZE: &str = "DYN_KV_CACHE_BLOCK_SIZE";

/// Per-endpoint cap on shadow entries; oldest are swept past this.
const MAX_SHADOW_ENTRIES: usize = 200_000;
/// Emit an aggregate calibration summary every N joined completions.
const SUMMARY_EVERY: u64 = 20;
/// TTL adjustment bounds and steps (multiplicative, applied per signed error).
const TTL_MIN: f64 = 30.0;
const TTL_MAX: f64 = 7200.0;
const TTL_DECAY_ON_OVERPREDICT: f64 = 0.9;
const TTL_GROW_ON_UNDERPREDICT: f64 = 1.05;
/// TTFT EMA smoothing factor.
const EMA_ALPHA: f64 = 0.2;

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok())
}

struct Inflight {
    endpoint: String,
    predicted_tokens: u64,
    prompt_tokens: u64,
    t_pick: Instant,
}

#[derive(Default)]
struct EndpointState {
    /// chained block hash -> last-touch instant
    shadow: HashMap<u64, Instant>,
    ttft_ema_ms: Option<f64>,
}

#[derive(Default)]
struct Stats {
    joined: u64,
    abs_err_tokens: u64,
    over: u64,
    under: u64,
    exact: u64,
}

pub struct CalibratedPicker<P> {
    inner: Arc<P>,
    enabled: bool,
    block_size: usize,
    ttl_secs: Mutex<f64>,
    forfeit_ttft_ms: Option<f64>,
    retry_after_secs: Option<u64>,
    endpoints: Mutex<HashMap<String, EndpointState>>,
    inflight: Mutex<HashMap<String, Inflight>>,
    stats: Mutex<Stats>,
}

impl<P> CalibratedPicker<P> {
    pub fn new(inner: Arc<P>) -> Self {
        let enabled = std::env::var(ENV_ENABLE)
            .map(|v| v.trim() == "calibration")
            .unwrap_or(false);
        let ttl0: f64 = env_parse(ENV_TTL_SECS).unwrap_or(600.0);
        let forfeit_ttft_ms: Option<f64> = env_parse(ENV_FORFEIT_TTFT_MS);
        let retry_after_secs: Option<u64> = env_parse(ENV_RETRY_AFTER_SECS);
        let block_size: usize = env_parse(ENV_BLOCK_SIZE).unwrap_or(128);
        if enabled {
            tracing::info!(
                block_size,
                ttl_secs = ttl0,
                ?forfeit_ttft_ms,
                "CalibratedPicker enabled: shadow-prefix calibration + TTFT viability EMA"
            );
        }
        Self {
            inner,
            enabled,
            block_size,
            ttl_secs: Mutex::new(ttl0),
            forfeit_ttft_ms,
            retry_after_secs,
            endpoints: Mutex::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
            stats: Mutex::new(Stats::default()),
        }
    }

    /// Chained hashes over full token blocks: h_i = fnv(h_{i-1} ++ block_i).
    /// Self-consistent only — never compared against engine block hashes.
    fn block_hashes(&self, token_ids: &[u32]) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(token_ids.len() / self.block_size);
        let mut prev: u64 = 0xcbf2_9ce4_8422_2325;
        for block in token_ids.chunks_exact(self.block_size) {
            let mut h = prev;
            for t in block {
                h ^= *t as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            prev = h;
            hashes.push(h);
        }
        hashes
    }

    /// Predicted cached tokens on `endpoint` = longest live prefix of
    /// `hashes` present in its shadow (monotone boundary, like the real
    /// index), then touch/insert all hashes for this request.
    fn predict_and_insert(&self, endpoint: &str, hashes: &[u64]) -> u64 {
        let now = Instant::now();
        let ttl = Duration::from_secs_f64(*self.ttl_secs.lock().unwrap());
        let mut eps = self.endpoints.lock().unwrap();
        let st = eps.entry(endpoint.to_string()).or_default();
        let mut live_prefix = 0usize;
        for h in hashes {
            match st.shadow.get(h) {
                Some(touched) if now.duration_since(*touched) < ttl => live_prefix += 1,
                _ => break,
            }
        }
        for h in hashes {
            st.shadow.insert(*h, now);
        }
        if st.shadow.len() > MAX_SHADOW_ENTRIES {
            st.shadow.retain(|_, touched| now.duration_since(*touched) < ttl);
        }
        (live_prefix * self.block_size) as u64
    }

    /// Forfeit gate: Saturated when EVERY endpoint has a TTFT EMA and all of
    /// them exceed the threshold. Endpoints without data keep the gate open
    /// (cold start must not self-block — same stance as #11865's thresholds).
    fn forfeit_if_unviable(&self, candidates: &[Endpoint]) -> Option<PickError> {
        let threshold = self.forfeit_ttft_ms?;
        let eps = self.endpoints.lock().unwrap();
        let mut seen = 0usize;
        for ep in candidates {
            let key = ep.address_port();
            match eps.get(&key).and_then(|s| s.ttft_ema_ms) {
                Some(ema) if ema > threshold => seen += 1,
                _ => return None,
            }
        }
        if seen == 0 {
            return None;
        }
        tracing::info!(
            threshold_ms = threshold,
            endpoints = seen,
            "ForfeitGate: all endpoints over TTFT viability threshold; forfeiting"
        );
        Some(PickError::Saturated {
            retry_after_secs: self.retry_after_secs,
        })
    }
}

#[tonic::async_trait]
impl<P: EndpointPicker> EndpointPicker for CalibratedPicker<P> {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if !self.enabled {
            return self.inner.pick(req, endpoints).await;
        }
        if let Some(shed) = self.forfeit_if_unviable(endpoints) {
            return Err(shed);
        }
        let result = self.inner.pick(req, endpoints).await?;
        if let Some(token_ids) = &result.token_ids {
            let hashes = self.block_hashes(token_ids);
            let predicted = self.predict_and_insert(&result.endpoint, &hashes);
            tracing::debug!(
                request_id = %req.request_id,
                endpoint = %result.endpoint,
                prompt_tokens = token_ids.len(),
                predicted_cached_tokens = predicted,
                "calibration: prediction recorded"
            );
            self.inflight.lock().unwrap().insert(
                req.request_id.clone(),
                Inflight {
                    endpoint: result.endpoint.clone(),
                    predicted_tokens: predicted,
                    prompt_tokens: token_ids.len() as u64,
                    t_pick: Instant::now(),
                },
            );
        }
        Ok(result)
    }

    async fn on_prefill_complete(&self, request_id: &str) {
        if self.enabled
            && let Some(inflight) = self.inflight.lock().unwrap().get(request_id)
        {
            let ttft_ms = inflight.t_pick.elapsed().as_secs_f64() * 1000.0;
            let mut eps = self.endpoints.lock().unwrap();
            let st = eps.entry(inflight.endpoint.clone()).or_default();
            let ema = match st.ttft_ema_ms {
                Some(prev) => prev + EMA_ALPHA * (ttft_ms - prev),
                None => ttft_ms,
            };
            st.ttft_ema_ms = Some(ema);
            tracing::debug!(
                request_id,
                endpoint = %inflight.endpoint,
                ttft_ms,
                ttft_ema_ms = ema,
                "viability: TTFT observed"
            );
        }
        self.inner.on_prefill_complete(request_id).await;
    }

    async fn on_request_complete(&self, request_id: &str, usage: Option<ResponseUsage>) {
        if self.enabled
            && let Some(inflight) = self.inflight.lock().unwrap().remove(request_id)
            && let Some(actual) = usage.as_ref().and_then(|u| u.cached_tokens)
        {
            let predicted = inflight.predicted_tokens;
            let err = predicted as i64 - actual as i64;
            let ttl_now = {
                let mut ttl = self.ttl_secs.lock().unwrap();
                if err > 0 {
                    *ttl = (*ttl * TTL_DECAY_ON_OVERPREDICT).clamp(TTL_MIN, TTL_MAX);
                } else if err < 0 {
                    *ttl = (*ttl * TTL_GROW_ON_UNDERPREDICT).clamp(TTL_MIN, TTL_MAX);
                }
                *ttl
            };
            tracing::info!(
                request_id,
                endpoint = %inflight.endpoint,
                prompt_tokens = inflight.prompt_tokens,
                predicted_cached_tokens = predicted,
                actual_cached_tokens = actual,
                error_tokens = err,
                shadow_ttl_secs = format!("{ttl_now:.0}"),
                "calibration: joined prediction with ground truth"
            );
            let mut stats = self.stats.lock().unwrap();
            stats.joined += 1;
            stats.abs_err_tokens += err.unsigned_abs();
            match err.cmp(&0) {
                std::cmp::Ordering::Greater => stats.over += 1,
                std::cmp::Ordering::Less => stats.under += 1,
                std::cmp::Ordering::Equal => stats.exact += 1,
            }
            if stats.joined.is_multiple_of(SUMMARY_EVERY) {
                tracing::info!(
                    joined = stats.joined,
                    exact = stats.exact,
                    over = stats.over,
                    under = stats.under,
                    mean_abs_err_tokens = stats.abs_err_tokens as f64 / stats.joined as f64,
                    shadow_ttl_secs = format!("{ttl_now:.0}"),
                    "calibration: summary"
                );
            }
        }
        self.inner.on_request_complete(request_id, usage).await;
    }
}
