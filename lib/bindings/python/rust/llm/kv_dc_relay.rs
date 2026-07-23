// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::OnceCell;

use super::*;
use crate::{Endpoint, to_pyerr};

#[pyclass]
pub struct KvDcRelay {
    endpoint: dynamo_runtime::component::Endpoint,
    dc_id: String,
    namespace_filter: Option<String>,
    endpoint_prefix: Option<String>,
    publication_threshold: usize,
    publication_delay_ms: u64,
    recovery_attempt_timeout_ms: u64,
    inner: Arc<OnceCell<Arc<llm_rs::kv_dc_relay::KvDcRelay>>>,
}

#[pymethods]
impl KvDcRelay {
    #[new]
    #[pyo3(signature = (endpoint, dc_id, namespace_filter=None, endpoint_prefix=None, publication_threshold=16, publication_delay_ms=1, recovery_attempt_timeout_ms=30_000))]
    fn new(
        endpoint: Endpoint,
        dc_id: String,
        namespace_filter: Option<String>,
        endpoint_prefix: Option<String>,
        publication_threshold: usize,
        publication_delay_ms: u64,
        recovery_attempt_timeout_ms: u64,
    ) -> Self {
        Self {
            endpoint: endpoint.inner,
            dc_id,
            namespace_filter,
            endpoint_prefix,
            publication_threshold,
            publication_delay_ms,
            recovery_attempt_timeout_ms,
            inner: Arc::new(OnceCell::new()),
        }
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let endpoint = self.endpoint.clone();
        let dc_id = self.dc_id.clone();
        let namespace_filter = self.namespace_filter.clone();
        let endpoint_prefix = self.endpoint_prefix.clone();
        let publication_threshold = self.publication_threshold;
        let publication_delay_ms = self.publication_delay_ms;
        let recovery_attempt_timeout_ms = self.recovery_attempt_timeout_ms;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_or_try_init(|| async move {
                    llm_rs::kv_dc_relay::KvDcRelay::start(
                        endpoint.component().clone(),
                        dc_id,
                        llm_rs::kv_dc_relay::KvDcRelayConfig {
                            namespace_filter,
                            endpoint_prefix,
                            publication_threshold,
                            publication_delay_ms,
                            recovery_attempt_timeout_ms,
                        },
                    )
                    .await
                    .map(Arc::new)
                })
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[cfg(feature = "ckf-diagnostics")]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stats = inner.stats().await.map_err(to_pyerr)?;
            Python::with_gil(|py| {
                pythonize::pythonize(py, &stats)
                    .map(|value| value.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let health = inner.health().await;
            Python::with_gil(|py| {
                pythonize::pythonize(py, &health)
                    .map(|value| value.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.flush().await.map_err(to_pyerr)
        })
    }

    #[cfg(feature = "ckf-diagnostics")]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        serving_endpoint: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let endpoint = dynamo_runtime::protocols::EndpointId::from(serving_endpoint.as_str());
            let diagnostic = inner
                .diagnostic_snapshot(&endpoint)
                .await
                .map_err(to_pyerr)?;
            Python::with_gil(|py| {
                pythonize::pythonize(py, &diagnostic)
                    .map(|value| value.unbind())
                    .map_err(to_pyerr)
            })
        })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.shutdown().await.map_err(to_pyerr)
        })
    }
}

impl KvDcRelay {
    fn started(&self) -> PyResult<Arc<llm_rs::kv_dc_relay::KvDcRelay>> {
        self.inner
            .get()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("KvDcRelay.start() must complete first"))
    }
}

/// In-process consumer over one or more colocated relays: each relay is one pool lane.
/// Construct with parallel `relays`/`labels` lists after every relay's `start()` has
/// completed, then `start()` the consumer and query `find_matches` synchronously.
#[pyclass]
pub struct KvDcRelayConsumer {
    relays: Vec<(String, Arc<llm_rs::kv_dc_relay::KvDcRelay>)>,
    consumer_instance: u64,
    inner: Arc<OnceCell<Arc<llm_rs::kv_dc_relay::KvDcRelayConsumer>>>,
}

#[pymethods]
impl KvDcRelayConsumer {
    #[new]
    #[pyo3(signature = (relays, labels, consumer_instance=1))]
    fn new(
        relays: Vec<PyRef<'_, KvDcRelay>>,
        labels: Vec<String>,
        consumer_instance: u64,
    ) -> PyResult<Self> {
        if relays.len() != labels.len() || relays.is_empty() {
            return Err(PyRuntimeError::new_err(
                "relays and labels must be non-empty lists of equal length",
            ));
        }
        let relays = labels
            .into_iter()
            .zip(relays.iter())
            .map(|(label, relay)| relay.started().map(|inner| (label, inner)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            relays,
            consumer_instance,
            inner: Arc::new(OnceCell::new()),
        })
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let relays = self.relays.clone();
        let consumer_instance = self.consumer_instance;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_or_try_init(|| async move {
                    let sources = relays
                        .into_iter()
                        .map(|(label, relay)| llm_rs::kv_dc_relay::RelayConsumerSource {
                            label,
                            relay,
                        })
                        .collect();
                    Ok::<_, PyErr>(Arc::new(llm_rs::kv_dc_relay::KvDcRelayConsumer::start(
                        sources,
                        consumer_instance,
                    )))
                })
                .await?;
            Ok(())
        })
    }

    /// Prefix-match `token_ids` against every relay lane. Returns one dict per lane,
    /// in construction order: {"label", "ready", "prefix_depth"} with depth in KV
    /// blocks of `kv_block_size` tokens. Synchronous and lock-free.
    #[pyo3(signature = (token_ids, kv_block_size, lora_name=None, cache_namespace=None))]
    fn find_matches(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        kv_block_size: u32,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        if kv_block_size == 0 {
            return Err(PyRuntimeError::new_err("kv_block_size cannot be 0"));
        }
        let consumer = self.consumer_started()?;
        let hashes = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &token_ids,
            kv_block_size,
            dynamo_kv_router::protocols::BlockHashOptions {
                block_mm_infos: None,
                lora_name: lora_name.as_deref(),
                cache_namespace: cache_namespace.as_deref(),
                is_eagle: None,
            },
        );
        let matches = consumer.find_matches(&hashes);
        pythonize::pythonize(py, &matches)
            .map(|value| value.unbind())
            .map_err(to_pyerr)
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let consumer = self.consumer_started()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            consumer.shutdown().await;
            Ok(())
        })
    }
}

impl KvDcRelayConsumer {
    fn consumer_started(&self) -> PyResult<Arc<llm_rs::kv_dc_relay::KvDcRelayConsumer>> {
        self.inner
            .get()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("KvDcRelayConsumer.start() must complete first"))
    }
}
