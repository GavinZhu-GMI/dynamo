#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Relay-fed prefix affinity for agg pool selection.

One colocated KV DC Relay per agg pool (scoped to the pool's Dynamo namespace,
with the namespace doubling as the relay's dc_id) feeds one CKF lane in an
in-process KvDcRelayConsumer. At request time the router prefix-matches the
request's token_ids against every lane and, when one pool's matched depth
clears the configured threshold and strictly beats the runner-up, overrides
the SLA-grid pool choice with that pool.

This is the "relay" lane feed kind; pools without a relay (opaque pools) are
"none" and are never scored here — they keep forfeit-only semantics.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# First N select() calls are logged at INFO so bring-up shows the raw per-lane
# matches without enabling debug logging fleet-wide.
_SELECT_LOG_CALLS = 8


@dataclass
class RelayAffinityConfig:
    """Config for relay-fed prefix affinity (``relay_affinity`` config key)."""

    enabled: bool = False
    # KV block size of the pool engines. Must match the workers' page size or
    # hashes never join; there is no server-side validation of this value.
    kv_block_size: int = 0
    # Minimum matched prefix depth (in KV blocks) before affinity may override
    # the SLA grid.
    min_match_blocks: int = 4
    # Depth lead (in KV blocks) the best pool must have over the runner-up;
    # ties fall through to the grid so the SLA targets keep deciding.
    min_lead_blocks: int = 1
    consumer_instance: int = 1

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.kv_block_size <= 0:
            raise ValueError(
                "relay_affinity.kv_block_size must be positive when enabled"
            )
        if self.min_match_blocks <= 0:
            raise ValueError(
                "relay_affinity.min_match_blocks must be positive when enabled"
            )
        if self.min_lead_blocks < 0:
            raise ValueError("relay_affinity.min_lead_blocks must be non-negative")


class RelayAffinity:
    """Owns per-pool relays plus the consumer and scores pools by prefix depth."""

    def __init__(self, config: RelayAffinityConfig, namespaces: List[str]):
        config.validate()
        if not config.enabled:
            raise ValueError("RelayAffinity constructed with disabled config")
        self.config = config
        self.namespaces = namespaces
        self._relays: List[Any] = []
        self._consumer: Optional[Any] = None
        self._select_calls = 0
        self._stats_task: Optional[asyncio.Task] = None

    async def start(self, runtime: Any) -> None:
        """Start one relay per pool namespace, then the consumer over all lanes."""
        from dynamo.llm import KvDcRelay, KvDcRelayConsumer

        for namespace in self.namespaces:
            endpoint = runtime.endpoint(f"{namespace}.router.generate")
            relay = KvDcRelay(endpoint, namespace, namespace_filter=namespace)
            await relay.start()
            self._relays.append(relay)
            logger.info(f"Relay lane started for pool namespace '{namespace}'")
        consumer = KvDcRelayConsumer(
            self._relays,
            list(self.namespaces),
            consumer_instance=self.config.consumer_instance,
        )
        await consumer.start()
        self._consumer = consumer
        logger.info(
            f"Relay consumer started over {len(self.namespaces)} pool lanes "
            f"(kv_block_size={self.config.kv_block_size})"
        )
        # Diagnostic builds (ckf-diagnostics feature) expose relay.stats();
        # log per-lane aggregation counters periodically so bring-up can see
        # whether worker events are reaching each relay at all.
        if self._relays and hasattr(self._relays[0], "stats"):
            self._stats_task = asyncio.create_task(self._log_stats_loop())

    async def _log_stats_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                for namespace, relay in zip(self.namespaces, self._relays):
                    stats = await relay.stats()
                    for endpoint in stats["endpoints"]:
                        # aggregation/publication are None until the endpoint
                        # actor materializes.
                        agg = endpoint["aggregation"]
                        pub = endpoint["publication"]
                        logger.info(
                            "RELAY-STATS pool=%s endpoint=%s lifecycle=%s "
                            "members=%s contributions=%s unique_blocks=%s "
                            "publication=%s",
                            namespace,
                            endpoint["serving_endpoint"],
                            endpoint["lifecycle"],
                            {m["worker_id"]: m["blocks"] for m in agg["members"]}
                            if agg
                            else None,
                            agg["contribution_count"] if agg else None,
                            agg["unique_block_count"] if agg else None,
                            pub,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("relay stats loop iteration failed")

    async def shutdown(self) -> None:
        if self._stats_task is not None:
            self._stats_task.cancel()
            self._stats_task = None
        if self._consumer is not None:
            await self._consumer.shutdown()
            self._consumer = None
        for relay in self._relays:
            await relay.shutdown()
        self._relays = []

    def select(
        self, token_ids: List[int]
    ) -> Tuple[Optional[int], List[Dict[str, Any]]]:
        """Return (pool index to override with, raw per-lane matches).

        The override is None when the consumer is not started, no lane is
        ready, the best depth is under ``min_match_blocks``, or the best pool
        does not lead the runner-up by ``min_lead_blocks``.
        """
        if self._consumer is None:
            return None, []
        matches = self._consumer.find_matches(token_ids, self.config.kv_block_size)
        if self._select_calls < _SELECT_LOG_CALLS:
            self._select_calls += 1
            logger.info(
                "relay affinity select #%s: isl=%s blocks=%s matches=%s",
                self._select_calls,
                len(token_ids),
                len(token_ids) // max(self.config.kv_block_size, 1),
                matches,
            )
        ready = [
            (idx, match["prefix_depth"])
            for idx, match in enumerate(matches)
            if match["ready"]
        ]
        if not ready:
            return None, matches
        ready.sort(key=lambda item: item[1], reverse=True)
        best_idx, best_depth = ready[0]
        runner_up_depth = ready[1][1] if len(ready) > 1 else 0
        if best_depth < self.config.min_match_blocks:
            return None, matches
        if best_depth - runner_up_depth < self.config.min_lead_blocks:
            return None, matches
        return best_idx, matches
