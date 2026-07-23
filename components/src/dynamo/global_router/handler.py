#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Handler for hierarchical routing to worker pools.

Supports two modes:
- "disagg": Routes prefill and decode requests to separate pool types
  based on (ISL, TTFT) and (context_length, ITL) respectively.
- "agg": Routes generate requests to unified pools that handle both
  prefill and decode, based on (TTFT, ITL) or optionally (ISL, TTFT, ITL).

Both modes support priority-based pool overrides from agent hints.
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from dynamo.runtime import Client, DistributedRuntime

from .opaque_pool import OpaquePoolClient, PoolLatencyTracker, make_opaque_clients
from .pool_selection import get_priority_retry_order, load_config

logger = logging.getLogger(__name__)


class GlobalRouterHandler:
    """
    Handler for the Global Router that routes requests to worker pools.

    The global router sits between the frontend and local routers. It:
    - In disagg mode: routes prefill/decode requests to separate pool types
    - In agg mode: routes generate requests to unified pools
    - Uses grid-based selection strategy from config to choose pools
    - Supports priority-based pool overrides from agent hints
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        config_path: str,
        model_name: str,
        default_ttft_target_ms: Optional[float] = None,
        default_itl_target_ms: Optional[float] = None,
    ):
        self.runtime = runtime
        self.config = load_config(config_path)
        self.model_name = model_name
        self.default_ttft_target_ms = default_ttft_target_ms
        self.default_itl_target_ms = default_itl_target_ms

        # Clients to local routers in each pool namespace
        # Will be populated in initialize()
        self.prefill_clients: Dict[str, Client] = {}
        self.decode_clients: Dict[str, Client] = {}
        self.agg_clients: Dict[str, Client] = {}

        # Opaque lane (agg mode only): observed-TTFT trackers per relay-backed
        # pool drive forfeit viability; opaque clients populated in initialize()
        self.agg_latency_trackers: Dict[str, PoolLatencyTracker] = {}
        self.opaque_clients: List[OpaquePoolClient] = []
        # Relay-fed prefix affinity; attached by the entrypoint after
        # initialize() when config.relay_affinity.enabled (agg mode only).
        self.relay_affinity: Optional[Any] = None

        if self.config.mode == "disagg":
            assert self.config.prefill_pool_dynamo_namespaces is not None
            assert self.config.decode_pool_dynamo_namespaces is not None
            self.prefill_namespace_to_idx: Dict[str, int] = {
                ns: idx
                for idx, ns in enumerate(self.config.prefill_pool_dynamo_namespaces)
            }
            self.decode_namespace_to_idx: Dict[str, int] = {
                ns: idx
                for idx, ns in enumerate(self.config.decode_pool_dynamo_namespaces)
            }
        elif self.config.mode == "agg":
            assert self.config.agg_pool_dynamo_namespaces is not None
            self.agg_namespace_to_idx: Dict[str, int] = {
                ns: idx for idx, ns in enumerate(self.config.agg_pool_dynamo_namespaces)
            }
            for ns in self.config.agg_pool_dynamo_namespaces:
                self.agg_latency_trackers[ns] = PoolLatencyTracker(
                    alpha=self.config.forfeit.ttft_ema_alpha
                )

    async def _forward_with_priority_retry(
        self,
        request: Dict[str, Any],
        request_type: str,
        initial_pool_idx: int,
        namespaces: List[str],
        clients: Dict[str, Client],
        pool_priorities: List[int],
        latency_trackers: Optional[Dict[str, PoolLatencyTracker]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Forward a request to the selected pool and retry faster pools on failure.

        Retry order is controlled by pool priorities: lower priority numbers are
        faster, so retries walk from the selected pool toward faster pools.

        When ``latency_trackers`` is provided, the observed TTFT of the pool
        that actually serves the request is recorded (forfeit viability signal).
        """
        pool_order = get_priority_retry_order(
            selected_pool=initial_pool_idx,
            pool_priorities=pool_priorities,
            enable_priority_retry=self.config.enable_priority_retry,
        )

        for attempt_idx, pool_idx in enumerate(pool_order):
            namespace = namespaces[pool_idx]
            client = clients[namespace]
            yielded_output = False
            start = time.monotonic()

            try:
                stream = await client.generate(request)
                async for output in stream:
                    if not yielded_output and latency_trackers is not None:
                        latency_trackers[namespace].record(
                            (time.monotonic() - start) * 1000.0
                        )
                    yielded_output = True
                    data = output.data() if hasattr(output, "data") else output
                    yield data
                return
            except Exception as e:
                is_last_attempt = attempt_idx == len(pool_order) - 1
                if yielded_output:
                    logger.error(
                        f"Error forwarding {request_type} request to {namespace} "
                        f"after streaming started; cannot safely retry: {e}"
                    )
                    raise

                if is_last_attempt:
                    logger.error(
                        f"Error forwarding {request_type} request to {namespace}; "
                        f"no priority retry pools remain: {e}"
                    )
                    raise

                next_pool_idx = pool_order[attempt_idx + 1]
                next_namespace = namespaces[next_pool_idx]
                if request_type == "decode":
                    # A failed decode attempt may already have caused the prefill
                    # engine to retire or drop its KV cache for this request. That
                    # is acceptable: the retry is a fresh decode request to another
                    # pool. The backend should handle any cache miss normally, but
                    # current Dynamo backends do not support this yet.
                    logger.debug("Retrying decode request after pool failure")
                logger.warning(
                    f"Error forwarding {request_type} request to pool {pool_idx} "
                    f"({namespace}): {e}; retrying faster pool {next_pool_idx} "
                    f"({next_namespace})"
                )

    async def initialize(self) -> None:
        """
        Initialize clients to all local routers.

        This connects to the local router in each pool's namespace.
        Local routers are expected at: {namespace}.router.generate
        """
        logger.info(f"Initializing Global Router Handler (mode={self.config.mode})...")

        if self.config.mode == "disagg":
            await self._initialize_disagg()
        elif self.config.mode == "agg":
            await self._initialize_agg()

    async def _initialize_disagg(self) -> None:
        """Initialize disagg mode clients to prefill and decode pools."""
        assert self.config.prefill_pool_dynamo_namespaces is not None
        assert self.config.decode_pool_dynamo_namespaces is not None

        # Connect to prefill pool local routers
        for idx, namespace in enumerate(self.config.prefill_pool_dynamo_namespaces):
            try:
                endpoint = self.runtime.endpoint(f"{namespace}.router.generate")
                client = await endpoint.client()
                self.prefill_clients[namespace] = client
                logger.info(
                    f"Connected to prefill pool {idx}: {namespace}.router.generate"
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to prefill pool {idx} ({namespace}): {e}"
                )
                raise

        # Connect to decode pool local routers
        for idx, namespace in enumerate(self.config.decode_pool_dynamo_namespaces):
            try:
                endpoint = self.runtime.endpoint(f"{namespace}.router.generate")
                client = await endpoint.client()
                self.decode_clients[namespace] = client
                logger.info(
                    f"Connected to decode pool {idx}: {namespace}.router.generate"
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to decode pool {idx} ({namespace}): {e}"
                )
                raise

        logger.info(
            f"Global Router initialized (disagg): {len(self.prefill_clients)} prefill pools, "
            f"{len(self.decode_clients)} decode pools"
        )

    async def _initialize_agg(self) -> None:
        """Initialize agg mode clients to unified pools."""
        assert self.config.agg_pool_dynamo_namespaces is not None

        for idx, namespace in enumerate(self.config.agg_pool_dynamo_namespaces):
            try:
                endpoint = self.runtime.endpoint(f"{namespace}.router.generate")
                client = await endpoint.client()
                self.agg_clients[namespace] = client
                logger.info(f"Connected to agg pool {idx}: {namespace}.router.generate")
            except Exception as e:
                logger.error(f"Failed to connect to agg pool {idx} ({namespace}): {e}")
                raise

        self.opaque_clients = make_opaque_clients(
            self.config.opaque_pools,
            self.model_name,
            self.config.forfeit.ttft_ema_alpha,
            kv_block_size=self.config.relay_affinity.kv_block_size
            if self.config.relay_affinity.enabled
            else None,
        )
        for client in self.opaque_clients:
            logger.info(
                f"Registered opaque pool '{client.config.name}' at "
                f"{client.config.url} (forfeit-only)"
            )

        logger.info(f"Global Router initialized (agg): {len(self.agg_clients)} pools")

    async def handle_prefill(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle prefill requests from the frontend (disagg mode).

        Selects the appropriate prefill pool based on ISL, TTFT target,
        and optional priority, then forwards the request to the local
        router in that pool.
        """
        assert self.config.prefill_pool_selection_strategy is not None
        assert self.config.prefill_pool_dynamo_namespaces is not None

        # Extract ISL (input sequence length)
        token_ids = request.get("token_ids", [])
        isl = len(token_ids)

        # Extract TTFT target from nvext.router (forwarded by the preprocessor
        # as the `router` field on PreprocessedRequest), fallback to CLI default.
        # Use `is None` to preserve explicit 0 (routes to the fastest bucket).
        router_params = request.get("router") or {}
        ttft_target_ms = router_params.get("ttft_target")
        if ttft_target_ms is None:
            ttft_target_ms = self.default_ttft_target_ms

        # Extract priority from routing hints (set by nvext.agent_hints.priority)
        routing = request.get("routing") or {}
        priority = routing.get("priority")

        # Select prefill pool
        pool_idx = self.config.prefill_pool_selection_strategy.select_pool(
            isl=isl, ttft_target_ms=ttft_target_ms, priority=priority
        )
        namespace = self.config.prefill_pool_dynamo_namespaces[pool_idx]
        assert self.config.prefill_pool_priorities is not None
        pool_order = get_priority_retry_order(
            selected_pool=pool_idx,
            pool_priorities=self.config.prefill_pool_priorities,
            enable_priority_retry=self.config.enable_priority_retry,
        )

        logger.info(
            f"Routing prefill request: ISL={isl}, TTFT_target={ttft_target_ms}ms, "
            f"priority={priority} -> pool {pool_idx} ({namespace}); "
            f"retry_order={pool_order}"
        )

        # Forward request to local router and stream back responses
        async for data in self._forward_with_priority_retry(
            request=request,
            request_type="prefill",
            initial_pool_idx=pool_idx,
            namespaces=self.config.prefill_pool_dynamo_namespaces,
            clients=self.prefill_clients,
            pool_priorities=self.config.prefill_pool_priorities,
        ):
            yield data

    async def handle_decode(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle decode requests from the frontend (disagg mode).

        Selects the appropriate decode pool based on context length, ITL target,
        and optional priority, then forwards the request to the local
        router in that pool.
        """
        assert self.config.decode_pool_selection_strategy is not None
        assert self.config.decode_pool_dynamo_namespaces is not None

        # Extract context length (input tokens + any previously generated)
        token_ids = request.get("token_ids", [])
        # context_length should be averaged ISL + OSL // 2
        # TODO: predict OSL based on ISL
        context_length = len(token_ids)

        router_params = request.get("router") or {}
        itl_target_ms = router_params.get("itl_target")
        if itl_target_ms is None:
            itl_target_ms = self.default_itl_target_ms

        # Extract priority from routing hints (set by nvext.agent_hints.priority)
        routing = request.get("routing") or {}
        priority = routing.get("priority")

        # Select decode pool
        pool_idx = self.config.decode_pool_selection_strategy.select_pool(
            context_length=context_length,
            itl_target_ms=itl_target_ms,
            priority=priority,
        )
        namespace = self.config.decode_pool_dynamo_namespaces[pool_idx]
        assert self.config.decode_pool_priorities is not None
        pool_order = get_priority_retry_order(
            selected_pool=pool_idx,
            pool_priorities=self.config.decode_pool_priorities,
            enable_priority_retry=self.config.enable_priority_retry,
        )

        logger.info(
            f"Routing decode request: context_length={context_length}, "
            f"ITL_target={itl_target_ms}ms, priority={priority} -> "
            f"pool {pool_idx} ({namespace}); retry_order={pool_order}"
        )

        # Forward request to local router and stream back responses
        async for data in self._forward_with_priority_retry(
            request=request,
            request_type="decode",
            initial_pool_idx=pool_idx,
            namespaces=self.config.decode_pool_dynamo_namespaces,
            clients=self.decode_clients,
            pool_priorities=self.config.decode_pool_priorities,
        ):
            yield data

    async def handle_generate(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle generate requests (agg mode).

        Selects the appropriate agg pool based on TTFT target, ITL target,
        optional ISL, and optional priority, then forwards the request to the
        local router in that pool. The pool's workers handle both prefill and
        decode.
        """
        assert self.config.agg_pool_selection_strategy is not None
        assert self.config.agg_pool_dynamo_namespaces is not None

        token_ids = request.get("token_ids", [])
        isl = len(token_ids)

        # Extract SLA targets from nvext.router (forwarded by the preprocessor
        # as the `router` field on PreprocessedRequest), fallback to CLI defaults.
        # Use `is None` checks to preserve explicit 0 values.
        router_params = request.get("router") or {}
        ttft_target_ms = router_params.get("ttft_target")
        if ttft_target_ms is None:
            ttft_target_ms = self.default_ttft_target_ms
        itl_target_ms = router_params.get("itl_target")
        if itl_target_ms is None:
            itl_target_ms = self.default_itl_target_ms

        # Extract priority from routing hints (set by nvext.agent_hints.priority)
        routing = request.get("routing") or {}
        priority = routing.get("priority")

        # Select agg pool
        pool_idx = self.config.agg_pool_selection_strategy.select_pool(
            ttft_target_ms=ttft_target_ms,
            itl_target_ms=itl_target_ms,
            priority=priority,
            isl=isl,
        )
        namespace = self.config.agg_pool_dynamo_namespaces[pool_idx]
        assert self.config.agg_pool_priorities is not None

        # Relay-fed prefix affinity: when one relay-backed pool holds a
        # decisively deeper cached prefix for this request, prefer it over the
        # SLA-grid choice. Viability/forfeit checks below still apply to the
        # overridden pool. Opaque pools have no relay lane and are never
        # considered here.
        if self.relay_affinity is not None:
            affinity_idx, matches = self.relay_affinity.select(token_ids)
            if affinity_idx is not None and affinity_idx != pool_idx:
                logger.info(
                    "AFFINITY: relay prefix match overrides grid pool %s -> %s "
                    "(matches=%s)",
                    pool_idx,
                    affinity_idx,
                    matches,
                )
                pool_idx = affinity_idx
                namespace = self.config.agg_pool_dynamo_namespaces[pool_idx]

            # Approx lanes: a trusted opaque shadow competes against the relay
            # depths under the same thresholds. Forfeit semantics are separate
            # and unchanged; this is a positive, calibrated choice.
            approx = self._select_approx_opaque(token_ids, matches)
            if approx is not None:
                client, depth, relay_best = approx
                logger.info(
                    "APPROX-AFFINITY: opaque pool '%s' shadow depth %s beats "
                    "relay best %s; routing to opaque",
                    client.config.name,
                    depth,
                    relay_best,
                )
                async for data in client.generate(request):
                    yield data
                return

        # Opaque lane: forfeit-only semantics. The opaque pool is chosen only
        # when EVERY relay-backed pool is non-viable — never by score
        # comparison against fabricated overlap.
        if self.opaque_clients:
            viable = self._viable_agg_pools(ttft_target_ms)
            if not viable:
                opaque = next((c for c in self.opaque_clients if c.healthy()), None)
                if opaque is not None:
                    logger.info(
                        f"FORFEIT: all {len(self.agg_clients)} relay-backed pools "
                        f"non-viable (TTFT_target={ttft_target_ms}ms) -> "
                        f"opaque pool '{opaque.config.name}'"
                    )
                    async for data in opaque.generate(request):
                        yield data
                    return
                logger.warning(
                    "All relay-backed pools non-viable and no healthy opaque "
                    "pool; falling back to grid selection"
                )
            elif pool_idx not in viable:
                fallback = min(
                    viable, key=lambda idx: self.config.agg_pool_priorities[idx]
                )
                logger.info(
                    f"Grid-selected pool {pool_idx} non-viable; rerouting to "
                    f"viable pool {fallback}"
                )
                pool_idx = fallback
                namespace = self.config.agg_pool_dynamo_namespaces[pool_idx]

        pool_order = get_priority_retry_order(
            selected_pool=pool_idx,
            pool_priorities=self.config.agg_pool_priorities,
            enable_priority_retry=self.config.enable_priority_retry,
        )

        logger.info(
            "Routing agg request: ISL=%s, TTFT_target=%sms, ITL_target=%sms, "
            "priority=%s -> pool %s (%s); retry_order=%s",
            isl,
            ttft_target_ms,
            itl_target_ms,
            priority,
            pool_idx,
            namespace,
            pool_order,
        )

        # Forward request to local router and stream back responses
        async for data in self._forward_with_priority_retry(
            request=request,
            request_type="agg",
            initial_pool_idx=pool_idx,
            namespaces=self.config.agg_pool_dynamo_namespaces,
            clients=self.agg_clients,
            pool_priorities=self.config.agg_pool_priorities,
            latency_trackers=self.agg_latency_trackers,
        ):
            yield data

    def _select_approx_opaque(
        self, token_ids: List[int], relay_matches: List[Dict[str, Any]]
    ):
        """Best trusted approx-feed opaque pool, if it decisively beats relay.

        Returns (client, depth, relay_best_depth) or None. Thresholds are the
        relay-affinity ones so 'approx beats relay' means the same thing as
        'relay beats relay'.
        """
        if not self.opaque_clients or self.relay_affinity is None:
            return None
        best = None
        for client in self.opaque_clients:
            depth = client.approx_depth(token_ids)
            if depth is None:
                continue
            if best is None or depth > best[1]:
                best = (client, depth)
        if best is None:
            return None
        client, depth = best
        cfg = self.relay_affinity.config
        relay_best = max(
            (m["prefix_depth"] for m in relay_matches if m["ready"]), default=0
        )
        if depth < cfg.min_match_blocks:
            return None
        if depth - relay_best < cfg.min_lead_blocks:
            return None
        return client, depth, relay_best

    def _viable_agg_pools(self, ttft_target_ms: Optional[float]) -> List[int]:
        """Return indices of relay-backed agg pools considered viable.

        A pool is viable during its cold-start window (fewer than
        ``forfeit.min_samples`` observed requests) or while its observed TTFT
        EMA is within ``forfeit.ttft_viability_factor`` x the TTFT target.
        """
        assert self.config.agg_pool_dynamo_namespaces is not None
        assert self.config.agg_pool_selection_strategy is not None
        strategy = self.config.agg_pool_selection_strategy
        if ttft_target_ms is None:
            ttft_target_ms = (strategy.ttft_min_ms + strategy.ttft_max_ms) / 2

        forfeit = self.config.forfeit
        viable: List[int] = []
        for idx, namespace in enumerate(self.config.agg_pool_dynamo_namespaces):
            tracker = self.agg_latency_trackers[namespace]
            if tracker.samples < forfeit.min_samples:
                viable.append(idx)
                continue
            assert tracker.ema_ttft_ms is not None
            if tracker.ema_ttft_ms <= ttft_target_ms * forfeit.ttft_viability_factor:
                viable.append(idx)
        return viable

    def get_pool_info(self) -> Dict[str, Any]:
        """Get information about connected pools for debugging/monitoring."""
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "mode": self.config.mode,
        }
        if self.config.mode == "disagg":
            info.update(
                {
                    "num_prefill_pools": self.config.num_prefill_pools,
                    "num_decode_pools": self.config.num_decode_pools,
                    "prefill_pools": self.config.prefill_pool_dynamo_namespaces,
                    "decode_pools": self.config.decode_pool_dynamo_namespaces,
                    "prefill_pool_priorities": self.config.prefill_pool_priorities,
                    "decode_pool_priorities": self.config.decode_pool_priorities,
                    "enable_priority_retry": self.config.enable_priority_retry,
                    "prefill_connected": list(self.prefill_clients.keys()),
                    "decode_connected": list(self.decode_clients.keys()),
                }
            )
        elif self.config.mode == "agg":
            info.update(
                {
                    "num_agg_pools": self.config.num_agg_pools,
                    "agg_pools": self.config.agg_pool_dynamo_namespaces,
                    "agg_pool_priorities": self.config.agg_pool_priorities,
                    "enable_priority_retry": self.config.enable_priority_retry,
                    "agg_connected": list(self.agg_clients.keys()),
                    "agg_pool_ttft_ema_ms": {
                        ns: tracker.ema_ttft_ms
                        for ns, tracker in self.agg_latency_trackers.items()
                    },
                    "opaque_pools": [
                        client.stats() for client in self.opaque_clients
                    ],
                }
            )
        return info
