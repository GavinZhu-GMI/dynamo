#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Opaque pool client for the Global Router.

An opaque pool is an external OpenAI-compatible endpoint with no relay, no KV
index, and no internal visibility (e.g. a black-box reseller or partner DC).
It is deliberately NOT scored against relay-backed pools — there is no overlap
signal to fabricate. Instead it participates via *forfeit-only* selection:
the handler forwards to an opaque pool only when every relay-backed pool is
non-viable for the request (see GlobalRouterHandler._viable_agg_pools).

Black-box observability kept per pool:
- observed TTFT EMA per ISL bucket (reporting/scoring signal, never compared
  against relay-backed overlap scores)
- health via consecutive first-chunk timeouts / transport failures, with a
  cooldown before the pool is eligible again
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from .pool_selection import OpaquePoolConfig
from .shadow_lane import ShadowLane, ShadowLaneConfig

logger = logging.getLogger(__name__)

# Upper bounds (tokens) of the ISL buckets used for TTFT EMA tracking.
# The last bucket is open-ended.
ISL_BUCKET_BOUNDS = (1024, 8192, 32768)


def isl_bucket(isl: int) -> int:
    """Return the ISL bucket index for an input sequence length."""
    for idx, bound in enumerate(ISL_BUCKET_BOUNDS):
        if isl < bound:
            return idx
    return len(ISL_BUCKET_BOUNDS)


class PoolLatencyTracker:
    """Observed-TTFT EMA for a relay-backed pool (the forfeit viability signal)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema_ttft_ms: Optional[float] = None
        self.samples = 0

    def record(self, ttft_ms: float) -> None:
        self.samples += 1
        if self.ema_ttft_ms is None:
            self.ema_ttft_ms = ttft_ms
        else:
            self.ema_ttft_ms = (
                self.alpha * ttft_ms + (1.0 - self.alpha) * self.ema_ttft_ms
            )


class OpaquePoolClient:
    """Streams completions from an external OpenAI-compatible endpoint.

    The global router receives preprocessed (tokenized) requests, so the prompt
    is reconstructed by detokenizing and each streamed text delta is re-encoded
    into token ids to match the backend response contract
    ({"index": 0, "token_ids": [...]} chunks with finish_reason on the last).
    A production multi-DC opaque lane would forward the original request before
    tokenization; the detokenize round-trip is a testbed shim.
    """

    def __init__(
        self,
        config: OpaquePoolConfig,
        tokenizer,
        ema_alpha: float,
        shadow: Optional[ShadowLane] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.ema_alpha = ema_alpha
        self.shadow = shadow
        self.ttft_ema_by_bucket: Dict[int, float] = {}
        self.consecutive_failures = 0
        self.unhealthy_until = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    def healthy(self) -> bool:
        return time.monotonic() >= self.unhealthy_until

    def approx_depth(self, token_ids: List[int]) -> Optional[int]:
        """Trusted shadow-lane depth in blocks, or None when not scoreable."""
        if self.shadow is None or not self.shadow.trusted() or not self.healthy():
            return None
        return self.shadow.query(token_ids)

    def _record_ttft(self, bucket: int, ttft_ms: float) -> None:
        prev = self.ttft_ema_by_bucket.get(bucket)
        if prev is None:
            self.ttft_ema_by_bucket[bucket] = ttft_ms
        else:
            self.ttft_ema_by_bucket[bucket] = (
                self.ema_alpha * ttft_ms + (1.0 - self.ema_alpha) * prev
            )

    def _record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.config.unhealthy_after_failures:
            self.unhealthy_until = time.monotonic() + self.config.health_cooldown_s
            logger.warning(
                f"Opaque pool '{self.config.name}' marked unhealthy for "
                f"{self.config.health_cooldown_s}s after "
                f"{self.consecutive_failures} consecutive failures"
            )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _build_payload(self, request: Dict[str, Any]) -> Dict[str, Any]:
        token_ids = request["token_ids"]
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "stream": True,
            # Ask for the final usage frame; needed for shadow-lane
            # calibration (cached_tokens) and harmless where unsupported.
            "stream_options": {"include_usage": True},
        }
        if self.config.api == "chat":
            # Chat-only black boxes re-apply their own template; send the
            # user-visible text without special tokens to avoid nesting ours.
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            payload["messages"] = [{"role": "user", "content": text}]
        else:
            payload["prompt"] = self.tokenizer.decode(
                token_ids, skip_special_tokens=False
            )
        stop_conditions = request.get("stop_conditions") or {}
        max_tokens = stop_conditions.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        sampling_options = request.get("sampling_options") or {}
        for key in ("temperature", "top_p"):
            value = sampling_options.get(key)
            if value is not None:
                payload[key] = value
        return payload

    async def generate(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Forward a preprocessed request to the opaque endpoint and stream chunks."""
        payload = self._build_payload(request)
        request_token_ids = request["token_ids"]
        bucket = isl_bucket(len(request_token_ids))
        # Depth we believed at dispatch — the calibration join's prediction.
        predicted_blocks = (
            self.shadow.query(request_token_ids) if self.shadow is not None else 0
        )
        cached_tokens: Optional[int] = None
        session = await self._ensure_session()
        path = "/chat/completions" if self.config.api == "chat" else "/completions"

        start = time.monotonic()
        first_chunk = True
        finish_reason: Optional[str] = None
        try:
            async with session.post(
                f"{self.config.url}{path}",
                json=payload,
                # first-chunk timeout governs connect too: a black-box endpoint
                # that can't even accept the connection within it is unhealthy
                timeout=aiohttp.ClientTimeout(
                    total=None, sock_connect=self.config.first_chunk_timeout_s
                ),
            ) as resp:
                resp.raise_for_status()
                lines = resp.content.__aiter__()
                while True:
                    try:
                        if first_chunk:
                            line = await asyncio.wait_for(
                                lines.__anext__(),
                                timeout=self.config.first_chunk_timeout_s,
                            )
                        else:
                            line = await lines.__anext__()
                    except StopAsyncIteration:
                        break
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:") :].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage")
                    if usage:
                        details = usage.get("prompt_tokens_details") or {}
                        cached_tokens = details.get("cached_tokens")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if first_chunk:
                        ttft_ms = (time.monotonic() - start) * 1000.0
                        self._record_ttft(bucket, ttft_ms)
                        self.consecutive_failures = 0
                        first_chunk = False
                        logger.info(
                            f"Opaque pool '{self.config.name}': first chunk in "
                            f"{ttft_ms:.0f}ms (isl_bucket={bucket})"
                        )
                        if self.shadow is not None:
                            self.shadow.record_egress(request_token_ids)
                    if self.config.api == "chat":
                        text = (choice.get("delta") or {}).get("content")
                    else:
                        text = choice.get("text")
                    if text:
                        token_ids = self.tokenizer.encode(
                            text, add_special_tokens=False
                        )
                        yield {"index": 0, "token_ids": token_ids}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self._record_failure()
            raise RuntimeError(
                f"Opaque pool '{self.config.name}' failed "
                f"({'before' if first_chunk else 'after'} first chunk): {e}"
            ) from e

        if first_chunk:
            # Stream ended without any data: treat as a failure for health.
            self._record_failure()
            raise RuntimeError(
                f"Opaque pool '{self.config.name}' returned an empty stream"
            )

        if self.shadow is not None and cached_tokens is not None:
            self.shadow.observe_usage(predicted_blocks, cached_tokens)
            logger.info(
                "Shadow calibration join for '%s': predicted=%s blocks, "
                "cached_tokens=%s, trusted=%s",
                self.config.name,
                predicted_blocks,
                cached_tokens,
                self.shadow.trusted(),
            )

        yield {"index": 0, "token_ids": [], "finish_reason": finish_reason or "stop"}

    def stats(self) -> Dict[str, Any]:
        stats = {
            "name": self.config.name,
            "url": self.config.url,
            "healthy": self.healthy(),
            "consecutive_failures": self.consecutive_failures,
            "ttft_ema_ms_by_isl_bucket": dict(self.ttft_ema_by_bucket),
        }
        if self.shadow is not None:
            shadow = self.shadow.stats()
            stats["shadow"] = {
                "entries": shadow.entries,
                "recorded_requests": shadow.recorded_requests,
                "calibration_samples": shadow.calibration_samples,
                "mean_abs_err_blocks": shadow.mean_abs_err_blocks,
                "divergences": shadow.divergences,
                "ttl_s": shadow.ttl_s,
                "trusted": self.shadow.trusted(),
            }
        return stats


def make_tokenizer(model_name: str):
    """Load the HF tokenizer used to detokenize prompts / re-encode deltas."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _make_block_hasher(kv_block_size: int):
    """Token-ids -> per-block local hashes, matching relay-lane hashing."""
    from dynamo.llm import compute_block_hash_for_seq

    def hasher(token_ids: List[int]) -> List[int]:
        return compute_block_hash_for_seq(list(token_ids), kv_block_size)

    return hasher


def make_opaque_clients(
    configs: List[OpaquePoolConfig],
    model_name: str,
    ema_alpha: float,
    kv_block_size: Optional[int] = None,
) -> List[OpaquePoolClient]:
    if not configs:
        return []
    tokenizer = make_tokenizer(model_name)
    clients = []
    for cfg in configs:
        shadow = None
        if cfg.feed == "approx":
            # validate() guarantees relay_affinity (and its kv_block_size)
            # is configured whenever an approx feed exists.
            assert kv_block_size is not None and kv_block_size > 0
            shadow = ShadowLane(
                ShadowLaneConfig(**cfg.shadow),
                kv_block_size,
                _make_block_hasher(kv_block_size),
            )
        clients.append(OpaquePoolClient(cfg, tokenizer, ema_alpha, shadow=shadow))
    return clients
