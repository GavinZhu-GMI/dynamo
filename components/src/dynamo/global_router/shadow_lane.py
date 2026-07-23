#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shadow prefix lane for opaque pools (the "approx" lane feed kind).

An opaque pool exposes no relay lane, so the router cannot know what the
black-box backend has cached. What the router DOES know is its own egress:
every prefix it forwarded to that pool. A ShadowLane indexes those prefixes
with a TTL (a stand-in for the backend's unknown eviction behavior) and
answers depth queries in KV blocks, comparable to relay-lane depths.

The TTL guess is calibrated against ground truth when the backend reports
`usage.prompt_tokens_details.cached_tokens`: each response joins the depth we
predicted at dispatch with the tokens the backend actually had cached. The
prediction error drives an online TTL adjustment (over-prediction means the
backend evicts sooner than our TTL, so shrink; under-prediction means it
retains longer, so grow) — the same algorithm as the EPP CalibratedPicker's
shadow index. A lane only becomes `trusted()` (eligible to influence routing)
after enough joins land within tolerance; without usage reporting it stays an
observe-only shadow forever.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1


def _chain(key: int, block_hash: int) -> int:
    """Deterministic prefix-chain key (process-local, not a wire format)."""
    return ((key * _FNV_PRIME) ^ block_hash) & _MASK64


@dataclass
class ShadowLaneConfig:
    ttl_s: float = 600.0
    # Calibration joins required within tolerance before the lane is trusted.
    min_calibration_samples: int = 8
    # Tolerated mean absolute prediction error, in KV blocks.
    max_err_blocks: float = 2.0
    # Multiplicative online TTL adjustment per divergent join.
    ttl_adjust_factor: float = 1.25
    ttl_min_s: float = 30.0
    ttl_max_factor: float = 4.0
    # Hard cap on stored prefix entries; hitting it clears the lane (loudly).
    max_entries: int = 200_000

    def validate(self) -> None:
        if self.ttl_s <= 0:
            raise ValueError("shadow ttl_s must be positive")
        if self.min_calibration_samples <= 0:
            raise ValueError("shadow min_calibration_samples must be positive")
        if self.ttl_adjust_factor <= 1.0:
            raise ValueError("shadow ttl_adjust_factor must be > 1.0")


@dataclass
class ShadowLaneStats:
    entries: int = 0
    recorded_requests: int = 0
    calibration_samples: int = 0
    calibration_err_blocks_sum: float = 0.0
    divergences: int = 0
    ttl_s: float = 0.0

    @property
    def mean_abs_err_blocks(self) -> Optional[float]:
        if self.calibration_samples == 0:
            return None
        return self.calibration_err_blocks_sum / self.calibration_samples


class ShadowLane:
    """Egress-derived approximate prefix index for one opaque pool."""

    def __init__(
        self,
        config: ShadowLaneConfig,
        kv_block_size: int,
        hasher,
        clock=time.monotonic,
    ):
        """``hasher(token_ids) -> List[int]`` yields per-block local hashes."""
        config.validate()
        if kv_block_size <= 0:
            raise ValueError("shadow lane kv_block_size must be positive")
        self.config = config
        self.kv_block_size = kv_block_size
        self._hasher = hasher
        self._clock = clock
        self._initial_ttl = config.ttl_s
        self._ttl = config.ttl_s
        self._expiry: Dict[int, float] = {}
        self._stats = ShadowLaneStats(ttl_s=config.ttl_s)

    def _chain_keys(self, token_ids: List[int]) -> List[int]:
        keys = []
        key = 0
        for block_hash in self._hasher(token_ids):
            key = _chain(key, block_hash)
            keys.append(key)
        return keys

    def record_egress(self, token_ids: List[int]) -> None:
        """Index every block prefix of a request this pool accepted."""
        now = self._clock()
        expiry = now + self._ttl
        keys = self._chain_keys(token_ids)
        if len(self._expiry) + len(keys) > self.config.max_entries:
            self._purge(now)
            if len(self._expiry) + len(keys) > self.config.max_entries:
                logger.warning(
                    "Shadow lane cleared: %s live entries exceed max_entries=%s",
                    len(self._expiry),
                    self.config.max_entries,
                )
                self._expiry.clear()
        for key in keys:
            prev = self._expiry.get(key)
            if prev is None or prev < expiry:
                self._expiry[key] = expiry
        self._stats.recorded_requests += 1
        self._stats.entries = len(self._expiry)

    def query(self, token_ids: List[int]) -> int:
        """Predicted cached-prefix depth in KV blocks."""
        now = self._clock()
        depth = 0
        for key in self._chain_keys(token_ids):
            expiry = self._expiry.get(key)
            if expiry is None or expiry < now:
                break
            depth += 1
        return depth

    def observe_usage(self, predicted_blocks: int, cached_tokens: int) -> None:
        """Join a dispatch-time prediction with backend-reported ground truth."""
        actual_blocks = cached_tokens / self.kv_block_size
        err = abs(predicted_blocks - actual_blocks)
        self._stats.calibration_samples += 1
        self._stats.calibration_err_blocks_sum += err
        if err > self.config.max_err_blocks:
            self._stats.divergences += 1
            if actual_blocks < predicted_blocks:
                # Backend evicted sooner than our TTL pretends: shrink.
                self._ttl = max(
                    self.config.ttl_min_s, self._ttl / self.config.ttl_adjust_factor
                )
            else:
                # Backend retains longer (or shares cache with other traffic).
                self._ttl = min(
                    self._initial_ttl * self.config.ttl_max_factor,
                    self._ttl * self.config.ttl_adjust_factor,
                )
            self._stats.ttl_s = self._ttl
            logger.info(
                "Shadow lane TTL adjusted to %.0fs (predicted=%s actual=%.1f blocks)",
                self._ttl,
                predicted_blocks,
                actual_blocks,
            )

    def trusted(self) -> bool:
        """True once enough calibration joins landed within tolerance."""
        stats = self._stats
        if stats.calibration_samples < self.config.min_calibration_samples:
            return False
        mean_err = stats.mean_abs_err_blocks
        assert mean_err is not None
        return mean_err <= self.config.max_err_blocks

    def _purge(self, now: float) -> None:
        self._expiry = {k: v for k, v in self._expiry.items() if v >= now}
        self._stats.entries = len(self._expiry)

    def stats(self) -> ShadowLaneStats:
        self._stats.entries = len(self._expiry)
        self._stats.ttl_s = self._ttl
        return self._stats
