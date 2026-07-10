#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for opaque-lane (forfeit-only) pool semantics in the global router."""

import json
from pathlib import Path

import pytest

from dynamo.global_router.handler import GlobalRouterHandler
from dynamo.global_router.opaque_pool import PoolLatencyTracker, isl_bucket
from dynamo.global_router.pool_selection import load_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


# --- Helpers ---


def _agg_config_data(**overrides) -> dict:
    config = {
        "mode": "agg",
        "num_agg_pools": 2,
        "agg_pool_dynamo_namespaces": ["ns-agg-0", "ns-agg-1"],
        "agg_pool_selection_strategy": {
            "ttft_min_ms": 10,
            "ttft_max_ms": 2000,
            "ttft_resolution": 2,
            "itl_min_ms": 5,
            "itl_max_ms": 200,
            "itl_resolution": 2,
            "agg_pool_mapping": [[0, 0], [1, 1]],
        },
    }
    config.update(overrides)
    return config


def _opaque_pool_data(**overrides) -> dict:
    pool = {
        "name": "external-0",
        "url": "http://gateway.example/v1",
        "model": "Qwen/Qwen3-0.6B",
    }
    pool.update(overrides)
    return pool


def _write_config(tmp_path: Path, config_data: dict) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def _make_handler(tmp_path: Path, config_data: dict) -> GlobalRouterHandler:
    config_path = _write_config(tmp_path, config_data)
    handler = GlobalRouterHandler(
        runtime=object(),  # only stored, never touched by these tests
        config_path=str(config_path),
        model_name="Qwen/Qwen3-0.6B",
    )
    for namespace in handler.config.agg_pool_dynamo_namespaces:
        handler.agg_latency_trackers[namespace] = PoolLatencyTracker(
            alpha=handler.config.forfeit.ttft_ema_alpha
        )
    return handler


def _fill_tracker(tracker: PoolLatencyTracker, ttft_ms: float, samples: int) -> None:
    for _ in range(samples):
        tracker.record(ttft_ms)


# --- Config parsing ---


class TestOpaqueConfigParsing:
    def test_defaults_when_absent(self, tmp_path):
        config = load_config(_write_config(tmp_path, _agg_config_data()))
        config.validate()
        assert config.opaque_pools == []
        assert config.forfeit.ttft_ema_alpha == 0.2
        assert config.forfeit.ttft_viability_factor == 1.5
        assert config.forfeit.min_samples == 3

    def test_opaque_pool_parsed(self, tmp_path):
        data = _agg_config_data(
            opaque_pools=[_opaque_pool_data(first_chunk_timeout_s=5.0)],
            forfeit={"ttft_viability_factor": 2.0},
        )
        config = load_config(_write_config(tmp_path, data))
        config.validate()
        assert len(config.opaque_pools) == 1
        assert config.opaque_pools[0].name == "external-0"
        assert config.opaque_pools[0].first_chunk_timeout_s == 5.0
        assert config.forfeit.ttft_viability_factor == 2.0
        # unspecified forfeit fields keep defaults
        assert config.forfeit.min_samples == 3

    def test_duplicate_opaque_names_rejected(self, tmp_path):
        data = _agg_config_data(
            opaque_pools=[_opaque_pool_data(), _opaque_pool_data()]
        )
        with pytest.raises(ValueError, match="unique"):
            load_config(_write_config(tmp_path, data))

    def test_empty_url_rejected(self, tmp_path):
        data = _agg_config_data(opaque_pools=[_opaque_pool_data(url="")])
        with pytest.raises(ValueError, match="non-empty url"):
            load_config(_write_config(tmp_path, data))


# --- Viability / forfeit semantics ---


class TestViability:
    def test_cold_start_pools_are_viable(self, tmp_path):
        handler = _make_handler(
            tmp_path, _agg_config_data(opaque_pools=[_opaque_pool_data()])
        )
        # No samples recorded: both pools viable regardless of target
        assert handler._viable_agg_pools(ttft_target_ms=1.0) == [0, 1]

    def test_slow_pool_becomes_non_viable(self, tmp_path):
        handler = _make_handler(
            tmp_path, _agg_config_data(opaque_pools=[_opaque_pool_data()])
        )
        # Pool 0: EMA 1000ms over min_samples; pool 1 cold
        _fill_tracker(handler.agg_latency_trackers["ns-agg-0"], 1000.0, samples=3)
        # target 100ms * factor 1.5 = 150ms < 1000ms -> pool 0 out
        assert handler._viable_agg_pools(ttft_target_ms=100.0) == [1]

    def test_all_pools_non_viable(self, tmp_path):
        handler = _make_handler(
            tmp_path, _agg_config_data(opaque_pools=[_opaque_pool_data()])
        )
        _fill_tracker(handler.agg_latency_trackers["ns-agg-0"], 1000.0, samples=3)
        _fill_tracker(handler.agg_latency_trackers["ns-agg-1"], 900.0, samples=3)
        assert handler._viable_agg_pools(ttft_target_ms=100.0) == []

    def test_fast_pool_stays_viable(self, tmp_path):
        handler = _make_handler(
            tmp_path, _agg_config_data(opaque_pools=[_opaque_pool_data()])
        )
        _fill_tracker(handler.agg_latency_trackers["ns-agg-0"], 100.0, samples=5)
        _fill_tracker(handler.agg_latency_trackers["ns-agg-1"], 1000.0, samples=5)
        assert handler._viable_agg_pools(ttft_target_ms=100.0) == [0]

    def test_default_target_uses_grid_midpoint(self, tmp_path):
        handler = _make_handler(
            tmp_path, _agg_config_data(opaque_pools=[_opaque_pool_data()])
        )
        # midpoint of [10, 2000] = 1005ms; factor 1.5 -> threshold 1507.5ms
        _fill_tracker(handler.agg_latency_trackers["ns-agg-0"], 1400.0, samples=3)
        _fill_tracker(handler.agg_latency_trackers["ns-agg-1"], 1600.0, samples=3)
        assert handler._viable_agg_pools(ttft_target_ms=None) == [0]


# --- Building blocks ---


class TestBuildingBlocks:
    def test_latency_tracker_ema(self):
        tracker = PoolLatencyTracker(alpha=0.5)
        tracker.record(100.0)
        assert tracker.ema_ttft_ms == 100.0
        tracker.record(200.0)
        assert tracker.ema_ttft_ms == 150.0
        assert tracker.samples == 2

    def test_isl_buckets(self):
        assert isl_bucket(0) == 0
        assert isl_bucket(1023) == 0
        assert isl_bucket(1024) == 1
        assert isl_bucket(8192) == 2
        assert isl_bucket(32768) == 3
        assert isl_bucket(300_000) == 3
