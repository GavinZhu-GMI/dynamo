#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for the shadow lane and the approx opaque lane feed."""

import json
from pathlib import Path
from typing import Any

import pytest

from dynamo.global_router.handler import GlobalRouterHandler
from dynamo.global_router.pool_selection import load_config
from dynamo.global_router.relay_affinity import RelayAffinity, RelayAffinityConfig
from dynamo.global_router.shadow_lane import ShadowLane, ShadowLaneConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]

BLOCK = 4


def _hasher(token_ids: list[int]) -> list[int]:
    return [
        hash(tuple(token_ids[i : i + BLOCK]))
        for i in range(0, len(token_ids) - len(token_ids) % BLOCK, BLOCK)
    ]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _lane(clock=None, **overrides) -> ShadowLane:
    return ShadowLane(
        ShadowLaneConfig(**overrides),
        BLOCK,
        _hasher,
        clock=clock or FakeClock(),
    )


def test_shadow_record_query_roundtrip():
    lane = _lane()
    tokens = list(range(BLOCK * 5))
    assert lane.query(tokens) == 0
    lane.record_egress(tokens)
    assert lane.query(tokens) == 5
    # A shared 3-block prefix with a different tail matches partially.
    assert lane.query(tokens[: BLOCK * 3] + [999] * BLOCK) == 3
    # Disjoint tokens match nothing.
    assert lane.query([777] * (BLOCK * 4)) == 0


def test_shadow_entries_expire():
    clock = FakeClock()
    lane = _lane(clock=clock, ttl_s=100.0)
    tokens = list(range(BLOCK * 4))
    lane.record_egress(tokens)
    assert lane.query(tokens) == 4
    clock.now += 101.0
    assert lane.query(tokens) == 0


def test_shadow_trust_requires_samples_within_tolerance():
    lane = _lane(min_calibration_samples=3, max_err_blocks=1.0)
    assert not lane.trusted()
    for _ in range(3):
        lane.observe_usage(predicted_blocks=4, cached_tokens=4 * BLOCK)
    assert lane.trusted()


def test_shadow_trust_denied_on_large_errors():
    lane = _lane(min_calibration_samples=2, max_err_blocks=1.0)
    for _ in range(4):
        lane.observe_usage(predicted_blocks=10, cached_tokens=0)
    assert not lane.trusted()


def test_shadow_ttl_shrinks_on_overprediction_and_grows_back():
    lane = _lane(ttl_s=100.0, max_err_blocks=1.0, ttl_adjust_factor=2.0)
    # Backend had far less cached than predicted -> evicts sooner -> shrink.
    lane.observe_usage(predicted_blocks=10, cached_tokens=0)
    assert lane.stats().ttl_s == 50.0
    # Backend had far more cached than predicted -> retains longer -> grow.
    lane.observe_usage(predicted_blocks=0, cached_tokens=40 * BLOCK)
    assert lane.stats().ttl_s == 100.0


class FakeApproxClient:
    def __init__(self, name: str, depth, healthy: bool = True):
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.name = name
        self._depth = depth
        self._healthy = healthy
        self.calls = 0

    def healthy(self) -> bool:
        return self._healthy

    def approx_depth(self, token_ids):
        return self._depth if self._healthy else None

    async def generate(self, request):
        self.calls += 1
        yield {"pool": f"opaque:{self.config.name}"}


class FakeRelayClient:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    async def generate(self, request: dict[str, Any]):
        self.calls += 1

        async def stream():
            yield {"pool": self.name}

        return stream()


class FakeConsumer:
    def __init__(self, matches):
        self.matches = matches

    def find_matches(self, token_ids, kv_block_size):
        return self.matches


def _agg_config(opaque_extra: dict | None = None) -> dict[str, Any]:
    config = {
        "mode": "agg",
        "num_agg_pools": 2,
        "agg_pool_dynamo_namespaces": ["ns-a", "ns-b"],
        "agg_pool_selection_strategy": {
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 1,
            "itl_min": 5,
            "itl_max": 200,
            "itl_resolution": 1,
            "agg_pool_mapping": [[0]],
        },
        "relay_affinity": {"enabled": True, "kv_block_size": 16},
    }
    if opaque_extra is not None:
        config["opaque_pools"] = [
            {
                "name": "op",
                "url": "http://op/v1",
                "model": "m",
                **opaque_extra,
            }
        ]
    return config


def _write_config(tmp_dir: Path, config_data: dict[str, Any]) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def test_config_approx_feed_requires_relay_affinity(tmp_path):
    data = _agg_config(opaque_extra={"feed": "approx"})
    data["relay_affinity"] = {"enabled": False}
    with pytest.raises(ValueError, match="relay_affinity"):
        load_config(str(_write_config(tmp_path, data)))


def test_config_rejects_unknown_feed_and_api(tmp_path):
    with pytest.raises(ValueError, match="feed"):
        load_config(
            str(_write_config(tmp_path, _agg_config(opaque_extra={"feed": "bogus"})))
        )
    with pytest.raises(ValueError, match="api"):
        load_config(
            str(_write_config(tmp_path, _agg_config(opaque_extra={"api": "bogus"})))
        )


def _handler_with_affinity(tmp_path, relay_matches) -> GlobalRouterHandler:
    handler = GlobalRouterHandler(
        runtime=object(),
        config_path=str(
            _write_config(tmp_path, _agg_config(opaque_extra={"feed": "approx"}))
        ),
        model_name="test-model",
    )
    ns_a = FakeRelayClient("ns-a")
    ns_b = FakeRelayClient("ns-b")
    handler.agg_clients = {"ns-a": ns_a, "ns-b": ns_b}
    affinity = RelayAffinity(
        RelayAffinityConfig(enabled=True, kv_block_size=16), ["ns-a", "ns-b"]
    )
    affinity._consumer = FakeConsumer(relay_matches)
    handler.relay_affinity = affinity
    return handler


def _match(label: str, ready: bool, depth: int) -> dict[str, Any]:
    return {"label": label, "ready": ready, "prefix_depth": depth}


@pytest.mark.asyncio
async def test_trusted_approx_lane_wins_over_relay(tmp_path):
    handler = _handler_with_affinity(
        tmp_path, [_match("ns-a", True, 2), _match("ns-b", True, 1)]
    )
    opaque = FakeApproxClient("op", depth=9)
    handler.opaque_clients = [opaque]

    outputs = [
        output async for output in handler.handle_generate({"token_ids": [1] * 160})
    ]

    assert outputs == [{"pool": "opaque:op"}]
    assert opaque.calls == 1
    assert handler.agg_clients["ns-a"].calls == 0


@pytest.mark.asyncio
async def test_untrusted_approx_lane_is_ignored(tmp_path):
    handler = _handler_with_affinity(
        tmp_path, [_match("ns-a", True, 0), _match("ns-b", True, 0)]
    )
    # approx_depth None models an untrusted (or missing) shadow lane.
    opaque = FakeApproxClient("op", depth=None)
    handler.opaque_clients = [opaque]

    outputs = [
        output async for output in handler.handle_generate({"token_ids": [1] * 160})
    ]

    assert outputs == [{"pool": "ns-a"}]
    assert opaque.calls == 0


@pytest.mark.asyncio
async def test_approx_lane_without_decisive_lead_falls_back(tmp_path):
    handler = _handler_with_affinity(
        tmp_path, [_match("ns-a", True, 9), _match("ns-b", True, 0)]
    )
    opaque = FakeApproxClient("op", depth=9)  # ties relay best -> no lead
    handler.opaque_clients = [opaque]

    outputs = [
        output async for output in handler.handle_generate({"token_ids": [1] * 160})
    ]

    assert outputs == [{"pool": "ns-a"}]
    assert opaque.calls == 0
