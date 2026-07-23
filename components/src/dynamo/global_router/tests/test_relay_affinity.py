#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for relay-fed prefix affinity in agg pool selection."""

import json
from pathlib import Path
from typing import Any

import pytest

from dynamo.global_router.handler import GlobalRouterHandler
from dynamo.global_router.pool_selection import load_config
from dynamo.global_router.relay_affinity import RelayAffinity, RelayAffinityConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


class FakeConsumer:
    """Duck-typed stand-in for dynamo.llm.KvDcRelayConsumer."""

    def __init__(self, matches: list[dict[str, Any]]):
        self.matches = matches
        self.calls: list[tuple[list[int], int]] = []

    def find_matches(self, token_ids: list[int], kv_block_size: int):
        self.calls.append((token_ids, kv_block_size))
        return self.matches


class FakeClient:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    async def generate(self, request: dict[str, Any]):
        self.calls += 1

        async def stream():
            yield {"pool": self.name}

        return stream()


def _affinity(
    matches: list[dict[str, Any]],
    namespaces: list[str] | None = None,
    min_match_blocks: int = 4,
    min_lead_blocks: int = 1,
) -> RelayAffinity:
    affinity = RelayAffinity(
        RelayAffinityConfig(
            enabled=True,
            kv_block_size=16,
            min_match_blocks=min_match_blocks,
            min_lead_blocks=min_lead_blocks,
        ),
        namespaces or ["ns-a", "ns-b"],
    )
    affinity._consumer = FakeConsumer(matches)
    return affinity


def _match(label: str, ready: bool, depth: int) -> dict[str, Any]:
    return {"label": label, "ready": ready, "prefix_depth": depth}


def test_config_requires_block_size_when_enabled():
    with pytest.raises(ValueError, match="kv_block_size"):
        RelayAffinityConfig(enabled=True).validate()
    RelayAffinityConfig(enabled=False).validate()
    RelayAffinityConfig(enabled=True, kv_block_size=16).validate()


def test_select_prefers_decisively_deeper_pool():
    affinity = _affinity([_match("ns-a", True, 2), _match("ns-b", True, 8)])
    idx, matches = affinity.select([1] * 128)
    assert idx == 1
    assert len(matches) == 2


def test_select_falls_through_below_min_match_blocks():
    affinity = _affinity([_match("ns-a", True, 3), _match("ns-b", True, 0)])
    idx, _ = affinity.select([1] * 128)
    assert idx is None


def test_select_falls_through_on_tie():
    affinity = _affinity([_match("ns-a", True, 8), _match("ns-b", True, 8)])
    idx, _ = affinity.select([1] * 128)
    assert idx is None


def test_select_ignores_not_ready_lanes():
    # ns-b's lane is between generations; its depth must not count.
    affinity = _affinity([_match("ns-a", True, 8), _match("ns-b", False, 0)])
    idx, _ = affinity.select([1] * 128)
    assert idx == 0


def test_select_none_when_no_lane_ready():
    affinity = _affinity([_match("ns-a", False, 0), _match("ns-b", False, 0)])
    idx, matches = affinity.select([1] * 128)
    assert idx is None
    assert len(matches) == 2


def test_select_none_before_start():
    affinity = RelayAffinity(
        RelayAffinityConfig(enabled=True, kv_block_size=16), ["ns-a"]
    )
    idx, matches = affinity.select([1] * 128)
    assert idx is None
    assert matches == []


def _agg_config() -> dict[str, Any]:
    return {
        "mode": "agg",
        "num_agg_pools": 2,
        "agg_pool_dynamo_namespaces": ["ns-a", "ns-b"],
        "agg_pool_priorities": [0, 5],
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


def _write_config(tmp_dir: Path, config_data: dict[str, Any]) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def test_load_config_parses_relay_affinity(tmp_path):
    config = load_config(str(_write_config(tmp_path, _agg_config())))
    assert config.relay_affinity.enabled
    assert config.relay_affinity.kv_block_size == 16
    assert config.relay_affinity.min_match_blocks == 4


def test_load_config_rejects_relay_affinity_without_block_size(tmp_path):
    data = _agg_config()
    data["relay_affinity"] = {"enabled": True}
    with pytest.raises(ValueError, match="kv_block_size"):
        load_config(str(_write_config(tmp_path, data)))


@pytest.mark.asyncio
async def test_handler_affinity_overrides_grid_pool(tmp_path):
    handler = GlobalRouterHandler(
        runtime=object(),
        config_path=str(_write_config(tmp_path, _agg_config())),
        model_name="test-model",
    )
    ns_a = FakeClient("ns-a")
    ns_b = FakeClient("ns-b")
    handler.agg_clients = {"ns-a": ns_a, "ns-b": ns_b}
    # Grid always selects pool 0; affinity says pool 1 holds a deep prefix.
    handler.relay_affinity = _affinity(
        [_match("ns-a", True, 0), _match("ns-b", True, 8)]
    )

    outputs = [
        output async for output in handler.handle_generate({"token_ids": [1] * 128})
    ]

    assert outputs == [{"pool": "ns-b"}]
    assert ns_b.calls == 1
    assert ns_a.calls == 0


@pytest.mark.asyncio
async def test_handler_falls_back_to_grid_without_decisive_match(tmp_path):
    handler = GlobalRouterHandler(
        runtime=object(),
        config_path=str(_write_config(tmp_path, _agg_config())),
        model_name="test-model",
    )
    ns_a = FakeClient("ns-a")
    ns_b = FakeClient("ns-b")
    handler.agg_clients = {"ns-a": ns_a, "ns-b": ns_b}
    handler.relay_affinity = _affinity(
        [_match("ns-a", True, 1), _match("ns-b", True, 1)]
    )

    outputs = [
        output async for output in handler.handle_generate({"token_ids": [1] * 128})
    ]

    assert outputs == [{"pool": "ns-a"}]
    assert ns_a.calls == 1
    assert ns_b.calls == 0
