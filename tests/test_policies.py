"""Contract tests for adaptive stopping policies."""

import pytest

from research.policies.beta_stopping import BetaStoppingPolicy
from research.policies.fixed_k import FixedKPolicy
from research.policies.learned_stopper import LearnedStopperPolicy
from research.policies.margin_entropy import MarginEntropyPolicy
from research.policies.three_action_router import ThreeActionRouterPolicy
from research.simulator import simulate_one


def _record() -> dict:
    return {
        "candidates": [
            {"sql": f"SELECT {i}", "fingerprint": "a" if i % 2 else "b"}
            for i in range(64)
        ]
    }


@pytest.mark.parametrize(
    "policy,limit",
    [
        (FixedKPolicy(k=16), 16),
        (BetaStoppingPolicy(min_k=4, max_k=16), 16),
        (MarginEntropyPolicy(min_k=4, max_k=16), 16),
        (LearnedStopperPolicy(min_k=4, max_k=16), 16),
        (ThreeActionRouterPolicy(min_k=4, max_k=16), 16),
    ],
)
def test_policy_does_not_exceed_max_k(policy, limit):
    result = simulate_one(_record(), policy, "a", list(range(64)), batch_size=4)

    assert result["n_samples_used"] <= limit
    assert result["action"] in {"STOP", "SAMPLE_MORE", "CALL_CSC"}


def test_router_emits_only_three_protocol_actions():
    policy = ThreeActionRouterPolicy(min_k=4, max_k=16)
    candidates = _record()["candidates"]

    actions = {
        policy.should_stop(candidates[:step], step)[1]
        for step in range(0, 17)
    }

    assert actions <= {"STOP", "SAMPLE_MORE", "CALL_CSC"}
