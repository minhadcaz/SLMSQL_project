from core.result_grouper import group_candidates
from core.result_normalizer import fingerprint
from core.types import Candidate, ExecutionResult


def test_normalization():
    assert fingerprint([[1], [2.0]], 1, False) == fingerprint([[2], [1.0]], 1, False)
    assert fingerprint([[1], [2]], 1, True) != fingerprint([[2], [1]], 1, True)
    assert fingerprint([[1], [1]], 1, False) != fingerprint([[1]], 1, False)
    assert fingerprint([[None]], 1, False) != fingerprint([["None"]], 1, False)
    assert fingerprint([[b"abc"]], 1, False) != fingerprint([["abc"]], 1, False)
    assert fingerprint([], 1, False) != fingerprint([], 2, False)


def test_grouping():
    candidates = [Candidate(i, execution=ExecutionResult(success=True, normalized_result=key))
                  for i, key in enumerate(["a", "a", "b"])]
    candidates.append(Candidate(3, execution=ExecutionResult(success=False)))
    groups = group_candidates(candidates)
    assert len(groups) == 2
    assert candidates[0].vote_score == 2 / 3
    assert candidates[3].group is None
