from src.profile_quality_policy import profile_quality_policy, profile_quality_policy_sha256


def test_canonical_policy_is_not_mutable_through_a_returned_value() -> None:
    original_hash = profile_quality_policy_sha256()
    mutated = profile_quality_policy()
    thresholds = mutated["fret_thresholds"]
    assert isinstance(thresholds, list)
    thresholds[0] = 0.01
    assert profile_quality_policy()["fret_thresholds"] == [0.5] * 5
    assert profile_quality_policy_sha256() == original_hash
