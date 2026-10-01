from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorComparison:
    baseline: str
    variant: str
    changed: bool
    observations: list[str]


def compare_responses(baseline: str, variant: str) -> BehaviorComparison:
    baseline_normalized = " ".join(baseline.split()).lower()
    variant_normalized = " ".join(variant.split()).lower()

    observations = []

    if baseline_normalized == variant_normalized:
        observations.append("The normalized responses are identical.")
        changed = False
    else:
        observations.append("The normalized responses differ.")
        changed = True

    if len(baseline) != len(variant):
        observations.append(
            f"Response length changed from {len(baseline)} to {len(variant)} characters."
        )

    return BehaviorComparison(
        baseline=baseline,
        variant=variant,
        changed=changed,
        observations=observations,
    )
