"""Registry mapping --policy names to Policy classes.

To add a new policy:
1. Create a class in policy_runner/policies/ that subclasses Policy
   (implement act(), and reset() if it's stateful).
2. Register it below with a short name.
"""

from policy_runner.policies.ppo_policy import PPOPolicy
from policy_runner.policies.random_policy import RandomPolicy

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "ppo": PPOPolicy,
}


def load_policy(name, action_space, observation_space, **policy_kwargs):
    try:
        policy_cls = POLICY_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Unknown policy '{name}'. Available policies: {available}")
    return policy_cls(action_space, observation_space, **policy_kwargs)
