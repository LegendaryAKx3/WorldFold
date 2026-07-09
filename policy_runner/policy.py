# Base policy interface, every policy inherits from this class

from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self) -> None:
        """Called at the start of each episode. Override for stateful policies."""

    @abstractmethod
    def act(self, observation):
        """Return an action for the given observation."""
        raise NotImplementedError
