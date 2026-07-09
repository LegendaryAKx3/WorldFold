from policy_runner.policy import Policy


class RandomPolicy(Policy):
    def act(self, observation):
        return self.action_space.sample()
