from stable_baselines3 import PPO

from policy_runner.policy import Policy


class PPOPolicy(Policy):
    def __init__(self, action_space, observation_space, checkpoint):
        super().__init__(action_space, observation_space)
        if not checkpoint:
            raise ValueError("PPOPolicy requires a checkpoint path, e.g. --checkpoint model.zip")
        self.model = PPO.load(checkpoint)

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
