'''

Verify installaion

'''

# import so101_nexus.mujoco
# import gymnasium as gym

# env = gym.make("MuJoCoPickLift-v1")
# print("Installation OK")
# env.close()


'''



'''

import time

import gymnasium as gym
import so101_nexus.mujoco
import mujoco.viewer
from so101_nexus import PickConfig, YCBObject

config = PickConfig(objects=YCBObject(model_id="011_banana"))
env = gym.make("MuJoCoPickLift-v1", config=config, render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1/50)
    print("the sim is running btw")
    if terminated or truncated:
        obs, info = env.reset()

env.close()