'''Arkhai seller-side cloud market environment'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.arkhai import binding

# Note: I have moved out config into the .ini because the list of args is quite long. 
# Let me know if you'd rather have them duplicated explicitly.
class Arkhai(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0, **kwargs):
        node_types = int(kwargs.get('node_types', 3))
        obs_dim = 12 + 3 * node_types
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(obs_dim,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 2])
        self.render_mode = render_mode
        ai_sellers = int(kwargs.get('ai_sellers', 1))
        ai_buyers  = int(kwargs.get('ai_buyers', 0))
        num_ai_agents = max(1, ai_sellers + ai_buyers)
        self.num_agents = num_envs * num_ai_agents
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, **kwargs)
 
    def reset(self, seed=0):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 4096
    env = Arkhai(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randn(CACHE, N, env.single_action_space.shape[0])

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('Squared SPS:', int(env.num_agents*steps / (time.time() - start)))
