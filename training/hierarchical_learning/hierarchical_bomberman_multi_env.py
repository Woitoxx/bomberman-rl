from typing import Tuple

from gym.vector.utils import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from training.hierarchical_learning.bomberman_multi_env import BombermanEnv

COIN_OBSERVATION_SPACE = spaces.Box(low=0, high=1, shape=(15, 15, 10))
CRATE_OBSERVATION_SPACE = spaces.Box(low=0, high=1, shape=(15, 15, 13))
KILL_OBSERVATION_SPACE = spaces.Box(low=0, high=1, shape=(15, 15, 13))

class HierarchicalBombermanMultiEnv(MultiAgentEnv):
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.flat_env = BombermanEnv(agent_ids)
        self.action_space = spaces.Discrete(3)
        self.actions = ['COLLECT', 'DESTROY', 'KILL']
        self.high_level_mode = True
        self.action_buffer = {}

    def reset(self):
        obs = self.flat_env.reset()
        self.high_level_mode = True
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        #self.low_level_agent_id = "low_level_{}".format(
        #    self.num_high_level_steps)
        return obs

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent_name, action in action_dict.items():
            if agent_name.endswith('_high'):
                agent = self.flat_env.agents[agent_name]
                agent_id = f'{agent.low_level_prefix}{agent.high_level_steps}'
                if self.actions[action] == 'COLLECT':
                    obs = {agent_id : BombermanEnv.get_observation_from_game_state(agent.last_game_state, self.agent_ids)}
                elif self.actions[action] == 'DESTROY':
                    pass
                elif self.actions[action] == 'KILL':
                    pass
                else:
                    obs = None
                pass
            else:
                #agent_1_high
                #agent_1_low_1
                #agent_2_low_5
                high_level_agent_name = f'{agent_name[:7]}_high'
                self.action_buffer[high_level_agent_name] = action
                #agent = self.flat_env.agents[high_level_agent_name]
                pass


        if len(self.action_buffer) == len(self.agent_ids):
            self.action_buffer = {}
            self.flat_env.step()
            pass

        dones.update({"__all__": False})

        return obs, rewards, dones, infos