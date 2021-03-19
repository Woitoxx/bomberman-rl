from typing import Tuple

from gym.vector.utils import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from training.hierarchical_learning.bomberman_multi_env import *

import numpy as np

COLLECT_OBSERVATION_SPACE = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 11)), spaces.MultiBinary(6)))
DESTROY_OBSERVATION_SPACE = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 11)), spaces.MultiBinary(6)))
KILL_OBSERVATION_SPACE = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 13)), spaces.MultiBinary(6)))
HIGH_LEVEL_ACTIONS = ['COLLECT', 'DESTROY', 'KILL']

class HierarchicalBombermanMultiEnv(MultiAgentEnv):
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.flat_env = BombermanEnv(agent_ids)

        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 14)),
                                               spaces.Box(low=0, high=1, shape=(4,)),
                                               #spaces.MultiBinary(3),
                                               #spaces.Box(low=0, high=1, shape=(1,)),
                                               spaces.MultiBinary(len(HIGH_LEVEL_ACTIONS))))
        self.action_space = spaces.Discrete(3)
        self.high_level_mode = True
        self.action_buffer = {}
        self.high_low_mapping = {}

    def reset(self):
        obs = {}
        self.flat_env.new_round()
        self.flat_env.agents_last_obs = {}
        self.action_buffer = {}
        for agent in self.flat_env.active_agents:
            agent.store_game_state(self.flat_env.get_state_for_agent(agent))
            obs[agent.name] = get_high_level_observation_from_game_state(agent.last_game_state, self.flat_env.agents.keys())
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
                action_name = HIGH_LEVEL_ACTIONS[action]
                agent.high_level_steps += 1
                agent_id = f'{agent.low_level_prefix}{action_name}_{agent_name}_{agent.high_level_steps}'
                if action_name == 'COLLECT':
                    obs.update({agent_id : get_collect_observation_from_game_state(agent.last_game_state)})
                elif action_name == 'DESTROY':
                    obs.update({agent_id : get_destroy_observation_from_game_state(agent.last_game_state)})
                elif action_name == 'KILL':
                    obs.update({agent_id : get_kill_observation_from_game_state(agent.last_game_state)})
                else:
                    raise Exception()
                rewards.update({agent_id: 0})
                dones.update({agent_id: False })
                self.high_low_mapping[agent_name] = agent_id
                agent.current_mode = action_name
                agent.current_sub_id = agent_id
                #print(f'Agent {agent_name} now {action_name}')
            else:
                #agent_1_high
                #agent_1_low_1
                #agent_2_low_5
                agent_parts = agent_name.split('_')
                high_level_agent_name = f'{agent_parts[2]}_{agent_parts[3]}_high'
                self.action_buffer[high_level_agent_name] = action
                #print(f'Add to buffer: Agent {high_level_agent_name} - Action {action}')

                #agent = self.flat_env.agents[high_level_agent_name]

        if len(self.action_buffer) == len(self.flat_env.active_agents):
            obs, rewards, dones, infos = self.flat_env.step(self.action_buffer)

            self.action_buffer = {}
            pass
        else:
            dones.update({'__all__' : False})

        return obs, rewards, dones, infos
