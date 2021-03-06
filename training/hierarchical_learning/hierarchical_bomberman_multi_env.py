from typing import Tuple

from gym.vector.utils import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from training.bomberman_multi_env import BombermanEnv


class HierarchicalBombermanMultiEnv(MultiAgentEnv):
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.flat_env = BombermanEnv(agent_ids)
        self.action_space = spaces.Discrete(3)
        self.actions = ['COLLECT', 'DESTROY', 'KILL']
        self.high_level_mode = True

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

        if self.high_level_mode:
            obs = {}
            rewards = {}
            dones = {}
            infos = {}
            for a in action_dict:
                o, r = self._high_level_step(a, action_dict[a])
                obs.update(o)
                rewards.update(r)
                dones = {"__all__": False}

            return obs, rewards, dones, infos
        else:
            self.high_level_mode = True
            return self._low_level_step(action_dict)

    def _high_level_step(self, agent, action):
        #self.low_level_agent_id = "low_level_{}".format(
        #    self.num_high_level_steps)
        if self.actions[action] == 'COLLECT':
            obs = {f'{self.actions[action]}_{agent}': BombermanEnv.get_observation_from_game_state(self.flat_env.agents[agent].last_game_state, self.agent_ids, self.flat_env.current_step)}
        elif self.actions[action] == 'DESTROY':
            obs = {f'{self.actions[action]}_{agent}': BombermanEnv.get_observation_from_game_state(self.flat_env.agents[agent].last_game_state, self.agent_ids, self.flat_env.current_step)}
        elif self.actions[action] == 'KILL':
            obs = {f'{self.actions[action]}_{agent}': BombermanEnv.get_observation_from_game_state(self.flat_env.agents[agent].last_game_state, self.agent_ids, self.flat_env.current_step)}
        else:
            obs = None
        self.high_level_mode = False
        return obs, {f'{self.actions[action]}_{agent}':0}

    def _low_level_step(self, action_dict):
        low_level_action_dict = {}
        for k,v in action_dict.items():
            key = k.split('_')
            key = f'{key[1]}_{key[2]}'
            low_level_action_dict[key] = v
        return self.flat_env.step(low_level_action_dict)