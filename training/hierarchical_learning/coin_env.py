import logging
import pickle
import random
from collections import namedtuple
from datetime import datetime
from itertools import chain
from logging.handlers import RotatingFileHandler
from os.path import dirname
from threading import Event
from time import time
from typing import List, Union, Tuple
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

import events as e
import settings as s
from fallbacks import pygame
from training.bomberman_multi_env import Agent
from training.items import Coin, Explosion, Bomb
import uuid
import gym


class BombermanCoinEnv(gym.Env):
    running: bool = False
    current_step: int
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    round_id: str
    available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def __init__(self):

        self.round = 0
        self.running = False
        self.agents = {}
        self.phase = 0
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 4)),))
        self.action_space = spaces.Discrete(4)
        self.agent = Agent('agent_0')
        self.new_round()

    def render(self, mode='human'):
        pass

    def step(self, action : int):
        self.current_step += 1
        self.perform_agent_action(self.available_actions[action])
        self.collect_coins()
        reward = self.calculate_reward()
        self.agent.store_game_state(self.get_state_for_agent(self.agent))
        obs = self.get_observation_from_game_state(self.agent.last_game_state)

        if self.done():
            self.end_round()
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def reset(self):
        self.new_round()
        self.agent.store_game_state(self.get_state_for_agent(self.agent))
        obs = self.get_observation_from_game_state(self.agent.last_game_state)
        return obs

    def calculate_reward(self):
        reward = (self.agent.score - self.agent.last_game_state['self'][1])
        return -0.001 if reward == 0 else reward

    def new_round(self):
        if self.running:
            self.end_round()

        self.round += 1

        # Bookkeeping
        self.current_step = 0

        # Arena with wall and crate layout
        self.arena = np.zeros((s.COLS, s.ROWS))
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    self.arena[x, y] = -1

        # Starting positions
        start_position = None
        while start_position is None or self.arena[start_position] == -1:
            start_position = np.random.randint(1, s.COLS), np.random.randint(1, s.ROWS)

        self.agent.x, self.agent.y = start_position

        # Distribute coins evenly
        self.coins = []
        for i in range(3):
            for j in range(3):
                n_crates = 0
                while True:
                    x, y = np.random.randint(1 + 5 * i, 6 + 5 * i), np.random.randint(1 + 5 * j, 6 + 5 * j)
                    if (x,y) == start_position:
                        continue

                    if n_crates == 0 and self.arena[x, y] == 0:
                        self.coins.append(Coin((x, y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x, y] == 1:
                        self.coins.append(Coin((x, y)))
                        break

        # Reset agents and distribute starting positions

        self.agent.start_round()
        self.agent.store_game_state(self.get_state_for_agent(self.agent))

        self.running = True

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        return is_free

    def perform_agent_action(self, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(self.agent.x, self.agent.y - 1):
            self.agent.y -= 1
        elif action == 'DOWN' and self.tile_is_free(self.agent.x, self.agent.y + 1):
            self.agent.y += 1
        elif action == 'LEFT' and self.tile_is_free(self.agent.x - 1, self.agent.y):
            self.agent.x -= 1
        elif action == 'RIGHT' and self.tile_is_free(self.agent.x + 1, self.agent.y):
            self.agent.x += 1
        self.agent.last_action = action

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                if self.agent.x == coin.x and self.agent.y == coin.y:
                    coin.collectable = False
                    self.agent.update_score(s.REWARD_COIN)

    def end_round(self):
        assert self.running, "End of round requested while not running"

        # Mark round as ended
        self.running = False

    def done(self):
        if all([not c.collectable for c in self.coins]):
            return True
        return False

    def get_state_for_agent(self, agent: Agent):
        state = {
            'round': self.round,
            'current_step': self.current_step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
        }

        return state

    #TODO: fix. make static, remove dependency on self.agents
    @staticmethod
    def get_observation_from_game_state(game_state):
        field = game_state['field']
        walls = np.where(field == -1, 1, 0)[1:-1, 1:-1]
        free = np.where(field == 0, 1, 0)[1:-1, 1:-1]

        player = np.zeros(field.shape, dtype=int)
        player[game_state['self'][3]] = 1
        player = player[1:-1, 1:-1]

        coins = np.zeros(field.shape, dtype=int)
        for c in game_state['coins']:
            coins[c] = 1
        coins = coins[1:-1, 1:-1]

        out = np.stack((walls, free, coins, player), axis=2)
        return (out,)
