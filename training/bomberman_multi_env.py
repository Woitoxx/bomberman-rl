import logging
import pickle
import random
from collections import namedtuple
from datetime import datetime
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
from items import Coin, Explosion, Bomb


class Agent:
    def __init__(self, agent_name):
        self.name = agent_name

        self.total_score = 0

        self.dead = None
        self.score = None
        self.trophies = None

        self.x = None
        self.y = None
        self.bombs_left = None

        self.last_game_state = None
        self.last_action = None

    def __str__(self):
        return f'Agent: {self.name} | Score: {self.score} | Alive: {not self.dead} | Bombs left: {self.bombs_left} | Position: {self.x}, {self.y}'

    def start_round(self):
        self.dead = False
        self.score = 0
        self.trophies = []

        self.bombs_left = True

        self.last_game_state = None
        self.last_action = None

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return self.name, self.score, self.bombs_left, (self.x, self.y)

    def update_score(self, delta):
        """Add delta to both the current round's score and the total score."""
        self.score += delta
        self.total_score += delta

    def store_game_state(self, game_state):
        self.last_game_state = game_state

    def get_observation_from_game_state(self, game_state):
        field = game_state['field']
        walls = np.where(field == -1, 1, 0)
        free = np.where(field == 0, 1, 0)
        crates = np.where(field == 1, 1, 0)
        player = np.zeros(field.shape, dtype=int)
        player[self.x, self.y] = 1
        opponents = np.zeros(field.shape, dtype=int)
        for o in game_state['others']:
            opponents[o[3]] = 1

        coins = np.zeros(field.shape, dtype=int)
        ind = tuple(zip(*game_state['coins']))
        if len(ind)>0:
            coins[ind] = 1
        bombs = np.zeros(field.shape)
        game_state_bombs = game_state['bombs']
        #ind = list(zip(*game_state['bombs']))
        for b in game_state_bombs:
            bombs[b[0]] = b[1]
        explosions = game_state['explosion_map']
        return {'walls': walls.flatten(), 'free': free.flatten(), 'crates': crates.flatten(), 'coins': coins.flatten(), 'bombs': bombs, 'player': player.flatten(),
                'opponents': opponents.flatten(), 'explosions': explosions}


class BombermanEnv(MultiAgentEnv):
    running: bool = False
    current_step: int
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    round_id: str
    available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

    def __init__(self, num_agents):

        self.round = 0
        self.running = False
        self.agents = {}
        tiles = 17 * 17
        self.observation_space = spaces.Dict({'walls': spaces.MultiBinary(tiles),
                                              'free': spaces.MultiBinary(tiles),
                                              'crates': spaces.MultiBinary(tiles),
                                              'player': spaces.MultiBinary(tiles),
                                              'opponents': spaces.MultiBinary(tiles),
                                              'coins': spaces.MultiBinary(tiles),
                                              'bombs': spaces.Box(low=0, high=4, shape=(17, 17)),
                                              'explosions': spaces.Box(low=0, high=2, shape=(17, 17))
                                              })
        self.action_space = spaces.Discrete(6)
        for i in range(num_agents):
            agent_id = f'agent_{i}'
            self.agents[agent_id] = Agent(agent_id)
        self.new_round()

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs = {}
        rewards = {}
        dones = {}
        infos = {}

        self.current_step += 1

        random.shuffle(self.active_agents)
        for agent in self.active_agents:
            self.perform_agent_action(agent, self.available_actions[action_dict[agent.name]])

        self.collect_coins()
        self.update_bombs()
        self.evaluate_explosions()

        for agent_name in action_dict.keys():
            agent = self.agents[agent_name]
            agent.store_game_state(self.get_state_for_agent(agent))
            dones[agent_name] = self.agents[agent_name].dead
            rewards[agent.name] = self.calculate_reward(agent)
            obs[agent.name] = agent.get_observation_from_game_state(agent.last_game_state)

        if self.done():
            self.end_round()
            dones['__all__'] = True
            for agent_name in action_dict.keys():
                agent = self.agents[agent_name]
                obs[agent.name] = agent.get_observation_from_game_state(agent.last_game_state)
                rewards[agent.name] = self.calculate_reward(agent)
        else:
            dones['__all__'] = False

        return obs, rewards, dones, infos

    def reset(self) -> MultiAgentDict:
        obs = {}
        self.new_round()
        for agent in self.active_agents:
            agent.store_game_state(self.get_state_for_agent(agent))
            obs[agent.name] = agent.get_observation_from_game_state(agent.last_game_state)
        return obs


    def calculate_reward(self, agent: Agent):
        if not agent.dead:
            '''
            opp_score_max = max([v.score for k, v in self.agents.items() if k != agent.name])
            reward = agent.score - opp_score_max
            '''
            reward = agent.score - agent.last_game_state['self'][1]
            return reward
        return -20.

    def new_round(self):
        if self.running:
            self.end_round()

        self.round += 1

        # Bookkeeping
        self.current_step = 0
        self.active_agents = []
        self.bombs = []
        self.explosions = []
        self.round_id = f'Replay {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'

        # Arena with wall and crate layout
        self.arena = (np.random.rand(s.COLS, s.ROWS) < s.CRATE_DENSITY).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    self.arena[x, y] = -1

        # Starting positions
        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        random.shuffle(start_positions)
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if self.arena[xx, yy] == 1:
                    self.arena[xx, yy] = 0

        # Distribute coins evenly
        self.coins = []
        for i in range(3):
            for j in range(3):
                n_crates = (self.arena[1 + 5 * i:6 + 5 * i, 1 + 5 * j:6 + 5 * j] == 1).sum()
                while True:
                    x, y = np.random.randint(1 + 5 * i, 6 + 5 * i), np.random.randint(1 + 5 * j, 6 + 5 * j)
                    if n_crates == 0 and self.arena[x, y] == 0:
                        self.coins.append(Coin((x, y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x, y] == 1:
                        self.coins.append(Coin((x, y)))
                        break

        # Reset agents and distribute starting positions
        for agent in self.agents.values():
            self.active_agents.append(agent)
            agent.x, agent.y = start_positions.pop()
            agent.start_round()

        self.replay = {
            'round': self.round,
            'arena': np.array(self.arena),
            'coins': [c.get_state() for c in self.coins],
            'agents': [a.get_state() for a in self.agents.values()],
            'actions': dict([(a.name, []) for a in self.agents.values()]),
            'permutations': []
        }

        for agent in self.agents.values():
            agent.store_game_state(self.get_state_for_agent(agent))

        self.running = True

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent: Agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
        elif action == 'BOMB' and agent.bombs_left:
            self.bombs.append(Bomb((agent.x, agent.y), agent, s.BOMB_TIMER, s.BOMB_POWER))
            agent.bombs_left = False
        agent.last_action = action

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        a.update_score(s.REWARD_COIN)

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer <= 0:
                # Explode when timer is finished
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True

                # Create explosion
                screen_coords = [(s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y) for (x, y) in
                                 blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner, s.EXPLOSION_TIMER))
                bomb.active = False
                bomb.owner.bombs_left = True
            else:
                # Progress countdown
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is not explosion.owner:
                            explosion.owner.update_score(s.REWARD_KILL)
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False

            # Progress countdown
            explosion.timer -= 1
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
        self.explosions = [exp for exp in self.explosions if exp.active]

    def end_round(self):
        assert self.running, "End of round requested while not running"

        # Mark round as ended
        self.running = False

    def done(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            return True

        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            return True

        if self.current_step >= s.MAX_STEPS:
            return True

        return False

    def get_state_for_agent(self, agent: Agent):
        state = {
            'round': self.round,
            'current_step': self.current_step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            for (x, y) in exp.blast_coords:
                explosion_map[x, y] = max(explosion_map[x, y], exp.timer)
        state['explosion_map'] = explosion_map

        return state

    @staticmethod
    def get_observation_from_game_state(game_state):
        field = game_state['field']
        walls = np.where(field == -1, 1, 0)
        free = np.where(field == 0, 1, 0)
        crates = np.where(field == 1, 1, 0)
        player = np.zeros(field.shape, dtype=int)
        player[game_state['self'][3]] = 1
        opponents = np.zeros(field.shape, dtype=int)
        for o in game_state['others']:
            opponents[o[3]] = 1

        coins = np.zeros(field.shape, dtype=int)
        ind = tuple(zip(*game_state['coins']))
        if len(ind) > 0:
            coins[ind] = 1
        bombs = np.zeros(field.shape)
        game_state_bombs = game_state['bombs']
        # ind = list(zip(*game_state['bombs']))
        for b in game_state_bombs:
            bombs[b[0]] = b[1]
        explosions = game_state['explosion_map']
        return {'walls': walls.flatten(), 'free': free.flatten(), 'crates': crates.flatten(), 'coins': coins.flatten(),
                'bombs': bombs, 'player': player.flatten(),
                'opponents': opponents.flatten(), 'explosions': explosions}
