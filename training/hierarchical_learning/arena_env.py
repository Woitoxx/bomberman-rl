import random
from collections import namedtuple
from datetime import datetime
from typing import List, Union, Tuple
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from training.bomberman_multi_env import Agent
from training.items import Coin, Explosion, Bomb
import settings as s

arena_settings = namedtuple('ArenaSettings',['COLS','ROWS', 'CRATE_DENSITY', 'BOMB_TIMER', 'BOMB_POWER','EXPLOSION_TIMER'])

class BombermanArenaEnv(MultiAgentEnv):
    running: bool = False
    current_step: int
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]
    COLS = 7
    ROWS = 7
    round_id: str
    available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

    def __init__(self, agent_ids):

        self.round = 0
        self.running = False
        self.agents = {}
        self.phase = 0
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 8)),
                                               spaces.Box(low=0, high=1, shape=(1,))))
        self.action_space = spaces.Discrete(6)
        for i in agent_ids:
            #agent_id = f'agent_{i}'  # _{uuid.uuid1()}'
            self.agents[i] = Agent(i)
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

        self.update_bombs()
        self.evaluate_explosions()

        for agent_name in action_dict.keys():
            agent = self.agents[agent_name]
            rewards[agent.name] = self.calculate_reward(agent)
            agent.store_game_state(self.get_state_for_agent(agent))
            dones[agent_name] = self.agents[agent_name].dead
            obs[agent.name] = self.get_observation_from_game_state(agent.last_game_state, self.agents.keys(), self.current_step)
            infos[agent.name] = agent.score

        if self.done():
            self.end_round()
            dones['__all__'] = True
            if self.phase == 2:
                w, l = self.get_winner_loser()
                for agent_name in action_dict.keys():
                    if agent_name in w:
                        rewards[agent_name] += 1.
                    elif agent_name in l:
                        rewards[agent_name] -= 1.
        else:
            dones['__all__'] = False

        return obs, rewards, dones, infos

    def reset(self) -> MultiAgentDict:
        obs = {}
        self.new_round()
        for agent in self.active_agents:
            agent.store_game_state(self.get_state_for_agent(agent))
            obs[agent.name] = self.get_observation_from_game_state(agent.last_game_state, self.agents.keys(), self.current_step)
        return obs

    def set_phase(self, phase):
        self.phase = phase

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
        self.arena = (np.random.rand(self.COLS, self.ROWS) < s.CRATE_DENSITY).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(self.s.COLS):
            for y in range(self.s.ROWS):
                if (x + 1) * (y + 1) % 2 == 0:
                    self.arena[x, y] = -1

        # Starting positions
        start_positions = [(i % s.COLS, i // s.COLS) for i in range(s.COLS*s.ROWS) if ((i % s.COLS)+1) * ((i // s.COLS)+1) % 2 == 1]
        start_positions = np.random.choice(start_positions, len(self.agents), False)
        random.shuffle(start_positions)
        for (x,y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if self.arena[xx, yy] == 1:
                    self.arena[xx, yy] = 0

        # Reset agents and distribute starting positions
        for agent in self.agents.values():
            self.active_agents.append(agent)
            agent.x, agent.y = start_positions.pop()
            agent.start_round()


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

                # Create explosion
                self.explosions.append(Explosion(blast_coords, None, bomb.owner, s.EXPLOSION_TIMER))
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
                        else:
                            a.is_suicide_bomber = True
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
                and len(self.bombs) + len(self.explosions) == 0):
            return True

        if self.current_step >= s.MAX_STEPS:
            return True

        return False

    def get_winner_loser(self):
        score_max = max([v.score for k, v in self.agents.items()])
        players_with_max_score = len([k for k, v in self.agents.items() if v.score == score_max])
        winner = []
        loser = []
        if self.phase == 1:
            for k, v in self.agents.items():
                if v.score == score_max and players_with_max_score == 1:
                    winner.append(k)
                elif v.score != score_max:
                    loser.append(k)
        return winner, loser

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
    def get_observation_from_game_state(game_state, agent_ids, current_round):
        pos = game_state['self'][3]


        field = game_state['field']
        walls = np.where(field == -1, 1, 0)[1:-1, 1:-1]
        free = np.where(field == 0, 1, 0)[1:-1, 1:-1]
        crates = np.where(field == 1, 1, 0)[1:-1, 1:-1]

        player = np.zeros(field.shape, dtype=int)
        player[game_state['self'][3]] = 1
        player = player[1:-1, 1:-1]

        #scores = np.zeros(4)
        #scores[0] = game_state['self'][1]
        scores = {game_state['self'][0]: 0}
        scores.update({a.name: 0 for a in agent_ids if a != game_state['self'][0]})
        alives = {a.name: 0 for a in agent_ids if a != game_state['self'][0]}
        scores[game_state['self'][0]] = game_state['self'][1]

        opponents = np.zeros(field.shape, dtype=int)
        #opp_alive = np.zeros(3)
        for i, o in enumerate(game_state['others']):
            opponents[o[3]] = 1
            #opp_alive[i] = 1
            alives[o[0]] = 1
            #scores[i + 1] = o[1]
            scores[o[0]] = o[1]
        #score_alive[game_state[0]] = (1, game_state['self'][1])
        opponents = opponents[1:-1, 1:-1]

        bombs = np.zeros(field.shape)
        for b in game_state['bombs']:
            bombs[b[0]] = b[1]

        bombs = bombs[1:-1, 1:-1]

        explosions = game_state['explosion_map'][1:-1, 1:-1]

        out = np.stack((walls, free, crates, bombs / 4., player, opponents, explosions / 2.), axis=2)
        # out = {'walls': walls, 'free': free, 'crates': crates, 'coins': coins, 'bombs': bombs, 'player': player,
        #        'opponents': opponents, 'explosions': explosions, 'scores': scores / 100., 'current_round': np.array([current_round / 400.])}
        return out, np.array([score for score in scores.values()]) / 24., np.array([alive for alive in alives.values()]), np.array([current_round / 400.])
