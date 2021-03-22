import random
from datetime import datetime
from typing import List, Tuple
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

import settings as s
from training.items import Coin, Explosion, Bomb

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
        self.crates_destroyed = 0
        self.is_suicide_bomber = False
        self.aux_score = 0
        self.step_reward = 0
        self.penalty = 0

        self.last_game_state = None
        self.last_action = None

    def __str__(self):
        return f'Agent: {self.name} | Score: {self.score} | Alive: {not self.dead} | Bombs left: {self.bombs_left} | Position: {self.x}, {self.y}'

    def start_round(self):
        self.dead = False
        self.score = 0
        self.trophies = []

        self.bombs_left = True
        self.crates_destroyed = 0
        self.aux_score = 0
        self.step_reward = 0
        self.penalty = 0
        self.is_suicide_bomber = False

        self.last_game_state = None
        self.last_action = None

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return self.name, self.score, self.bombs_left, (self.x, self.y), self.dead

    def update_score(self, delta):
        """Add delta to both the current round's score and the total score."""
        self.score += delta
        self.total_score += delta

    def store_game_state(self, game_state):
        self.last_game_state = game_state


class BombermanArenaEnv(MultiAgentEnv):
    running: bool = False
    current_step: int
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    round_id: str
    available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

    def __init__(self, agent_ids):

        self.round = 0
        self.running = False
        self.agents = {}
        self.agents_last_obs = {}
        self.phase = 0
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(15, 15, 14)),
                                               spaces.Box(low=0, high=1, shape=(4,)),
                                               #spaces.MultiBinary(3),
                                               #spaces.Box(low=0, high=1, shape=(1,)),
                                               spaces.MultiBinary(len(self.available_actions))))
        # spaces.Dict({'walls': spaces.MultiBinary(tiles),
        # 'free': spaces.MultiBinary(tiles),
        # 'crates': spaces.MultiBinary(tiles),
        # 'player': spaces.MultiBinary(tiles),
        # 'opponents': spaces.MultiBinary(tiles),
        # 'coins': spaces.MultiBinary(tiles),
        # 'bombs': spaces.Box(low=0, high=4, shape=(17, 17)),
        # 'explosions': spaces.Box(low=0, high=2, shape=(17, 17)),
        # 'scores' : spaces.Box(low=0, high=1, shape=(4,)),
        # 'current_round': spaces.Box(low=0, high=1, shape=(1,))
        # })
        # spaces.Box(low=0, high=4, shape=(17,17,8))
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

        # Determine which agent gets to act first
        random.shuffle(self.active_agents)

        for agent in self.active_agents:
            self.perform_agent_action(agent, self.available_actions[action_dict[agent.name]])

        # Update arena after handling actions from agents
        self.collect_coins()
        self.update_bombs()
        self.evaluate_explosions()

        # Set obs, reward, done, info for agents still alive
        # Agents that died during this step will get their next obs, reward, done, info later when the round finishes
        for agent in self.active_agents:
            rewards[agent.name] = agent.aux_score
            rewards[agent.name] -= np.average([v.aux_score for k, v in self.agents.items() if k != agent.name])
            agent.store_game_state(self.get_state_for_agent(agent))
            dones[agent.name] = False
            obs[agent.name] = get_kill_observation_from_game_state(agent.last_game_state)
            infos[agent.name] = agent.score

        for agent_name in action_dict.keys():
            if agent_name not in map(lambda a: a.name, self.active_agents):
                agent = self.agents[agent_name]
                agent.store_game_state(self.get_state_for_agent(agent))
                self.agents_last_obs[agent_name] = get_kill_observation_from_game_state(agent.last_game_state)

        for agent in self.agents.values():
            if agent.dead:
                agent.penalty += agent.aux_score
                agent.penalty -= np.average([v.aux_score for k, v in self.agents.items() if k != agent.name])

        for agent in self.agents.values():
            agent.aux_score = 0

        if self.done():
            self.end_round()
            dones['__all__'] = True
            for a in self.agents.values():
                # Add observation for agents that died ealier
                if a not in self.active_agents:
                    rewards[a.name] = a.penalty
                    obs[a.name] = self.agents_last_obs[a.name]
                dones[a.name] = True
                infos[a.name] = a.score
        else:
            dones['__all__'] = False
        return obs, rewards, dones, infos

    def reset(self) -> MultiAgentDict:
        obs = {}
        self.new_round()
        self.agents_last_obs = {}
        for agent in self.active_agents:
            agent.store_game_state(self.get_state_for_agent(agent))
            obs[agent.name] = get_kill_observation_from_game_state(agent.last_game_state)
        return obs

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
            agent.aux_score = 0
            agent.crates_destroyed = 0
            agent.penalty = 0
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
            agent.aux_score += 0.001
        agent.last_action = action

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        a.update_score(s.REWARD_COIN)
                        #a.aux_score += 1

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
                        #bomb.owner.crates_destroyed += 1
                        #bomb.owner.aux_score += 0.001

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
                            explosion.owner.aux_score += 5
                        else:
                            a.is_suicide_bomber = True
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False

            # Progress countdown
            explosion.timer -= 1
        last_man_standing = len(self.active_agents) - len(agents_hit) == 0
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            #if not last_man_standing:
            #    a.aux_score -= 1
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
                #and all([not c.collectable for c in self.coins])
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
        for k, v in self.agents.items():
            if v.score == score_max and players_with_max_score < 4:
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


def get_available_actions_for_agent(agent, obs_arena):
    def tile_is_free(x, y):
        is_free = 0 <= x <= 14 and 0 <= y <= 14 and obs_arena[0][x, y] == 1
        if is_free:
            bombs = obs_arena[2:6]
            pos = bombs[:,x, y]
            is_free = is_free and not np.any(bombs[:,x, y])
        return is_free

    x, y = agent[3][0]-1, agent[3][1]-1
    action_mask = np.zeros(6, dtype=int)
    action_mask[0] = tile_is_free(x, y - 1)
    action_mask[1] = tile_is_free(x, y + 1)
    action_mask[2] = tile_is_free(x - 1, y)
    action_mask[3] = tile_is_free(x + 1, y)
    action_mask[4] = agent[2]
    action_mask[5] = 1
    return action_mask


def get_common_observation_from_game_state(game_state):
    field = game_state['field']
    free = np.where(field == 0, 1, 0)[None, 1:-1, 1:-1]

    player = np.zeros(field.shape, dtype=int)
    player[game_state['self'][3]] = 1
    player = player[None, 1:-1, 1:-1]

    bombs = np.zeros((4, field.shape[0], field.shape[1]), dtype=int)
    for b in game_state['bombs']:
        bombs[3 - b[1], b[0][0], b[0][1]] = 1

    bombs = bombs[:, 1:-1, 1:-1]

    explosions = game_state['explosion_map'][None, 1:-1, 1:-1]

    all_ones = np.ones_like(free)

    out = np.vstack((free, player, bombs,  explosions, all_ones))
    return out


def get_kill_observation_from_game_state(game_state):
    obs = get_common_observation_from_game_state(game_state)
    field = game_state['field']
    walls = np.where(field == -1, 1, 0)[None, 1:-1, 1:-1]
    opponents = np.zeros((3, field.shape[0], field.shape[1]), dtype=int)
    for i, o in enumerate(game_state['others']):
        opponents[i, o[3][0], o[3][1]] = 1
    opponents = opponents[:, 1:-1, 1:-1]

    crates = np.where(field == 1, 1, 0)[None, 1:-1, 1:-1]

    legal_actions = get_available_actions_for_agent(game_state['self'], obs)
    return np.moveaxis(np.vstack((obs, walls, crates, opponents)), 0, 2), legal_actions
