import os

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

model_path = './model-1000/'

def setup(self):
    self.available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    model = tf.saved_model.load(f'{model_path}main')
    model1 = tf.saved_model.load(f'{model_path}collect')
    model2 = tf.saved_model.load(f'{model_path}destroy')
    model3 = tf.saved_model.load(f'{model_path}kill')
    np.set_printoptions(suppress=True)
    #Action output
    #self.agent = model.signatures['serving_default']#.prune("policy_01/Placeholder:0",fetches=["policy_01/cond_1/Merge:0",'policy_01/add:0'])

    #Actions distribution output
    self.agent_main = model.signatures['serving_default'].prune("policy_01/Placeholder:0", "policy_01/add:0")#"policy_01/cond_1/Merge:0")
    self.collect_policy = model1.signatures['serving_default'].prune("policy_collect/Placeholder:0",
                                                                "policy_collect/add:0")  # "policy_01/cond_1/Merge:0")
    self.destroy_policy = model2.signatures['serving_default'].prune("policy_destroy/Placeholder:0",
                                                                "policy_destroy/add:0")  # "policy_01/cond_1/Merge:0")
    self.kill_policy = model3.signatures['serving_default'].prune("policy_kill/Placeholder:0",
                                                                "policy_kill/add:0")  # "policy_01/cond_1/Merge:0")



def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    current_step = game_state['step'] -1
    if current_step == 0:
        self.agents = [game_state['self'][0]]+[o[0] for o in game_state['others']]
    orig_obs = get_high_level_observation_from_game_state(game_state, self.agents)
    obs = tf.cast(np.concatenate([orig_obs[0].flatten(),orig_obs[1],orig_obs[2]]).reshape(1,-1),dtype=tf.float32)
    a = self.agent_main(obs)
    mask = tf.maximum(tf.math.log(tf.cast(orig_obs[-1], dtype=tf.float32)), tf.float32.min)
    a += mask
    a = tf.nn.softmax(a)
    print(a)
    #a = tf.argmax(a,1)
    a = np.random.choice(3, 1, p=np.array(a).reshape(-1))
    if a == 0:
        orig_obs = get_collect_observation_from_game_state(game_state)
        obs = tf.cast(np.concatenate([orig_obs[0].flatten(), orig_obs[1]]).reshape(1, -1),
                      dtype=tf.float32)
        a = self.collect_policy(obs)
        print(f'Collect')

    elif a == 1:
        orig_obs = get_destroy_observation_from_game_state(game_state)
        obs = tf.cast(np.concatenate([orig_obs[0].flatten(), orig_obs[1]]).reshape(1, -1),
                      dtype=tf.float32)
        a = self.destroy_policy(obs)
        print(f'Destroy')

    elif a == 2:
        orig_obs = get_kill_observation_from_game_state(game_state)
        obs = tf.cast(np.concatenate([orig_obs[0].flatten(), orig_obs[1]]).reshape(1, -1),
                      dtype=tf.float32)
        a = self.kill_policy(obs)
        print(f'Kill')

    mask = tf.maximum(tf.math.log(tf.cast(orig_obs[-1], dtype=tf.float32)), tf.float32.min)
    a += mask
    a = tf.nn.softmax(a)
    print(np.array(a))
    a = tf.argmax(a, 1)
    a = np.array(a)[0]

    game_state['user_input'] = self.available_actions[a]
    #print(game_state['user_input'])
    return game_state['user_input']


def get_high_level_observation_from_game_state(game_state, agent_ids):
    field = game_state['field']
    walls = np.where(field == -1, 1, 0)[None, 1:-1, 1:-1]
    free = np.where(field == 0, 1, 0)[None, 1:-1, 1:-1]
    crates = np.where(field == 1, 1, 0)[None, 1:-1, 1:-1]

    player = np.zeros(field.shape, dtype=int)
    player[game_state['self'][3]] = 1
    player = player[None, 1:-1, 1:-1]

    scores = {game_state['self'][0]: 0}
    scores.update({a: 0 for a in agent_ids if a != game_state['self'][0]})
    scores[game_state['self'][0]] = game_state['self'][1]

    opponents = np.zeros((3,field.shape[0], field.shape[1]), dtype=int)
    for i, o in enumerate(game_state['others']):
        opponents[i,o[3][0],o[3][1]] = 1
        scores[o[0]] = o[1]
    opponents = opponents[:, 1:-1, 1:-1]
    coins = np.zeros(field.shape, dtype=int)
    for c in game_state['coins']:
        coins[c] = 1
    coins = coins[None, 1:-1, 1:-1]

    bombs = np.zeros((4,field.shape[0], field.shape[1]), dtype=int)
    for b in game_state['bombs']:
        bombs[3-b[1],b[0][0],b[0][1]] = 1

    bombs = bombs[:,1:-1, 1:-1]

    explosions = game_state['explosion_map'][None, 1:-1, 1:-1]

    all_ones = np.ones_like(free)

    out = np.vstack((walls, free, crates, coins, bombs, player, opponents, explosions, all_ones))
    legal_actions = np.array([len(game_state['coins'])>0, np.any(crates), len(game_state['others'])>0], dtype=int)
    return np.moveaxis(out, 0 , 2), \
           np.array([score for score in scores.values()]) / 24.,\
           legal_actions
           #get_available_actions_for_agent(game_state['self'], out)
           #np.array([game_state['current_step'] / 400.0]),\


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


def get_collect_observation_from_game_state(game_state):
    obs = get_common_observation_from_game_state(game_state)

    field = game_state['field']
    blocked = np.where(field != 0, 1, 0)[None, 1:-1, 1:-1]

    opponents = np.zeros((field.shape[0], field.shape[1]), dtype=int)
    for o in game_state['others']:
        opponents[o[3][0], o[3][1]] = 1
    opponents = opponents[None, 1:-1, 1:-1]

    coins = np.zeros(field.shape, dtype=int)

    for c in game_state['coins']:
        coins[c] = 1
    coins = coins[None, 1:-1, 1:-1]

    legal_actions = get_available_actions_for_agent(game_state['self'], obs)
    legal_actions[4] = 0   # Bomb not available
    return np.moveaxis(np.vstack((obs, blocked, coins, opponents)), 0, 2), legal_actions


def get_destroy_observation_from_game_state(game_state):
    obs = get_common_observation_from_game_state(game_state)
    field = game_state['field']
    walls = np.where(field == -1, 1, 0)[None, 1:-1, 1:-1]
    crates = np.where(field == 1, 1, 0)[None, 1:-1, 1:-1]
    opponents = np.zeros((field.shape[0], field.shape[1]), dtype=int)
    for o in game_state['others']:
        opponents[o[3][0], o[3][1]] = 1
    opponents = opponents[None, 1:-1, 1:-1]

    legal_actions = get_available_actions_for_agent(game_state['self'], obs)
    return np.moveaxis(np.vstack((obs, walls, crates, opponents)), 0, 2), legal_actions


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
