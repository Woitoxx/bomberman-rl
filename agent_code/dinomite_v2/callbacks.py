import os

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def setup(self):
    model_path = './model-11250'
    self.available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    model = tf.saved_model.load(model_path)

    #Action output
    #self.agent = model.signatures['serving_default']#.prune("policy_01/Placeholder:0",fetches=["policy_01/cond_1/Merge:0",'policy_01/add:0'])

    #Actions distribution output
    self.agent = model.signatures['serving_default'].prune("policy_01/Placeholder:0", "policy_01/functional_7/logits/BiasAdd:0")#"policy_01/cond_1/Merge:0")


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    current_step = game_state['step'] -1
    if current_step == 0:
        self.agents = [game_state['self'][0]]+[o[0] for o in game_state['others']]
    orig_obs = get_observation_from_game_state(game_state, self.agents)
    obs = tf.cast(np.concatenate([orig_obs[0].flatten(),orig_obs[1]]).reshape(1,-1),dtype=tf.float32)
    a = self.agent(obs)
    '''
    a = self.agent(observations=obs,is_training=tf.constant(False, tf.bool),
                   prev_action=tf.constant(0, tf.int64),
                   prev_reward=tf.constant(0, tf.float32),
                   seq_lens=tf.constant(0,tf.int32),
                    timestep=tf.constant(0, tf.int64))
    '''
    #print(a)
    mask = tf.cast(np.abs(orig_obs[-1]-1), dtype=tf.float32)
    a = a + mask *tf.float32.min
    a = tf.nn.softmax(a)
    #a = np.array(a).reshape(-1)
    #action = np.random.choice(np.arange(6), p=a)
    #game_state['user_input'] =  self.available_actions[action]
    a = tf.argmax(a,1)
    a = np.array(a)[0]
    game_state['user_input'] = self.available_actions[a]
    #print(game_state['user_input'])
    return game_state['user_input']

def get_available_actions_for_agent(agent, obs_arena):
    def tile_is_free(x, y):
        is_free = 0 <= x <= 14 and 0 <= y <= 14 and obs_arena[1][x, y] == 1
        if is_free:
            bombs = obs_arena[4:8]
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


def get_observation_from_game_state(game_state, agent_ids):
    field = game_state['field']
    walls = np.where(field == -1, 1, 0)[None, 1:-1, 1:-1]
    free = np.where(field == 0, 1, 0)[None, 1:-1, 1:-1]
    crates = np.where(field == 1, 1, 0)[None, 1:-1, 1:-1]

    player = np.zeros(field.shape, dtype=int)
    player[game_state['self'][3]] = 1
    player = player[None, 1:-1, 1:-1]

    #scores = np.zeros(4)
    #scores[0] = game_state['self'][1]
    scores = {game_state['self'][0]: 0}
    scores.update({a: 0 for a in agent_ids if a != game_state['self'][0]})
    #alives = {a: 0 for a in agent_ids if a != game_state['self'][0]}
    scores[game_state['self'][0]] = game_state['self'][1]

    opponents = np.zeros((3,field.shape[0], field.shape[1]), dtype=int)
    #opp_alive = np.zeros(3)
    for i, o in enumerate(game_state['others']):
        opponents[i,o[3][0],o[3][1]] = 1
        #opp_alive[i] = 1
        #alives[o[0]] = 1
        #scores[i + 1] = o[1]
        scores[o[0]] = o[1]
    #score_alive[game_state[0]] = (1, game_state['self'][1])
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
    #out = np.stack((walls, free, crates, coins, bombs, player, opponents, explosions, all_ones), axis=2)
    # out = {'walls': walls, 'free': free, 'crates': crates, 'coins': coins, 'bombs': bombs, 'player': player,
    #        'opponents': opponents, 'explosions': explosions, 'scores': scores / 100., 'current_round': np.array([current_round / 400.])}
    return np.moveaxis(out, 0 , 2), \
           np.array([score for score in scores.values()]) / 24.,\
           get_available_actions_for_agent(game_state['self'], out)
           #np.array([game_state['current_step'] / 400.0]),\
