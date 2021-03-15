import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def setup(self):
    model_path = './model'
    self.available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    model = tf.saved_model.load(model_path)

    #Action output
    self.agent = model.signatures['serving_default'].prune("policy_01/Placeholder:0","policy_01/cond_1/Merge:0")

    #Actions distribution output
    #self.agent = model.signatures['serving_default'].prune("policy_01/Placeholder:0", "policy_01/functional_7/logits/BiasAdd:0")#"policy_01/cond_1/Merge:0")


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    current_step = game_state['step'] -1
    if current_step == 0:
        self.agents = [game_state['self'][0]]+[o[0] for o in game_state['others']]
    orig_obs = get_observation_from_game_state(game_state, self.agents)
    obs = tf.cast(np.concatenate([orig_obs[0].flatten(),orig_obs[1],orig_obs[2],orig_obs[3]]).reshape(1,-1),dtype=tf.float32)

    a = self.agent(obs)

    #print(a)
    #a = tf.nn.softmax(a)
    #a = tf.argmax(a,1)
    a = np.array(a)[0]
    game_state['user_input'] = self.available_actions[a]
    #print(game_state['user_input'])
    return game_state['user_input']


def get_observation_from_game_state(game_state, agent_ids):
    field = game_state['field']
    walls = np.where(field == -1, 1, 0)[1:-1, 1:-1]
    free = np.where(field == 0, 1, 0)[1:-1, 1:-1]
    crates = np.where(field == 1, 1, 0)[1:-1, 1:-1]

    player = np.zeros(field.shape, dtype=int)
    player[game_state['self'][3]] = 1
    player = player[1:-1, 1:-1]

    # scores = np.zeros(4)
    # scores[0] = game_state['self'][1]
    scores = {game_state['self'][0]: 0}
    scores.update({a: 0 for a in agent_ids if a != game_state['self'][0]})
    alives = {a: 0 for a in agent_ids if a != game_state['self'][0]}
    scores[game_state['self'][0]] = game_state['self'][1]

    opponents = np.zeros(field.shape, dtype=int)
    # opp_alive = np.zeros(3)
    for i, o in enumerate(game_state['others']):
        opponents[o[3]] = 1
        # opp_alive[i] = 1
        alives[o[0]] = 1
        # scores[i + 1] = o[1]
        scores[o[0]] = o[1]
    # score_alive[game_state[0]] = (1, game_state['self'][1])
    opponents = opponents[1:-1, 1:-1]
    coins = np.zeros(field.shape, dtype=int)
    for c in game_state['coins']:
        coins[c] = 1
    coins = coins[1:-1, 1:-1]

    bombs = np.zeros(field.shape)
    for b in game_state['bombs']:
        bombs[b[0]] = b[1]

    bombs = bombs[1:-1, 1:-1]

    explosions = game_state['explosion_map'][1:-1, 1:-1]

    all_ones = np.ones_like(free)

    out = np.stack((walls, free, crates, coins, bombs / 4., player, opponents, explosions / 2., all_ones), axis=2)
    return out, np.array([score for score in scores.values()]) / 24., np.array(
        [alive for alive in alives.values()]), np.array([game_state['step'] / 400.])
