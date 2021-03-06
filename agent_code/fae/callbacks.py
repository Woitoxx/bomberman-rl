import numpy as np
from training.bomberman_multi_env import get_observation_from_game_state
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def setup(self):
    model_path = '/home/florian/Code/bomberman-rl/agent_code/fae/model-2000'
    agent_ids = {'fae','random_agent_0','random_agent_1','random_agent_2'}
    self.available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    model = tf.saved_model.load(model_path)
    #self.agent = model.signatures['serving_default'].prune("policy_01/Placeholder:0","policy_01/cond_1/Merge:0")
    #self.agent = model.signatures['serving_default'].prune("policy_01/Placeholder:0", "policy_01/functional_5/logits/BiasAdd:0")
    self.agent = model.signatures['serving_default'].prune("policy_01/Placeholder:0", "policy_01/cond_1/Merge:0")


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    current_step = game_state['step'] -1
    if current_step == 0:
        self.agents = [game_state['self'][0]]+[o[0] for o in game_state['others']]
    orig_obs = get_observation_from_game_state(game_state, self.agents, current_step)
    board = np.moveaxis(orig_obs[0],2,0)
    obs = tf.cast(np.concatenate([orig_obs[0].flatten(),orig_obs[1],orig_obs[2],orig_obs[3]]).reshape(1,-1),dtype=tf.float32)

    a = self.agent(obs)

    #print(a)
    #a = tf.nn.softmax(a)
    #a = tf.argmax(a,1)
    a = np.array(a)[0]
    game_state['user_input'] = self.available_actions[a]

    return game_state['user_input']
