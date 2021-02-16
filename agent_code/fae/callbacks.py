import ray
from ray import tune
from ray.rllib.agents.ppo import ppo

from training.bomberman_multi_env import BombermanEnv, Agent


def setup(self):
    ray.init()
    env = BombermanEnv(4)
    tune.register_env('BomberMan-v0', lambda cfg: BombermanEnv(4))
    agent = ppo.PPOTrainer(config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            "use_gae": True,
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'vf_clip_param': 10.0,
            'entropy_coeff': 0.01,
            'train_batch_size': 40,
            'sgd_minibatch_size': 40,
            'shuffle_sequences': True,
            'num_sgd_iter': 10,
            'num_workers': 0,
            'num_envs_per_worker': 1,
            'rollout_fragment_length': 10,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'vf_share_layers': 'true',
            'num_gpus': 0,
            'lr': 2.5e-4,
            'log_level': 'INFO',
            'framework': 'torch',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    "policy_01": (None, env.observation_space, env.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                    "policy_03": (None, env.observation_space, env.action_space, {}),
                    "policy_04": (None, env.observation_space, env.action_space, {})
                },
            },
    })

    # trainer = PGTrainer(config)
    # trainer.train()
    agent.restore('./agent')
    self.agent = agent
    self.env = env
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    obs = self.env.get_observation_from_game_state(game_state)
    action = self.agent.compute_action(obs, policy_id='policy_01')
    game_state['user_input'] = self.env.available_actions[action]
    return game_state['user_input']
