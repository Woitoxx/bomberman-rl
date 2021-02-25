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
            'gamma': 0.99,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.001,
            'train_batch_size': 32768,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 6,
            'num_workers': 4,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 8,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "dim": 17, "conv_filters": [[128, [5, 5], 1], [256, [3, 3], 2], [256, [3, 3], 2], [512, [3, 3], 2], [64, [3, 3], 1]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                     # "fcnet_hiddens": [256,256],
                # "vf_share_layers": 'true'
                 },
            'rollout_fragment_length': 512,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 1e-4,
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
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    lambda x : "policy_01",
            },
        },
    )

    # trainer = PGTrainer(config)
    # trainer.train()
    agent.restore('./checkpoint-640')
    self.agent = agent
    self.env = env
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    obs = self.env.get_observation_from_game_state(game_state,0)
    action = self.agent.compute_action(obs, policy_id='policy_01')
    game_state['user_input'] = self.env.available_actions[action]
    return game_state['user_input']
