import random

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.util.client import ray

from training.bomberman_multi_env import BombermanEnv
from training.hierarchical_learning.coin_env import BombermanCoinEnv
from training.hierarchical_learning.hierarchical_bomberman_multi_env import HierarchicalBombermanMultiEnv
from training.tfnet import ComplexInputNetwork

if __name__ == '__main__':
    ray.init()
    env = BombermanCoinEnv()
    ModelCatalog.register_custom_model("custom_model", ComplexInputNetwork)
    tune.register_env('BomberMan-v0', lambda c: BombermanCoinEnv())

    def train(config, checkpoint_dir=None):
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        iter = 1
        while True:
            result = trainer.train()
            if iter % 10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
            iter += 1

    train(config={
        'env': 'BomberMan-v0',
            "use_critic": True,
            'horizon' : 500,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.99,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.001,
            'train_batch_size': 8192,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 6,
            'num_cpus_per_worker': 4,
            'num_workers': 0,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 16,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 15, "conv_filters": [[16, [5, 5], 2], [32, [3, 3], 2], [32, [3, 3], 2]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                     # "fcnet_hiddens": [256,256],
                # "vf_share_layers": 'true'
                 },
            'rollout_fragment_length': 500,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-4,
            'log_level': 'INFO',
            'framework': 'tf',
            #'simple_optimizer': args.simple,
    },)