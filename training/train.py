import ray
from ray.tune import Callback
from stable_baselines3.ppo import *
from ray.rllib.agents.pg import PGTrainer
from training.bomberman_multi_env import BombermanEnv
from ray import tune
import numpy as np

if __name__ == '__main__':
    env = BombermanEnv(4)
    tune.register_env('BomberMan-v0', lambda cfg: BombermanEnv(4))
    config = {
        "env": 'BomberMan-v0',
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 100,
        "train_batch_size": 100,
        "model": {
            # By default, the MODEL_DEFAULTS dict above will be used.

            # Change individual keys in that dict by overriding them, e.g.
            "fcnet_hiddens": [512, 512],
        },
        "multiagent": {
            "policies_to_train": ["test"],
            "policies": {
                "test": (None, env.observation_space, env.action_space, {})
            },
            "policy_mapping_fn": lambda p: "test",
        },
        "framework": "torch",
    }

    trainer = PGTrainer(config)
    trainer.train()
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("agent_01"):
            return "policy_01"  # Choose 01 policy for agent_01
        else:
            return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"], 1,
                                    p=[.8, .2 / 3, .2 / 3, .2 / 3])[0]


    class UpdatePolicies(Callback):
        def on_trial_result(self, iteration, trials,
                        trial, result, **info):
            p3 = ray.get(trials[0].runner.get_weights.remote(["policy_03"]))["policy_03"]
            p2 = ray.get(trials[0].runner.get_weights.remote(["policy_02"]))["policy_02"]
            p1 = ray.get(trials[0].runner.get_weights.remote(["policy_01"]))["policy_01"]
            trials[0].runner.set_weights.remote({"policy_04":p3,
                                                 "policy_03":p2,
                                                 "policy_02":p1})

        def on_step_begin(self, iteration, trials, **info):
            print()

    tune.run(
        'PPO',
        stop={'training_iteration': 1000},
        checkpoint_freq=50,
        callbacks=[UpdatePolicies()],
        config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            "use_gae": True,
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'vf_clip_param': 10.0,
            'entropy_coeff': 0.01,
            'train_batch_size': 10000,
            'sgd_minibatch_size': 500,
            'shuffle_sequences': True,
            'num_sgd_iter': 10,
            'num_workers': 0,
            'num_envs_per_worker': 10,
            'rollout_fragment_length': 400,
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
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
        },
    )
