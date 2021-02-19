import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import Callback
from ray.tune.logger import pretty_print
from stable_baselines3.ppo import *
from ray.rllib.agents.pg import PGTrainer
from training.bomberman_multi_env import BombermanEnv
from ray import tune
import numpy as np

if __name__ == '__main__':
    ray.init()
    env = BombermanEnv(4)
    tune.register_env('BomberMan-v0', lambda c: BombermanEnv(4))
    '''
    cfg = {
        "env": 'BomberMan-v0',
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 512,
        "train_batch_size": 512,
        "model": {"dim": 17, "conv_filters": [[16, [5, 5], 2], [32, [3, 3], 2], [32, [3, 3], 2], [512, [3, 3], 1]]},
        "multiagent": {
            "policies_to_train": ["test"],
            "policies": {
                "test": (None, env.observation_space, env.action_space, {})
            },
            "policy_mapping_fn": lambda p: "test",
        },
        "framework": "torch",
            'log_level': 'INFO',
    }

    t = PGTrainer(cfg)
    #while True:
    #    result = trainer.train()
    #    print(pretty_print(result))
'''
    def policy_mapping_fn(agent_id):
        if agent_id.startswith("agent_0"):
            return "policy_01"  # Choose 01 policy for agent_01
        else:
            return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"], 1,
                                    p=[.8, .067, .067, .066])[0]


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


    def train(config, checkpoint_dir=None):
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        #trainer.restore('C:\\Users\\Florian\\ray_results\\PPO_BomberMan-v0_2021-02-18_21-11-43wckud60n\\checkpoint_10\\checkpoint-10')
        iter = 1

        def do_stuff(ev):
            ev.foreach_env(lambda env: env.set_phase(phase))
            p3 = ev.get_weights(["policy_03"])["policy_03"]
            p2 = ev.get_weights(["policy_02"])["policy_02"]
            p1 = ev.get_weights(["policy_01"])["policy_01"]

            ev.set_weights({"policy_04": p3,
                            "policy_03": p2,
                            "policy_02": p1})

        while True:
            result = trainer.train()
            if iter % 10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
            #reporter(**result)
            if result["episode_reward_mean"] > 10:
                phase = 1
            else:
                phase = 0
            trainer.workers.foreach_worker(do_stuff)
            '''
            P0key_P1val = {}  # temp storage with "policy_0" keys & "policy_1" values
            for (k, v), (k2, v2) in zip(trainer.get_policy("policy_01").get_weights().items(),
                                        trainer.get_policy("policy_02").get_weights().items()):
                P0key_P1val[k] = v2

            # set weights
            trainer.set_weights({"policy_01": P0key_P1val,  # weights or values from "policy_1" with "policy_0" keys
                                 "policy_02": trainer.get_policy("policy_02").get_weights()  # no change
                                 })

            # To check
            for (k, v), (k2, v2) in zip(trainer.get_policy("policy_01").get_weights().items(),
                                        trainer.get_policy("policy_02").get_weights().items()):
                assert (v == v2).all()
            '''
            iter+=1


    '''
    train(config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            "use_gae": True,
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.01,
            'train_batch_size': 40000,
            'sgd_minibatch_size': 512,
            'shuffle_sequences': True,
            'num_sgd_iter': 10,
            'num_workers': 0,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 10,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {"dim": 17, "conv_filters": [[16, [5, 5], 2], [32, [3, 3], 2], [32, [3, 3], 2], [512, [3, 3], 1]],
                      "vf_share_layers": 'true'},
            'rollout_fragment_length': 1000,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-5,
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
                #"policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
        },)
'''
    tune.run(
        train,
        name='PPO',
        config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.995,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.01,
            'train_batch_size': 40000,
            'sgd_minibatch_size': 512,
            'shuffle_sequences': True,
            'num_sgd_iter': 10,
            'num_workers': 4,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 10,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {"dim": 17, "conv_filters": [[16, [5, 5], 2], [32, [3, 3], 2], [32, [3, 3], 2], [512, [3, 3], 1]],
                      "vf_share_layers": 'true'},
            'rollout_fragment_length': 1000,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 1e-5,
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
        stop={'training_iteration': 1000},
        resources_per_trial={'gpu': 1}
        #callbacks=[UpdatePolicies()],

    )

'''
    train(config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            "use_gae": True,
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.01,
            'train_batch_size': 51200,
            'sgd_minibatch_size': 512,
            'shuffle_sequences': True,
            'num_sgd_iter': 10,
            'num_workers': 4,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 10,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {"dim": 17, "conv_filters": [[16, [5, 5], 2], [32, [3, 3], 2], [32, [3, 3], 2], [512, [3, 3], 1]],
                      "vf_share_layers": 'true'},
            'rollout_fragment_length': 1000,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-5,
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
        })
'''
