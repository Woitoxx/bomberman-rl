import copy
import os
import random

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import Callback
from ray.tune.logger import pretty_print
from stable_baselines3.ppo import *
from ray.rllib.agents.pg import PGTrainer
from training.bomberman_multi_env import BombermanEnv
from ray import tune
import numpy as np
from training.callbacks import MyCallbacks
from training.tfnet import ComplexInputNetwork

if __name__ == '__main__':
    phase = 0
    ray.init(object_store_memory=6000000000)
    env = BombermanEnv([f'agent_{i}' for i in range(4)])

    ModelCatalog.register_custom_model("custom_model", ComplexInputNetwork)
    tune.register_env('BomberMan-v0', lambda c: BombermanEnv([f'agent_{i}' for i in range(4)]))


    def policy_mapping_fn(agent_id):
        #if phase == 0:
        #    return "policy_01"
        #else:
        if agent_id.startswith("agent_0") or np.random.rand() > 0.2:
            return "policy_01"  # Choose 01 policy for agent_01
        else:
            return f'policy_{np.random.randint(2,12):02d}'
        '''
        if agent_id.startswith("agent_1"):
            return "policy_02"
        if agent_id.startswith("agent_2"):
            return "policy_03"
        if agent_id.startswith("agent_3"):
            return "policy_04"
        '''
        #else:
        #    return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"], 1,
        #                            p=[.8, .067, .067, .066])[0]

    def copy_weights(src_policy, dest_policy):
        P0key_P1val = {}  # temp storage with "policy_0" keys & "policy_1" values
        for (k, v), (k2, v2) in zip(dest_policy.get_weights().items(),
                                    src_policy.items()):
            P0key_P1val[k] = v2

        # set weights
        dest_policy.set_weights(P0key_P1val)

    def train(config, checkpoint_dir=None):
        global phase
        current_policy = None
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        trainer.restore('C:\\Users\\Florian\\ray_results\\PPO_BomberMan-v0_2021-03-03_14-01-29vzgtmiq9\\checkpoint_550\\checkpoint-550')
        iter = 0

        def update_policies_worker(ev, policyID=None):
            if policyID is not None:
                copy_weights(current_policy, ev.get_policy(policyID))
            else:
                ev.foreach_policy(lambda p, pid: copy_weights(current_policy, p))

        def update_phase(ev):
            ev.foreach_env(lambda e: e.set_phase(phase))

        phase = 0
        if phase == 0:
            trainer.workers.foreach_worker(update_phase)
            current_policy = trainer.get_policy("policy_01").get_weights()
            trainer.workers.foreach_worker(update_policies_worker)

        '''
        current_policy = copy.deepcopy(trainer.get_weights(["policy_01"])["policy_01"])
        model_pool.append(current_policy)
        phase = 2
        trainer.workers.foreach_worker(update_phase)
        trainer.workers.foreach_policy(update_policies)
        '''
        if not os.path.exists(f'./model-{iter}'):
            trainer.get_policy('policy_01').export_model(f'./model-{iter}')
        else:
            print("Model already saved.")
        while True:
            iter += 1
            result = trainer.train()
            if iter % 10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
            if iter % 100 == 0:
                if not os.path.exists(f'./model-{iter}'):
                    trainer.get_policy('policy_01').export_model(f'./model-{iter}')
                else:
                    print("model already saved")
            #reporter(**result)
            if phase >= 0 and iter % 5 == 0:
                current_policy = trainer.get_policy("policy_01").get_weights()
                policy_to_update = f'policy_{(iter//5)%10+2:02d}'
                print(f'Updating policy {policy_to_update}')
                trainer.workers.foreach_worker(lambda w: update_policies_worker(w,policy_to_update))
            '''
            if phase == 1 and result["policy_reward_mean"]["policy_01"] > 2:
                print(f'Phase 2 now.')
                phase = 2
                trainer.workers.foreach_worker(update_phase)
            '''

            if phase == 1 and result["policy_reward_mean"]["policy_01"] > 3:
                print(f'Phase 2 now.')
                phase = 2
                trainer.workers.foreach_worker(update_phase)
                #trainer.config['gamma'] = 0.995

            if phase == 0 and result["policy_reward_mean"]["policy_01"] > 3.5:
                print(f'Phase 1 now.')
                phase = 1
                trainer.workers.foreach_worker(update_phase)
    '''
    train(config={
        'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.99,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.001,
            'train_batch_size': 16384,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 6,
            'num_cpus_per_worker': 4,
            'num_workers': 2,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 1,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 15, "conv_filters": [[48, [5, 5], 2], [64, [3, 3], 2], [64, [3, 3], 2]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                     # "fcnet_hiddens": [256,256],
                # "vf_share_layers": 'true'
                 },
            'rollout_fragment_length': 512,
            'batch_mode': 'complete_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-4,
            'log_level': 'INFO',
            'framework': 'tf',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    f"policy_{p:02d}": (None, env.observation_space, env.action_space, {}) for p in range(50)
                },
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
    }, )
    '''
    tune.run(
        train,
        name='PPO',
        config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.995,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.003,
            'train_batch_size': 8192,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 5,
            'num_workers': 4,
            #'num_cpus_per_worker': 1,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 4,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 15, "conv_filters": [[48, [5, 5], 2], [64, [3, 3], 2], [64, [3, 3], 2]],
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
            'lr': 3e-4,
            'log_level': 'INFO',
            'framework': 'tf',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    f"policy_{p:02d}": (None, env.observation_space, env.action_space, {}) for p in range(1, 12)
                },
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn': policy_mapping_fn,
            },
        },
        resources_per_trial={'gpu': 1},
    )
    '''

    train(config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.99,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.001,
            'train_batch_size': 1024,
            'sgd_minibatch_size': 128,
            'shuffle_sequences': True,
            'num_sgd_iter': 6,
            'num_workers': 0,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 1,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 17, "conv_filters": [[32, [7, 7], 3], [64, [4, 4], 2], [64, [3, 3], 1]],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [512],
                "post_fcnet_activation": "relu",
                # "fcnet_hiddens": [256,256],
                # "vf_share_layers": 'true'},
                # Extra kwargs to be passed to your model's c'tor.
            },
            'rollout_fragment_length': 1024,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 1e-4,
            'log_level': 'INFO',
            'framework': 'tfe',
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
                    lambda a: policy_mapping_fn(a, phase),
            },
        },)

    '''
