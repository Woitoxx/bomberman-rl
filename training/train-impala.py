import copy
import os
import random

import ray
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.models import ModelCatalog
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
        if agent_id.startswith("agent_0"):
            return "policy_01"  # Choose 01 policy for agent_01
        if agent_id.startswith("agent_1"):
            return "policy_02"
        if agent_id.startswith("agent_2"):
            return "policy_03"
        if agent_id.startswith("agent_3"):
            return "policy_04"
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
        model_pool = []
        global phase
        current_policy = None
        trainer = ImpalaTrainer(config=config, env='BomberMan-v0')
        #trainer.restore('C:\\Users\\Florian\\ray_results\\PPO_BomberMan-v0_2021-02-28_15-34-57we64cvop\\checkpoint_2430\\checkpoint-2430')
        iter = 1

        def update_policies(policy, policyID):
            if policyID != "policy_01":
                new_policy = current_policy if random.random() > 0.2 else random.choice(model_pool)
                copy_weights(new_policy, policy)

        def update_policies_worker(ev):
            ev.foreach_policy(update_policies)

        def update_phase(ev):
            ev.foreach_env(lambda e: e.set_phase(phase))

        phase = 0
        if phase == 0:
            trainer.workers.foreach_worker(update_phase)
            current_policy = trainer.get_policy("policy_01").get_weights()
            model_pool.append(copy.deepcopy(current_policy))
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
            if phase >= 0:
                current_policy = trainer.get_policy("policy_01").get_weights()
                #if result["policy_reward_mean"]["policy_01"] > 0.02 or len(model_pool) < 10:
                model_pool.append(copy.deepcopy(current_policy))
                if len(model_pool) > 100:
                    model_pool.pop(0)
                trainer.workers.foreach_worker(update_policies_worker)
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
                trainer.config['gamma'] = 0.995

            if phase == 0 and result["policy_reward_mean"]["policy_01"] > 3.5:
                print(f'Phase 1 now.')
                phase = 1
                trainer.workers.foreach_worker(update_phase)
            iter += 1

    train(config={
        'env': 'BomberMan-v0',
            "vtrace": True,
            "vtrace_clip_rho_threshold": 1.0,
            "vtrace_clip_pg_rho_threshold": 1.0,
            "rollout_fragment_length": 1024,
            "train_batch_size": 4096,
            "min_iter_time_s": 10,
            "num_workers": 0,
            "num_envs_per_worker" : 4,
            "num_gpus": 1,
            "num_data_loader_buffers": 1,
            "minibatch_buffer_size": 1,
            "num_sgd_iter": 1,
            "replay_proportion": 0.0,
            "replay_buffer_num_slots": 0,
            "learner_queue_size": 16,
            "learner_queue_timeout": 300,
            "max_sample_requests_in_flight_per_worker": 2,
            "broadcast_interval": 1,
            "num_aggregation_workers": 0,
            "grad_clip": 40.0,
            "opt_type": "adam",
            "lr": 0.0005,
            "lr_schedule": None,
            # rmsprop considered
            "decay": 0.99,
            "momentum": 0.0,
            "epsilon": 0.1,
            # balancing the three losses
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": None,

            # Callback for APPO to use to update KL, target network periodically.
            # The input to the callback is the learner fetches dict.
            "after_train_step": None,
            "model": {
                "custom_model": "custom_model",
                "dim": 15, "conv_filters": [[48, [5, 5], 2], [64, [3, 3], 2], [64, [3, 3], 2]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                     # "fcnet_hiddens": [256,256],
                # "vf_share_layers": 'true'
                 },

            'log_level': 'INFO',
            'framework': 'tf',
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
    }, )

    tune.run(
        train,
        name='PPO',
        config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.99,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.003,
            'train_batch_size': 32768,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 6,
            'num_workers': 0,
            'num_cpus_per_worker': 4,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 32,
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
            'rollout_fragment_length': 1024,
            'batch_mode': 'complete_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-4,
            'log_level': 'INFO',
            'framework': 'tf',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    "policy_01": (None, env.observation_space, env.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                    "policy_03": (None, env.observation_space, env.action_space, {}),
                    "policy_04": (None, env.observation_space, env.action_space, {})
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
