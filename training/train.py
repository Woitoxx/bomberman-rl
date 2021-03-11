import os
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from training.bomberman_multi_env import BombermanEnv
from ray import tune
from training.callbacks import MyCallbacks
from training.tfnet_2 import ComplexInputNetwork


if __name__ == '__main__':
    ray.init(
        _redis_max_memory=1024 * 1024 * 100,num_gpus=1, object_store_memory=10*2**30)
    env = BombermanEnv([f'agent_{i}' for i in range(4)])

    ModelCatalog.register_custom_model("custom_model", ComplexInputNetwork)
    tune.register_env('BomberMan-v0', lambda c: BombermanEnv([f'agent_{i}' for i in range(4)]))


    def policy_mapping_fn(agent_id):
        #if phase == 0:
        #return "policy_01"
        #else:
        if agent_id.startswith("agent_0"):# or np.random.rand() > 0.2:
            return "policy_01"  # Choose 01 policy for agent_01
        else:
            return "policy_02"

    def train(config, checkpoint_dir=None):
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        trainer.restore('C:\\Users\\Florian\\ray_results\\PPO_BomberMan-v0_2021-03-10_14-16-50n_4knahb\\checkpoint_002700\\checkpoint-2700')
        iter = 0

        def update_phase(ev):
            ev.foreach_env(lambda e: e.set_phase(phase))

        phase = 2
        trainer.workers.foreach_worker(update_phase)

        while True:
            iter += 1
            result = trainer.train()
            if iter % 250 == 0:
                if not os.path.exists(f'./model-{iter}'):
                    trainer.get_policy('policy_01').export_model(f'./model-{iter}')
                    trainer.get_policy('policy_01')
                else:
                    print("model already saved")
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


    train(config={
        'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.99,
            'kl_coeff': 0.2,
            'vf_loss_coeff' : 0.5,
            'clip_rewards': False,
            'entropy_coeff': 0.003,
            'train_batch_size': 32768,#49152,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 10,
            'num_workers': 2,
            'num_cpus_per_worker': 3,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 8,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 15,
                #"conv_filters": [[16, [5, 5], 1], [32, [3, 3], 2], [32, [3, 3], 1], [64, [3, 3], 2], [64, [3, 3], 1], [128, [3, 3], 2], [128, [3, 3], 1], [2, [1, 1], 1]],
                "conv_filters" : [[32, [5,5], 2], [32, [3,3], 2], [64, [3,3], 2], [128, [3,3], 2], [256, [1,1], 1]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                 #"fcnet_hiddens": [256,256],
                 "vf_share_layers": 'true'
                 },
            'rollout_fragment_length': 2048,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-4,
            #"lr_schedule": [[0, 0.0005], [5e6, 0.0005], [5e6+1, 0.0003]],#, [2e7, 0.0003], [2e7+1, 0.0001]],
            'log_level': 'INFO',
            'framework': 'tf',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    "policy_01": (None, env.observation_space, env.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                    #f"policy_{p:02d}": (None, env.observation_space, env.action_space, {}) for p in range(1, 12)
                },
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
    }, )
    '''
    tune.run(
        "PPO",
        config={
            'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.995,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'entropy_coeff': 0.001,
            'train_batch_size': 16384,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 5,
            'num_workers': 0,
            #'num_cpus_per_worker': 1,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 16,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                #"dim": 15, "conv_filters": [[48, [5, 5], 2], [256, [3, 3], 2], [256, [3, 3], 2]],
                "dim": 15, "conv_filters": [[16, [5, 5], 2],  [32, [3, 3], 1], [32, [3, 3], 1], [32, [3, 3], 2],  [32, [3, 3], 2],  [32, [3, 3], 2], [64, [1, 1], 1]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256,256],
                "post_fcnet_activation": "relu",
                     # "fcnet_hiddens": [256,256],
                # "vf_share_layers": 'true'
                 },
            'rollout_fragment_length': 1024,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-4,
            'log_level': 'INFO',
            'framework': 'tf',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    #f"policy_{p:02d}": (None, env.observation_space, env.action_space, {}) for p in range(1, 7)
                    "policy_01": (None, env.observation_space, env.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                },
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn': policy_mapping_fn,
            },
        },
        #resources_per_trial={'gpu': 1},
    )
    '''
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
