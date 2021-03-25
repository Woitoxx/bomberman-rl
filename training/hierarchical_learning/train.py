import os
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from training.hierarchical_learning.hierarchical_bomberman_multi_env import *
from ray import tune
from training.hierarchical_learning.callbacks import MyCallbacks
from training.hierarchical_learning.tfnet_lstm import ComplexLstmNetwork
from training.train_with_action_masking_2.tfnet_with_masking import ComplexInputNetwork


if __name__ == '__main__':
    ray.init(
        _redis_max_memory=1024 * 1024 * 100,num_gpus=1, object_store_memory=10*2**30)
    env = HierarchicalBombermanMultiEnv([f'agent_{i}_high' for i in range(4)])

    ModelCatalog.register_custom_model("custom_model", ComplexInputNetwork)
    ModelCatalog.register_custom_model("custom_lstm_model", ComplexLstmNetwork)
    tune.register_env('BomberMan-v0', lambda c: HierarchicalBombermanMultiEnv([f'agent_{i}_high' for i in range(4)]))


    def policy_mapping_fn(agent_id):
        if agent_id.startswith("agent_0_high"):# or np.random.rand() > 0.2:
            return "policy_01"  # Choose 01 policy for agent_01
        #if agent_id.startswith("agent_0_low_COIN_"):
        #    return "policy_01_coin"
        #if agent_id.startswith("agent_0_low_DESTROY_"):
        #    return "policy_01_destroy"
        #if agent_id.startswith("agent_0_low_KILL_"):
        #    return "policy_01_kill"
        if agent_id[8:].startswith("high"):
            return "policy_02"
        if agent_id.startswith("low_COLLECT_"):
            return "policy_collect"
        if agent_id.startswith("low_DESTROY_"):
            return "policy_destroy"
        if agent_id.startswith("low_KILL_"):
            return "policy_kill"

    def train(config, checkpoint_dir=None):
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        init_w = trainer.get_policy('policy_01').get_weights()
        trainer.restore('C:\\Users\\Florian\\ray_results\\PPO_BomberMan-v0_2021-03-25_08-56-43eo23nmho\\checkpoint_002360\\checkpoint-2360')
        trainer.workers.foreach_worker(lambda w: w.get_policy('policy_01').set_weights(init_w))
        trainer.restore('.\\kill-policy-0\\checkpoint')
        trainer.import_model()
        iter = 0

        #def update_phase(ev):
        #    ev.foreach_env(lambda e: e.set_phase(phase))

        while True:
            iter += 1
            result = trainer.train()
            if iter % 200 == 0:
                if not os.path.exists(f'./model-{iter}'):
                    #trainer.get_policy('policy_01').export_model(f'./model-{iter}')
                    trainer.export_policy_model(f'./model-{iter}/main', 'policy_01')
                    trainer.export_policy_model(f'./model-{iter}/collect', 'policy_collect')
                    trainer.export_policy_model(f'./model-{iter}/destroy', 'policy_destroy')
                    trainer.export_policy_model(f'./model-{iter}/kill', 'policy_kill')

                else:
                    print("model already saved")

    train(config={
        'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.98,
            'kl_coeff': 0.2,
            'vf_loss_coeff' : 0.5,
            'clip_rewards': False,
            'entropy_coeff': 0.0001,
            'train_batch_size': 16384,#32768,#49152,
            'sgd_minibatch_size': 64,
            'shuffle_sequences': True,
            'num_sgd_iter': 15,
            'num_workers': 2,
            'num_cpus_per_worker': 3,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 4,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 15,
                "conv_filters": [[32, [7, 7], 2], [64, [3,3], 2], [128, [3,3], 2], [128, [1,1], 1]],
                #"conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [2, [1, 1], 1]],
                #8k run "conv_filters" : [[32, [5,5], 2], [32, [3,3], 2], [64, [3,3], 2], [128, [3,3], 2], [256, [1,1], 1]],
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
                    "policy_01": (None, env.observation_space, env.action_space, {
                        'entropy_coeff': 0.01, 'gamma': 0.999, 'lr': 1e-4, 'num_sgd_iter': 5,
                        #"model": {
                        #    'use_lstm': True,
                        #    "lstm_cell_size": 64,
                        #    "custom_model": "custom_lstm_model",
                        #    "dim": 15,
                        #    "conv_filters": [[32, [7, 7], 2], [64, [3, 3], 2], [128, [3, 3], 2], [128, [1, 1], 1]],
                        #    "conv_activation": "relu",
                        #    "post_fcnet_hiddens": [256],
                        #    "post_fcnet_activation": "relu",
                        #    "vf_share_layers": 'true'
                        #},
                    }),
                    #"policy_01_coin": (None, COIN_OBSERVATION_SPACE, env.action_space, {}),
                    #"policy_01_destroy": (None, DESTROY_OBSERVATION_SPACE, env.action_space, {}),
                    #"policy_01_kill": (None, KILL_OBSERVATION_SPACE, env.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                    "policy_collect": (None, COLLECT_OBSERVATION_SPACE, env.flat_env.action_space, {}),
                    "policy_destroy": (None, DESTROY_OBSERVATION_SPACE, env.flat_env.action_space, {}),
                    "policy_kill": (None, KILL_OBSERVATION_SPACE, env.flat_env.action_space, {}),
                },
                "policies_to_train": ["policy_01","policy_collect","policy_destroy","policy_kill"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
    }, )