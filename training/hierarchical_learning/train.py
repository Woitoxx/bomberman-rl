import copy
import random

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.util.client import ray

from training.bomberman_multi_env import BombermanEnv
from training.hierarchical_learning.hierarchical_bomberman_multi_env import HierarchicalBombermanMultiEnv
from training.tfnet import ComplexInputNetwork

if __name__ == '__main__':
    phase = 0
    ray.init(object_store_memory=6000000000)
    env, henv = BombermanEnv([f'agent_{i}' for i in range(4)]), HierarchicalBombermanMultiEnv([f'agent_{i}' for i in range(4)]),
    ModelCatalog.register_custom_model("custom_model", ComplexInputNetwork)
    tune.register_env('BomberMan-v0', lambda c: HierarchicalBombermanMultiEnv([f'agent_{i}' for i in range(4)]))

    def policy_mapping_fn(agent_id):
        #if phase == 0:
        #    return "policy_01"
        #else:
        if agent_id.startswith("agent_"):
            return "policy_01"  # Choose 01 policy for agent_01
        if agent_id.startswith("COLLECT_"):
            return "policy_02"
        if agent_id.startswith("DESTROY_"):
            return "policy_03"
        if agent_id.startswith("KILL_"):
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
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        #trainer.restore('C:\\Users\\Florian\\ray_results\\PPO_BomberMan-v0_2021-02-28_15-34-57we64cvop\\checkpoint_2430\\checkpoint-2430')
        iter = 1

        def update_policies(policy, policyID):
            if policyID != "policy_01":
                new_policy = current_policy if random.random() > 0.2 else random.choice(model_pool)
                copy_weights(new_policy, policy)

        while True:
            result = trainer.train()
            if iter % 10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
            iter += 1

    train(config={
        'env': 'BomberMan-v0',
            "use_critic": True,
            #'callbacks': MyCallbacks,
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
            'num_workers': 0,
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
                    "policy_01": (None, env.observation_space, henv.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                    "policy_03": (None, env.observation_space, env.action_space, {}),
                    "policy_04": (None, env.observation_space, env.action_space, {})
                },
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
    }, )