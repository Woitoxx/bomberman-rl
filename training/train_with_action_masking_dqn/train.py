import os
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from training.train_with_action_masking_dqn.bomberman_multi_env import BombermanEnv
from ray import tune

from typing import List

import events as e

from training.train_with_action_masking_dqn.callbacks import MyCallbacks
from training.train_with_action_masking_dqn.tfnet_with_masking import ComplexInputNetwork

# Custom Events
MOVED_TO_SAFETY = "MOVED_TO_SAFETY"
KILLED_LEADER = "KILLED_LEADER"
IN_DANGER_ZONE = "IN_DANGER_ZONE"
STEP_PENALTY = "STEP_PENALTY"

if __name__ == '__main__':
    ray.init(
        _redis_max_memory=1024 * 1024 * 100, num_gpus=1, object_store_memory=10 * 2 ** 30)
    env = BombermanEnv([f'agent_{i}' for i in range(4)])

    ModelCatalog.register_custom_model("custom_model", ComplexInputNetwork)
    tune.register_env('BomberMan-v0', lambda c: BombermanEnv([f'agent_{i}' for i in range(4)]))


    def policy_mapping_fn(agent_id):
        return "policy_01"

    def train(config, checkpoint_dir=None):
        trainer = DQNTrainer(config=config, env='BomberMan-v0')
        iter = 0

        while True:
            iter += 1
            result = trainer.train()
            if iter % 10 == 0:
                if not os.path.exists(f'./model-{iter}'):
                    trainer.get_policy('policy_01').export_model(f'./model-{iter}')
                else:
                    print("model already saved")


    def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
        # Always added
        events.append(STEP_PENALTY)

        # Only add when opponent is killed by agent and it had the highest score
        max_score = 0
        highest_score_killed = False
        for opponent in old_game_state['others']:
            if max_score < opponent[1]:
                max_score = opponent[1]
        for opponent in new_game_state['others']:
            if max_score> opponent[1]:
                highest_score_killed = True
        if events.__contains__("KILLED_OPPONENT") & highest_score_killed:
            events.append(KILLED_LEADER)

        # Still to add: MOVED_TO_SAFETY and IN_DANGER_ZONE

    # Not currently used, setup for more nuanced event-system
    def reward_from_events(self, events: List[str]) -> int:
        """
        *This is not a required function, but an idea to structure your code.*

        Here you can modify the rewards your agent get so as to en/discourage
        certain behavior.
        """
        game_rewards = {
            # Rewards
            e.MOVED_UP: 0.1,
            e.MOVED_DOWN: 0.1,
            e.MOVED_RIGHT: 0.1,
            e.MOVED_LEFT: 0.1,
            e.WAITED: 0.05,

            e.BOMB_DROPPED: 0.2,
            e.CRATE_DESTROYED: 0.15,
            e.COIN_FOUND: 0.2,
            e.COIN_COLLECTED: 0.3,

            e.KILLED_OPPONENT: 1.0,
            e.SURVIVED_ROUND: 1.0,
            e.OPPONENT_ELIMINATED: 0.2,

            # Penalties
            e.INVALID_ACTION: -0.1,

            e.GOT_KILLED: -0.5,
            e.KILLED_SELF: -0.5,

            #Custom
            KILLED_LEADER: 0.2,
            STEP_PENALTY: -0.01
        }
        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        return reward_sum


    train(config={
        "callbacks" : MyCallbacks,
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": True,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 5,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },

        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": 10000,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 3000,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 50000,
        # The number of contiguous environment steps to replay at once. This may
        # be set to greater than 1 to support recurrent models-dqn.
        "replay_sequence_length": 1,
        # If True prioritized replay buffer will be used.
        # "prioritized_replay": False,
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.4,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,

        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        "before_learn_on_batch": None,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 5e-4,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-4,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 40,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": 4,
        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 32,
        "model": {
            #"custom_model": "custom_model",
            "dim": 15,
            #"conv_filters": [[48, [7, 7], 2], [96, [3, 3], 2], [192, [3, 3], 2], [192, [1, 1], 1]],
            "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [2, [1, 1], 1]],
            # 8k run "conv_filters" : [[32, [5,5], 2], [32, [3,3], 2], [64, [3,3], 2], [128, [3,3], 2], [256, [1,1], 1]],
            "conv_activation": "relu",
            "fcnet_hiddens" : [256,256],
            #"post_fcnet_hiddens": [256],
            #"post_fcnet_activation": "relu",
            # "fcnet_hiddens": [256,256],
            "vf_share_layers": 'true'
        },
        'multiagent': {
            "policies": {
                "policy_01": (None, env.observation_space, env.action_space, {}),
                #"policy_02": (None, env.observation_space, env.action_space, {}),
            },
            "policies_to_train": ["policy_01"],
            'policy_mapping_fn':
                policy_mapping_fn,
        },
        # === Parallelism ===
        # Number of workers for collecting samples with. This only makes sense
        # to increase if your environment is particularly slow to sample, or if
        # you"re using the Async or Ape-X optimizers.
        "num_workers": 0,
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 1,
    })
