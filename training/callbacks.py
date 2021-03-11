import copy
import random
from typing import Dict
import numpy as np
from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode


class MyCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.policies = []
        self.player_scores = []
        self.opponent_scores = []

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        self.player_scores.append(episode.last_info_for(f'agent_0'))
        for i in range(1,4):
            self.opponent_scores.append(episode.last_info_for(f'agent_{i}'))

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print(f'Player max score: {np.max(self.player_scores)}')
        print(f'Player avg score: {np.average(self.player_scores)}')
        print(f'Opp max score: {np.max(self.opponent_scores)}')
        print(f'Opp avg score: {np.average(self.opponent_scores)}')
        self.player_scores.clear()
        self.opponent_scores.clear()
        pass

    @staticmethod
    # probably no longer required
    def copy_weights(src_policy, dest_policy):
        P0key_P1val = {}  # temp storage with "policy_0" keys & "policy_1" values
        for (k, v), (k2, v2) in zip(dest_policy.get_weights().items(),
                                    src_policy.items()):
            P0key_P1val[k] = v2

        # set weights
        dest_policy.set_weights(P0key_P1val)

    def on_train_result(self, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))

        # Add current policy to the menagerie

        current_policy = trainer.get_policy('policy_01').get_weights()
        if result["policy_reward_mean"]["policy_01"] > 0 or len(self.policies) == 0:
            self.policies.append(current_policy)
        # Maintain only the latest 100 previous policies
            if len(self.policies) > 100:
                self.policies.pop(0)
        #self.copy_weights(current_policy if np.random.rand() > 0.2 else np.random.choice(self.policies), trainer.get_policy('policy_02'))

        # Choose either current policy (80%) or random previous policy (20%) for our opponents
        new_policy = current_policy if np.random.rand() > 0.2 else random.choice(self.policies)

        trainer.workers.foreach_worker(lambda w: w.get_policy('policy_02').set_weights(new_policy))
        #trainer.workers.foreach_worker(lambda w: self.copy_weights(current_policy if np.random.rand() > 0.2 else np.random.choice(self.policies), w.get_policy('policy_02')))

        # Checkpoint
        if result["iterations_since_restore"] % 10 == 0:
            print(f'Checkpoint saved at iter {result["iterations_since_restore"]}')
            trainer.save()


    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
