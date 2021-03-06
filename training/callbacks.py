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

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        #print("episode {} started".format(episode.episode_id))
        #episode.user_data["pole_angles"] = []
        #episode.hist_data["pole_angles"] = []
        pass

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        #pole_angle = abs(episode.last_observation_for()[2])
        #raw_angle = abs(episode.last_raw_obs_for()[2])
        #assert pole_angle == raw_angle
        #episode.user_data["pole_angles"].append(pole_angle)
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        #pole_angle = np.mean(episode.user_data["pole_angles"])
        #print("episode {} ended with length {} and pole angles {}".format(
        #    episode.episode_id, episode.length, pole_angle))
        #winner, loser = worker.env.get_winner_loser()
        #for k in episode.agent_rewards:
        #    if k[0] in winner:
        #        episode.agent_rewards[k] = 1.
        #        episode.total_reward += 1
        #        episode._agent_reward_history[k[0]][-1] = 1
        #    elif k[0] in loser:
        #        episode.agent_rewards[k] = -1 / 3.
        #        episode.total_reward -= 1/3.
        #        episode._agent_reward_history[k[0]][-1] = -1/3.
        pass

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        #print("returned sample batch of size {}".format(samples.count))
        pass

    @staticmethod
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
        # you can mutate the result dict to add new fields to return
        #result["callback_ok"] = True
        current_policy = trainer.get_policy('policy_01').get_weights()
        #current_policy = {}
        #for k,v in trainer.get_policy('policy_01').get_weights().items():
        #    current_policy[k] = v

        self.policies.append(current_policy)
        if len(self.policies) > 100:
            self.policies.pop(0)
        #self.copy_weights(current_policy if np.random.rand() > 0.2 else np.random.choice(self.policies), trainer.get_policy('policy_02'))
        new_policy = current_policy if np.random.rand() > 0.2 else random.choice(self.policies)
        trainer.workers.foreach_worker(lambda w: w.get_policy('policy_02').set_weights(new_policy))
        #trainer.workers.foreach_worker(lambda w: self.copy_weights(current_policy if np.random.rand() > 0.2 else np.random.choice(self.policies), w.get_policy('policy_02')))
        if result["iterations_since_restore"] % 10 == 0:
            trainer.save()


    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        #winner, loser = worker.env.get_winner_loser()
        #if agent_id in winner:
        #    postprocessed_batch.data['rewards'][-1] = 1.
        #elif agent_id in loser:
        #    postprocessed_batch.data['rewards'][-1] = -1/3.
        pass
