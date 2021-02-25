from typing import Dict
import numpy as np
from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode


class MyCallbacks(DefaultCallbacks):
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
        pass

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        #print("returned sample batch of size {}".format(samples.count))
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        #result["callback_ok"] = True
        pass

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        winner, loser = worker.env.get_winner_loser()
        if agent_id in winner:
            postprocessed_batch.data['rewards'][-1] = 1.
        elif agent_id in loser:
            postprocessed_batch.data['rewards'][-1] = -1/3.
