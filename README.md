# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

#### Our entire training code is located in the folder 'training'

- 'training' itself contains our earliest attempts
- 'train_with_action_masking' and 'train_with_action_masking_2' contain subsequent attempts with <strong>action masking</strong>
- 'train_with_action_masking_3' contains attempts using a <strong>deep ResNet-like architecture</strong> + action masking
- 'hierarchical_learning' contains our attempts to implement <strong>hierarchical reinforcement learning</strong>
- 'train_with_action_masking_dqn' contains our attempts with <strong>Deep Q-Learning</strong> instead of PPO
- <strong>'train_with_action_masking_final' contains the code the agent we submitted has been trained with</strong>

Our implementation uses custom torch/tf models with RLlib. Starting code has been taken from https://docs.ray.io/en/master/rllib-models.html?highlight=custom%20model 