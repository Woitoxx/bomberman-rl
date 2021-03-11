from typing import List, Dict

import gym
from gym.spaces import Discrete, Box, Tuple
from ray.rllib import SampleBatch
from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_filter_config, get_activation_fn
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import one_hot, try_import_tf, override
import numpy as np
from ray.rllib.utils.typing import ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()

class ComplexInputNetwork(TFModelV2):
    """TFModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).
    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.
    The data flow is as follows:
    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and vaulue heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        # TODO: (sven) Support Dicts as well.

        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space
        assert isinstance(self.original_space, (Tuple)), \
            "`obs_space.original_space` must be Tuple!"

        super().__init__(self.original_space, action_space, num_outputs,
                         model_config, name)
        self.new_obs_space = obs_space
        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.one_hot = {}
        self.flatten = {}
        concat_size = 0
        for i, component in enumerate(self.original_space):
            # Image space.
            if len(component.shape) == 3:
                config = {
                    "conv_filters": model_config.get(
                        "conv_filters"),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],

                }
                cnn = CustomVisionNetwork(
                    component,
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    name="cnn_{}".format(i))
                cnn.base_model.summary()
                concat_size += cnn.num_outputs
                self.cnns[i] = cnn
            # Discrete inputs -> One-hot encode.
            elif isinstance(component, Discrete):
                self.one_hot[i] = True
                concat_size += component.n
            # TODO: (sven) Multidiscrete (see e.g. our auto-LSTM wrappers).
            # Everything else (1D Box).
            else:
                self.flatten[i] = int(np.product(component.shape))
                concat_size += self.flatten[i]

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation",
                                                 "relu"),
            "vf_share_layers" : 'False'
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"),
                float("inf"),
                shape=(concat_size,),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="tf",
            name="post_fc_stack")

        self.post_fc_stack_vf = ModelCatalog.get_model_v2(
            Box(float("-inf"),
                float("inf"),
                shape=(concat_size,),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="tf",
            name="post_fc_stack_vf")
        self.post_fc_stack.base_model.summary()
        self.post_fc_stack_vf.base_model.summary()

        # Actions and value heads.
        self.logits_and_value_model = None
        self._value_out = None
        if num_outputs:
            # Action-distribution head.
            p_layer = tf.keras.layers.Input(
                (self.post_fc_stack.num_outputs,))
            v_layer = tf.keras.layers.Input(
                (self.post_fc_stack.num_outputs,))
            logits_layer = tf.keras.layers.Dense(
                num_outputs,
                activation=tf.keras.activations.linear,
                name="logits")(p_layer)

            # Create the value branch model.
            value_layer = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=tf.keras.activations.tanh,
                kernel_initializer=normc_initializer(0.01))(v_layer)
            self.logits_model = tf.keras.models.Model(
                p_layer, [logits_layer])
            self.value_model = tf.keras.models.Model(
                v_layer, [value_layer]
            )
            self.logits_model.summary()
            self.value_model.summary()
        else:
            self.num_outputs = self.post_fc_stack.num_outputs

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS],
                                                   self.new_obs_space, "tf")
        # Push image observations through our CNNs.
        outs = []
        for i, component in enumerate(orig_obs):
            if i in self.cnns:
                cnn_out, _ = self.cnns[i]({SampleBatch.OBS: component})
                outs.append(cnn_out)
            elif i in self.one_hot:
                if component.dtype in [tf.int32, tf.int64, tf.uint8]:
                    outs.append(
                        one_hot(component, self.original_space.spaces[i]))
                else:
                    outs.append(component)
            else:
                outs.append(tf.reshape(component, [-1, self.flatten[i]]))
        # Concat all outputs and the non-image inputs.
        out = tf.concat(outs, axis=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out_p, _ = self.post_fc_stack({SampleBatch.OBS: out}, [], None)
        out_vf, _ = self.post_fc_stack_vf({SampleBatch.OBS: out}, [], None)

        # No logits/value branches.
        #if not self.logits_and_value_model:
        #    return out, []

        # Logits- and value branches.
        logits, values = self.logits_model(out_p), self.value_model(out_vf)
        self._value_out = tf.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out



class CustomVisionNetwork(TFModelV2):
    """Generic vision network implemented in ModelV2 API.

    An additional post-conv fully connected stack can be added and configured
    via the config keys:
    `post_fcnet_hiddens`: Dense layer sizes after the Conv2D stack.
    `post_fcnet_activation`: Activation function to use for this FC stack.

    Examples:


    """

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        super(CustomVisionNetwork, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        activation = get_activation_fn(
            self.model_config.get("conv_activation"), framework="tf")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0,\
            "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="tf")

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")
        self.traj_view_framestacking = False

        # Perform Atari framestacking via traj. view API.
        if model_config.get("num_framestacks") != "auto" and \
                model_config.get("num_framestacks", 0) > 1:
            input_shape = obs_space.shape + (model_config["num_framestacks"], )
            self.data_format = "channels_first"
            self.traj_view_framestacking = True
        else:
            input_shape = obs_space.shape
            self.data_format = "channels_last"

        inputs = tf.keras.layers.Input(shape=input_shape, name="observations")
        last_layer = inputs
        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False

        # Build the action layers
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                data_format="channels_last",
                name="conv{}".format(i))(last_layer)

        out_size, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            last_layer = tf.keras.layers.Conv2D(
                out_size if post_fcnet_hiddens else num_outputs,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv_out")(last_layer)
            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + ([num_outputs]
                                                     if post_fcnet_hiddens else
                                                     [])
            for i, out_size in enumerate(layer_sizes):
                last_layer = tf.keras.layers.Dense(
                    out_size,
                    name="post_fcnet_{}".format(i),
                    activation=post_fcnet_activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv{}".format(len(filters)))(last_layer)

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                if post_fcnet_hiddens:
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        post_fcnet_hiddens[0], [1, 1],
                        activation=post_fcnet_activation,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out")(last_layer)
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens[1:] +
                                                 [num_outputs]):
                        last_layer = tf.keras.layers.Dense(
                            out_size,
                            name="post_fcnet_{}".format(i + 1),
                            activation=post_fcnet_activation
                            if i < len(post_fcnet_hiddens) - 1 else None,
                            kernel_initializer=normc_initializer(1.0))(
                                last_layer)
                else:
                    last_cnn = last_layer = tf.keras.layers.Conv2D(
                        num_outputs, [1, 1],
                        activation=None,
                        padding="same",
                        data_format="channels_last",
                        name="conv_out")(last_layer)

                if last_cnn.shape[1] != 1 or last_cnn.shape[2] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, 1, "
                        "1, {} (`num_outputs`)] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the dims 1 and 2 "
                        "are both 1.".format(self.model_config["conv_filters"],
                                             self.num_outputs,
                                             list(last_cnn.shape)))

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                last_layer = tf.keras.layers.Flatten(
                    data_format="channels_last")(last_layer)

                # Add (optional) post-fc-stack after last Conv2D layer.
                for i, out_size in enumerate(post_fcnet_hiddens):
                    last_layer = tf.keras.layers.Dense(
                        out_size,
                        name="post_fcnet_{}".format(i),
                        activation=post_fcnet_activation,
                        kernel_initializer=normc_initializer(1.0))(last_layer)
                self.num_outputs = last_layer.shape[1]
        logits_out = last_layer


        self.base_model = tf.keras.Model(inputs, [logits_out])

        # Optional: framestacking obs/new_obs for Atari.
        if self.traj_view_framestacking:
            from_ = model_config["num_framestacks"] - 1
            self.view_requirements[SampleBatch.OBS].shift = \
                "-{}:0".format(from_)
            self.view_requirements[SampleBatch.OBS].shift_from = -from_
            self.view_requirements[SampleBatch.OBS].shift_to = 0
            self.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(
                data_col=SampleBatch.OBS,
                shift="-{}:1".format(from_ - 1),
                space=self.view_requirements[SampleBatch.OBS].space,
                used_for_compute_actions=False,
            )

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]
        if self.data_format == "channels_first":
            obs = tf.transpose(obs, [0, 2, 3, 1])
        # Explicit cast to float32 needed in eager.
        model_out = self.base_model(tf.cast(obs, tf.float32))
        # Our last layer is already flat.
        if self.last_layer_is_flattened:
            return model_out, state
        # Last layer is a n x [1,1] Conv2D -> Flatten.
        else:
            return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
