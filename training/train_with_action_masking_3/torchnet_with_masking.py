from gym.spaces import Box, Discrete, Tuple
import numpy as np

# TODO (sven): add IMPALA-style option.
# from ray.rllib.examples.models.impala_vision_nets import TorchImpalaVisionNet
from ray.rllib.models.torch.misc import normc_initializer as \
    torch_normc_initializer, SlimFC, SlimConv2d
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import one_hot

torch, nn = try_import_torch()


class ComplexTorchInputNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        # TODO: (sven) Support Dicts as well.
        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space
        assert isinstance(self.original_space, (Tuple)), \
            "`obs_space.original_space` must be Tuple!"

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, self.original_space, action_space,
                              num_outputs, model_config, name)
        self.new_obs_space = obs_space
        # Atari type CNNs or IMPALA type CNNs (with residual layers)?
        # self.cnn_type = self.model_config["custom_model_config"].get(
        #     "conv_type", "atari")

        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.one_hot = {}
        self.flatten = {}
        concat_size_p, concat_size_v = 0, 0
        for i, component in enumerate(self.original_space[:-1]):
            # Image space.
            if len(component.shape) == 3:
                config = {
                    "conv_filters": model_config["conv_filters"]
                    if "conv_filters" in model_config else
                    get_filter_config(obs_space.shape),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                }
                # if self.cnn_type == "atari":
                cnn = TorchBatchNormModel(component, action_space, None, config, 'cnn_{}'.format(i))
                print(cnn)
                concat_size_p += cnn.num_outputs_p
                concat_size_v += cnn.num_outputs_v
                self.cnns[i] = cnn
                self.add_module("cnn_{}".format(i), cnn)
            # Discrete inputs -> One-hot encode.
            elif isinstance(component, Discrete):
                self.one_hot[i] = True
                concat_size_p += component.n
                concat_size_v += component.n
            # Everything else (1D Box).
            else:
                self.flatten[i] = int(np.product(component.shape))
                concat_size_p += self.flatten[i]
                concat_size_v += self.flatten[i]

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation",
                                                 "relu"),
            "vf_share_layers": 'True'
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"),
                float("inf"),
                shape=(concat_size_p, ),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack")

        self.post_fc_stack_vf = ModelCatalog.get_model_v2(
            Box(float("-inf"),
                float("inf"),
                shape=(concat_size_v,),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack")

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        if num_outputs:
            # Action-distribution head.
            self.logits_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=num_outputs,
                activation_fn=None,
            )
            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=self.post_fc_stack_vf.num_outputs,
                out_size=1,
                activation_fn='tanh',
                initializer=torch_normc_initializer(0.01))
        else:
            raise NotImplementedError()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Push image observations through our CNNs.
        orig_obs = input_dict["obs"]
        outs = []
        v_outs = []
        for i, component in enumerate(orig_obs[:-1]):
            if i in self.cnns:
                cnn_out, _ = self.cnns[i]({"obs": component})
                outs.append(cnn_out)
                v_outs.append(self.cnns[i].value_function())
            elif i in self.one_hot:
                if component.dtype in [torch.int32, torch.int64, torch.uint8]:
                    outs.append(
                        one_hot(component, self.original_space.spaces[i]))
                    v_outs.append(
                        one_hot(component, self.original_space.spaces[i]))
                else:
                    outs.append(component)
                    v_outs.append(component)
            else:
                outs.append(torch.reshape(component, [-1, self.flatten[i]]))
                v_outs.append(torch.reshape(component, [-1, self.flatten[i]]))
        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        v_out = torch.cat(v_outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out_p, _ = self.post_fc_stack({"obs": out}, [], None)
        out_v, _ = self.post_fc_stack_vf({"obs": v_out}, [], None)

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        inf_mask = torch.max(torch.log(orig_obs[-1]), torch.float32.min)
        self._value_out = torch.reshape(values, [-1])
        return logits + inf_mask, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out


class TorchBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""
    capture_index = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        layers = []
        (w, h, in_channels) = obs_space.shape
        self._logits = None
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, \
            "Must provide at least 1 entry in `conv_filters`!"
        self.data_format = "channels_last"
        for i, (out_channels, kernel, stride) in enumerate(filters[:-1], 1):
            if i == 1:
                layers.append(
                    nn.Sequential(
                        SlimConv2d(in_channels, out_channels, kernel, stride, padding=1, activation_fn=None),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
                )
                in_channels = out_channels
            else:
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

        out_channels, kernel, stride = filters[-1]

        p_layer = nn.Sequential(
            SlimConv2d(in_channels, out_channels, kernel, stride, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        v_layer = nn.Sequential(
            SlimConv2d(in_channels, 1, kernel, stride, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self._logits = nn.Sequential(p_layer, nn.Flatten())
        self._value_branch = nn.Sequential(v_layer, nn.Flatten())

        self._convs = nn.Sequential(*layers)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._convs.train(mode=input_dict.get("is_training", False))
        self._logits.train(mode=input_dict.get("is_training", False))
        self._value_branch.train(mode=input_dict.get("is_training", False))
        self._conv_out = self._convs(input_dict["obs"])
        logits = self._logits(self._conv_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._conv_out), [-1])


def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(SlimConv2d(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, activation_fn=None, stride=1, padding=1),
            nn.ReLU(),
            conv_bn(self.out_channels, self.out_channels, activation_fn=None, stride=1, padding=1),
        )
        self.activate = nn.ReLU(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
