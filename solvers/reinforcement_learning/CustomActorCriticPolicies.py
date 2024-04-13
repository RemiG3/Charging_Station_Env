import gymnasium as gym
import warnings
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.utils import get_device
from itertools import zip_longest

import numpy as np
from gymnasium import spaces
import torch as th


class MlpNetwork(nn.Module):
    """
    Constructs a MLP that receives the output from a previous features extractor or directly the observations (if no features extractor is applied) as an input and outputs a latent representation for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[Dict[str, List[int]], List[Union[int, Dict[str, List[int]]]]],
        activation_fn: Union[List[Type[nn.Module]], Type[nn.Module]],
        nn_batchnorm: Union[List[bool], bool],
        nn_dropout: Union[List[float], float],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        
        device = get_device(device)
        shared_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        policy_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim
        
        idx = 0
        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            policy_only_layers = net_arch["pi"]
            value_only_layers = net_arch["vf"]
        else:
            # Iterate through the shared layers and build the shared parts of the network
            for layer in net_arch:
                if isinstance(layer, int):  # Check that this is a shared layer
                    shared_net.append(nn.Linear(last_layer_dim_shared, layer))
                    if ((isinstance(nn_batchnorm, list) and nn_batchnorm[idx]) or ((not isinstance(nn_batchnorm, list)) and nn_batchnorm)):
                        shared_net.append( nn.BatchNorm1d(num_features=layer) )
                    if ((isinstance(nn_dropout, list) and (nn_dropout[idx] > 0.)) or ((not isinstance(nn_dropout, list)) and (nn_dropout > 0.))):
                        shared_net.append( nn.Dropout(p=nn_dropout[idx] if isinstance(nn_dropout, list) else nn_dropout) )
                    if (not isinstance(activation_fn, list) and (activation_fn is not None)) or (isinstance(activation_fn, list) and (activation_fn[idx] is not None)):
                        shared_net.append(activation_fn[idx]() if isinstance(activation_fn, list) else activation_fn())
                    last_layer_dim_shared = layer
                    idx += 1
                else:
                    assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                    if "pi" in layer:
                        assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                        policy_only_layers = layer["pi"]

                    if "vf" in layer:
                        assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                        value_only_layers = layer["vf"]
                    break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for i, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                if ((isinstance(nn_batchnorm, list) and nn_batchnorm[idx]) or ((not isinstance(nn_batchnorm, list)) and nn_batchnorm)):
                    policy_net.append( nn.BatchNorm1d(num_features=pi_layer_size) )
                if ((isinstance(nn_dropout, list) and (nn_dropout[idx] > 0.)) or ((not isinstance(nn_dropout, list)) and (nn_dropout > 0.))):
                    policy_net.append( nn.Dropout(p=nn_dropout[idx] if isinstance(nn_dropout, list) else nn_dropout) )
                if (not isinstance(activation_fn, list) and (activation_fn is not None)) or (isinstance(activation_fn, list) and (activation_fn[idx] is not None)):
                    policy_net.append(activation_fn[idx]() if isinstance(activation_fn, list) else activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                if ((isinstance(nn_batchnorm, list) and nn_batchnorm[idx]) or ((not isinstance(nn_batchnorm, list)) and nn_batchnorm)):
                    value_net.append( nn.BatchNorm1d(num_features=vf_layer_size) )
                if ((isinstance(nn_dropout, list) and (nn_dropout[idx] > 0.)) or ((not isinstance(nn_dropout, list)) and (nn_dropout > 0.))):
                    value_net.append( nn.Dropout(p=nn_dropout[idx] if isinstance(nn_dropout, list) else nn_dropout) )
                if (not isinstance(activation_fn, list) and (activation_fn is not None)) or (isinstance(activation_fn, list) and (activation_fn[idx] is not None)):
                    value_net.append(activation_fn[idx]() if isinstance(activation_fn, list) else activation_fn())
                last_layer_dim_vf = vf_layer_size
            
            idx += 1

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        nn_batchnorm: Union[List[bool], bool] = False,
        nn_dropout: Union[List[float], float] = 0.,
        *args,
        **kwargs,
    ):
        self.nn_batchnorm = nn_batchnorm
        self.nn_dropout = nn_dropout
        
        warnings.filterwarnings(action='ignore', category=UserWarning, module='stable_baselines3')
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpNetwork(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn,
                                        nn_dropout=self.nn_dropout, nn_batchnorm=self.nn_batchnorm, device=self.device)



class CustomMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        nn_batchnorm: Union[List[bool], bool] = False,
        nn_dropout: Union[List[float], float] = 0.,
        *args,
        **kwargs,
    ):
        self.nn_batchnorm = nn_batchnorm
        self.nn_dropout = nn_dropout
        
        warnings.filterwarnings(action='ignore', category=UserWarning, module='sb3_contrib')
        super(CustomMaskableActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        # Disable orthogonal initialization
        self.ortho_init = False
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpNetwork(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn,
                                        nn_dropout=self.nn_dropout, nn_batchnorm=self.nn_batchnorm, device=self.device)




