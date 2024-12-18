import gymnasium as gym
import argparse
import time
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env
from utils_rl import CustomMaskableActorCriticPolicy, parse_str_with_None, parse_list_act_fn, parse_list_int_with_None, parse_list_bool, parse_list_float, parse_bool, parse_dic_args
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.patch_gym import _patch_env

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported


class CustomMPPO(MaskablePPO):
    def __init__(self, *args, **kwargs):
        super(CustomMPPO, self).__init__(*args, **kwargs)
        if 'clip_range' in kwargs:
            self.clip_range_value = kwargs['clip_range']
        else:
            self.clip_range_value = 0.2

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            actions, action_masks = self._post_process_actions(env.envs[0], actions, action_masks)
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def predict(self, env, *args, **kwargs):
        actions, state = super().predict(*args, **kwargs)
        actions, _ = self._post_process_actions(env, actions.reshape((1, -1)))
        return actions.reshape((-1,)), state

    def _post_process_actions(self, env, actions, action_masks=None):
        power_actions = actions[:, env.number_of_chargers:]
        power_chargers = np.repeat(np.array([env.schema['Charger_types'][env.schema['Chargers_config']["list_chargers"][charger]]['charging_rate'] for charger in range(env.number_of_chargers)])[np.newaxis, :], actions.shape[0], axis=0)
        power_demands = power_actions * power_chargers
        
        # Get the indices of the cumulative sum that exceed the limit power by beginning the sum from the index 0
        ts = env.timestep // env.schema['step_time']
        power_masks = np.where(np.cumsum(power_demands, axis=1) > env.schema['grid_limit']+env.energy['renewable'][ts], 0, 1)
        actions[:, env.number_of_chargers:] = actions[:, env.number_of_chargers:] * power_masks

        return actions, action_masks
    

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common import type_aliases
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            env.envs[0],
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        
        new_observations, rewards, dones, infos = env.step(actions.reshape((1, -1)))
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param log_path: The path where logs will be saved
    :param n_eval_episodes: Number of episodes to evaluate the agent
    :param eval_freq: Frequency of evaluation
    :param deterministic: Whether to use deterministic actions
    :param render: Whether to render the environment
    :param best_model_save_path: Path to save the best model
    :param log_best_mean_reward: Whether to log the best mean reward
    """
    def __init__(self, eval_env, log_path, n_eval_episodes=10, eval_freq=10000,
                 deterministic=True, render=False, best_model_save_path=None,
                 log_best_mean_reward=True):
        super(EvalCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.best_mean_reward = -np.inf
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.log_best_mean_reward = log_best_mean_reward

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env,
                                                      n_eval_episodes=self.n_eval_episodes,
                                                      deterministic=self.deterministic,
                                                      render=self.render)
            self.logger.record('validation/mean_reward', mean_reward)
            self.logger.record('validation/std_reward', std_reward)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.log_best_mean_reward:
                    self.logger.record('validation/best_mean_reward', self.best_mean_reward)
            self.logger.dump(self.num_timesteps)
        return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ChargingStationEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episodes", default=1_000_000, type=int)
    parser.add_argument("--eval_episodes", default=500, type=int)
    parser.add_argument("--eval_freq", default=20_000, type=int)
    parser.add_argument("--schema", default="../../schema.json")
    parser.add_argument("--initializer", default="Initializer")
    parser.add_argument("--simulation", default="Simulate_Station_FIFO")
    parser.add_argument("--action", default="Simulate_Actions_FIFO")
    parser.add_argument("--energy", default="Energy_Initializer")
    parser.add_argument("--model_path_saved", default=None, type=str, help="The path to the trained model to continue training")
    parser.add_argument("--initializer_args", default="nb_ev_min_range=50,nb_ev_max_range=150", type=parse_dic_args)
    parser.add_argument("--simulation_args", default={}, type=parse_dic_args)
    parser.add_argument("--action_args", default={}, type=parse_dic_args)
    parser.add_argument("--energy_args", default={}, type=parse_dic_args)
    parser.add_argument("--folder_name", default="MPPO", type=str)
    parser.add_argument("--learning_rate", default=0.006, type=float)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--n_steps", default=4096, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--clip_range", default=0.2, type=float)
    parser.add_argument("--gae_lambda", default=0.98, type=float)
    parser.add_argument("--max_grad_norm", default=0.97, type=float)
    parser.add_argument("--ent_coef", default=0.01, type=float)
    parser.add_argument("--vf_coef", default=0.2, type=float)
    parser.add_argument("--normalize_advantage", default=True, type=parse_bool)
    parser.add_argument("--policy_activation", default="tanh", type=parse_list_act_fn)
    parser.add_argument("--policy_net_shared", default="", type=parse_list_int_with_None)
    parser.add_argument("--policy_net_pi", default="512,512", type=parse_list_int_with_None)
    parser.add_argument("--policy_net_vf", default="512,512", type=parse_list_int_with_None)
    parser.add_argument("--policy_nn_dropout", default=0., type=parse_list_float)
    parser.add_argument("--policy_nn_batchnorm", default=False, type=parse_list_bool)
    args = parser.parse_args()

    env = gym.make(args.env,
                   schema=args.schema,
                   initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                   simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                   action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                   )
    env.seed(args.seed)

    num_id = int(time.time())
    policy_activation = args.policy_activation.__name__.lower()
    policy_shared = 'x'.join(map(str, args.policy_net_shared)) if(len(args.policy_net_shared) > 0) else None
    policy_pi = 'x'.join(map(str, args.policy_net_pi)) if(len(args.policy_net_pi) > 0) else None
    policy_vf = 'x'.join(map(str, args.policy_net_vf)) if(len(args.policy_net_vf) > 0) else None
    batchnorm = 'BN' if args.policy_nn_batchnorm else 'noBN'
    dropout = f'dropout=[{",".join(args.policy_nn_dropout)}]' if isinstance(args.policy_nn_dropout, list) else f'dropout={args.policy_nn_dropout}'
    models_dir = f"models/{args.folder_name}-{policy_activation}_{policy_shared}_{policy_pi}_{policy_vf}_{batchnorm}_{dropout}-{env.schema['schema_name']}-{num_id}"
    logdir = f"logs/{args.folder_name}-{policy_activation}_{policy_shared}_{policy_pi}_{policy_vf}_{batchnorm}_{dropout}-{env.schema['schema_name']}-{num_id}"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # It will check your custom environment and output additional warnings if needed
    check_env(env)

    policy_kwargs = dict(activation_fn=args.policy_activation,
                        net_arch=[*args.policy_net_shared, dict(pi=args.policy_net_pi, vf=args.policy_net_vf)],
                        nn_batchnorm=args.policy_nn_batchnorm,
                        nn_dropout=args.policy_nn_dropout)


    eval_env = gym.make(args.env,
                        schema=args.schema,
                        current_folder=None,
                        results_folder=None,
                        initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.initializer)(**args.initializer_args, energy_initializer=getattr(getattr(__import__('charging_station_env'), 'initializer'), args.energy)(**args.energy_args)),
                        simulation_controller=getattr(getattr(__import__('charging_station_env'), 'transition'), args.simulation)(**args.simulation_args),
                        action_controller=getattr(getattr(__import__('charging_station_env'), 'action'), args.action)(**args.action_args),
                        )

    eval_env = _patch_env(eval_env)
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(eval_env, log_path=logdir, n_eval_episodes=args.eval_episodes, eval_freq=20_000, deterministic=True, best_model_save_path=models_dir, log_best_mean_reward=True)

    if args.model_path_saved and os.path.exists(args.model_path_saved):
        model = CustomMPPO.load(args.model_path_saved, policy=CustomMaskableActorCriticPolicy, env=env, verbose=1, tensorboard_log=logdir, learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range, clip_range_vf=None, normalize_advantage=args.normalize_advantage, ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=None, seed=args.seed, device='auto', _init_setup_model=True)
    else:
        model = CustomMPPO(CustomMaskableActorCriticPolicy, env, verbose=1, tensorboard_log=logdir, policy_kwargs=policy_kwargs, learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range, clip_range_vf=None, normalize_advantage=args.normalize_advantage, ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=None, seed=args.seed, device='auto', _init_setup_model=True)

    print(model.policy)

    model.learn(total_timesteps=args.episodes, reset_num_timesteps=False, tb_log_name=args.folder_name, callback=eval_callback)
    model.save(f"{models_dir}/{args.episodes}")

    env.close()
    eval_env.close()



