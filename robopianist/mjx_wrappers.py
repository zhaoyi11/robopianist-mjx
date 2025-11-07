import abc
from typing import Any, Dict, List, Optional, Sequence, Union
import jax
from jax import numpy as jp
import dm_env
from dm_env import specs

from robopianist_mjx.mjx_env import Env, EnvState


class Wrapper(Env):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Env):
    self.env = env

  def reset(self, rng: jax.Array) -> EnvState:
    return self.env.reset(rng)

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> int:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Env:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)


class VmapWrapper(Wrapper):
  """Vectorizes MJX env."""

  def __init__(self, env: Env, num_envs: Optional[int] = None):
    super().__init__(env)
    self.num_envs = num_envs

  def reset(self, rng: jax.Array) -> EnvState:
    if self.num_envs is not None:
      rng = jax.random.split(rng, self.num_envs)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end.
    It also applies action repeat.
  """

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> EnvState:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    return state

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward
    
    # apply action repeat
    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat

    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - done, zero
    )
    state.info['steps'] = steps
    discount = jp.where(steps >= episode_length, one, 1.-state.done)
    state.info['discount'] = discount
    return state.replace(done=done)


class ObservationActionRewardWrapper(Wrapper):
  """Wrapper that puts the previous action and reward into the observation."""
  def __init__(self, env: Env):
    super().__init__(env)
    assert not hasattr(self, 'num_envs'), "VmapWrapper should be applied after this wrapper."

  def reset(self, rng: jax.Array) -> EnvState:
    state = self.env.reset(rng)
    obs = jp.concatenate([state.obs, 
                          jp.zeros((self.env.action_size)),
                          jp.ones((1))], axis=-1)
    return state.replace(obs=obs)

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    state = self.env.step(state, action)
    obs = jp.concatenate([state.obs, action, state.reward], axis=-1)
    return state.replace(obs=obs)

  @property
  def observation_size(self) -> int:
    return self.env.observation_size + self.env.action_size + 1


class CanonicalSpecWrapper(Wrapper):
  """Wrapper which converts environments to use canonical action specs.

    We refer to a canonical action spec as the bounding
    box [-1, 1]^d where d is the dimensionality of the spec. So the shape and
    dtype of the spec is unchanged, while the maximum/minimum values are set
    to +/- 1.
  """
  def __init__(self, env: Env, clip:bool=False):
    super().__init__(env)
    self.clip = clip
    ctrllimited = self.env.sys.actuator_ctrllimited.astype(bool)
    ctrlrange = self.env.sys.actuator_ctrlrange
    num_actouators = ctrlrange.shape[0]
    minima = jp.full(num_actouators, fill_value=-5.)
    maxima = jp.full(num_actouators, fill_value=5.)

    self.ctrl_min = minima.at[ctrllimited].set(ctrlrange[ctrllimited, 0].T)
    self.ctrl_max = maxima.at[ctrllimited].set(ctrlrange[ctrllimited, 1].T)
    
    self.scale = self.ctrl_max - self.ctrl_min
    self.offset = self.ctrl_min

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    """ The input range is [-1, 1] and the output range is [ctrl_min, ctrl_max]"""
    if self.clip:
      action = jp.clip(action, -1.0, 1.0)

    # map action to [0, 1]
    action = 0.5 * (action + 1.0)
    # map action to [ctrl_min, ctrl_max]
    action = action * self.scale + self.offset
    return self.env.step(state, action)


class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> EnvState:
    state = self.env.reset(rng)
    state.info['first_mjx_state'] = state.mjx_state
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: State, action: jax.Array) -> EnvState:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    mjx_state = jax.tree_map(
        where_done, state.info['first_mjx_state'], state.mjx_state
    )
    obs = where_done(state.info['first_obs'], state.obs)
    return state.replace(mjx_state=pipeline_state, obs=obs)