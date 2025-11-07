#%%
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".45"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
from typing import Any, Dict, List, Optional, Sequence

from robopianist_mjx.mjx_env import MJXEnv, EnvState, MjxState
from robopianist_mjx import mjcf
from pathlib import Path
import mujoco
import jax
import numpy as np
import robopianist_mjx.sim_math as utils
from jax import numpy as jp
from mujoco import mjx
from robopianist_mjx.linear_sum_assignment import linear_sum_assignment

from robopianist.music import midi_file
from robopianist import music

_HERE = Path(__file__).resolve().parent


################## Constants ##################
# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005
# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.
# Piano threshold for determining whether a key is activated
_KEY_THRESHOLD = 0.00872665  # 0.5 degrees.
# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05
# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3
# Bounds for the uniform distribution from which initial hand offset is sampled.
_POSITION_OFFSET = 0.05

_NUM_PIANO_KEYS = 88
_NUM_FINGERS = 10 

############### Helper Functions ###################
# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1

def _sigmoids(x, value_at_1, sigmoid='gaussian'):
  """Returns 1 when `x` == 0, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
    sigmoid: String, choice of sigmoid type.

  Returns:
    A numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
      `quadratic` sigmoids which allow `value_at_1` == 0.
    ValueError: If `sigmoid` is of an unknown type.
  """
  if not 0 < value_at_1 < 1:
    raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                      'got {}.'.format(value_at_1))
  if sigmoid != 'gaussian':
    raise ValueError('Unknown sigmoid type {!r}, only gaussian is supported, for more sigmoid, please refer dm_control.'.format(sigmoid))
  
  scale = jp.sqrt(-2 * jp.log(value_at_1))
  return jp.exp(-0.5 * (x*scale)**2)

def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded
      at one or both ends, or they can be equal to one another if the target
      value is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds.
      * If `margin == 0` then the output will be 0 for all values of `x`
        outside of `bounds`.
      * If `margin > 0` then the output will decrease sigmoidally with
        increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.

  Returns:
    A float or numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = jp.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = jp.where(in_bounds, 1.0, 0.0)
  else:
    d = jp.where(x < lower, lower - x, x - upper) / margin
    value = jp.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  return value

def get_piano_geom_xpos(geom_xpos, key_idx):
  # site_xpos is concatenated by site_xpos of left_hand fingertips right_hand fingertips and piano keys.
  key_site_xpos = geom_xpos[-_NUM_PIANO_KEYS:] 
  return key_site_xpos[key_idx]

def get_piano_state(qpos, qpos_range):
  # get piano joint state (qpos), normalized by the key joint range, and activation

  # MuJoCo joint limits are soft, so we clip any joint positions that are
  # outside their limits.
  piano_qpos = qpos[-_NUM_PIANO_KEYS:]
  piano_qpos = jp.clip(piano_qpos, *qpos_range.T)
  normalized_qpos = piano_qpos / qpos_range[:, 1] 
  activation = jp.abs(piano_qpos - qpos_range[:, 1]) <= _KEY_THRESHOLD
  return piano_qpos, normalized_qpos, activation

def get_fingertip_xpos(site_xpos):
  fingertips_site_xpos = site_xpos[:-_NUM_PIANO_KEYS] # for both hands
  return fingertips_site_xpos

def get_hands_state(qpos):
  return qpos[:-_NUM_PIANO_KEYS]

def get_finger_force(qfrc_actuator):
  return qfrc_actuator[:-_NUM_PIANO_KEYS]

# %%
class RoboPianist(MJXEnv):

  def __init__(
      self,
      midi_path,
      **kwargs,
  ):
    # Load model
    path = _HERE/ 'shadow_hand' / 'robopianist.xml'
    mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
    mj_model.opt.timestep = _PHYSICS_TIMESTEP
    sys = mjcf.load_model(mj_model)
    self.piano_joint_range = sys.jnt_range[-_NUM_PIANO_KEYS:]
    self.piano_key_geom_size = sys.geom_size[-_NUM_PIANO_KEYS:] # the last _NUM_PIANO_KEYS geoms are piano keys
    physics_steps_per_control_step = utils.compute_n_steps(_CONTROL_TIMESTEP, _PHYSICS_TIMESTEP)
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    super().__init__(sys, **kwargs)

    # process midi to get goal states
    self._midi = music.load(midi_path)
    if kwargs.get('trim_silence', True):
      self._midi = self._midi.trim_silence()
    self._initial_buffer_time = kwargs.get('initial_buffer_time', 0.)
    self._slice_music_length = int(kwargs.get('slice_music_length', 200)) # change to 200
    self._lookahead_steps = int(kwargs.get('lookahead_steps', 10))
    self._slice_music_idx = int(kwargs.get('slice_music_idx', 0))
    self._goal_state, self._sustain_states = self._parse_midi()    

  @property
  def goal_state(self):
    return self._goal_state[:self._slice_music_length] # remove paddings
  
  @property
  def sustain_state(self):
    return self._sustain_states

  ########## Reset ##########
  def _parse_midi(self) -> None:
    note_traj = midi_file.NoteTrajectory.from_midi(
        self._midi, _CONTROL_TIMESTEP
    )
    note_traj.add_initial_buffer_time(self._initial_buffer_time)
    # notes includes # of steps PianoNote pairs. Each step includes a left hand and right hand note.
    # PianoNote includes note number, velocity, key, name and fingering if available.
    self._notes = note_traj.notes
    _sustains = np.array(note_traj.sustains).astype(bool)

    if self._slice_music_length > 0:
      # select a slice of notes, starting from self._slice_music_idx
      slice_len = int(self._slice_music_length)
      slice_start_idx = int(self._slice_music_idx)
      self._notes = self._notes[slice_start_idx:slice_start_idx+slice_len]
      _sustains = _sustains[slice_start_idx:slice_start_idx+slice_len]
    self.notes_length = len(self._notes)
    # retrieve goal from notes
    _goal = np.zeros((len(self._notes), _NUM_PIANO_KEYS), dtype=np.float32)
    for i, note in enumerate(self._notes):
      keys = [_key.key for _key in note]
      _goal[i, keys] = 1.0 
    _goal = np.pad(_goal, ((0, self._lookahead_steps), (0, 0)), 'edge') # padding to support lookahead
    _goal = jp.array(_goal) # put the goal into jax array
    return _goal, _sustains
  #############################

  def reset(self, rng: jp.ndarray) -> EnvState:
    """Resets the environment to an initial state."""
    qpos = self.sys.qpos0
    qvel = jp.zeros((self.sys.nv,))

    mjx_state = self.mjx_init(qpos, qvel)
    action = jp.zeros(self.sys.nu)
    rew = jp.zeros(1)
    done = jp.zeros(1)

    _, normalized_piaon_qpos, key_activation = get_piano_state(mjx_state.qpos, self.piano_joint_range)
    goals = self._goal_state[0:self._lookahead_steps]
    hands_state = get_hands_state(mjx_state.qpos)
    fingertip_xpos = get_fingertip_xpos(mjx_state.site_xpos)
    obs = self._get_obs(goals,
                        normalized_piaon_qpos,
                        hands_state,
                        fingertip_xpos,)

    metrics = {}
    if self._return_substeps:
      # [x_dim] -> [n_frame, x_dim], such that the shape of leaves won't change in step().
      info = {'substeps': jax.tree_util.tree_map(lambda x: x[None].repeat(self._n_frames, axis=0), 
                                                 mjx_state)}
    else:
      info = {}
    return EnvState(mjx_state, obs, rew, done, metrics, info)

  def step(self, state: EnvState, action: jp.ndarray) -> EnvState:
    """Runs one timestep of the environment's dynamics
      return_substep: bool, whether to return the substep results.
    """
    mjx_state0 = state.mjx_state
    # TODO: apply piano sustain
    if self._return_substeps:
      mjx_state, substeps = self.mjx_step(mjx_state0, action)
    else:
      mjx_state = self.mjx_step(mjx_state0, action)

    # get required variables
    steps = state.info['steps'].astype(int)
    # # jax.debug.print('--------------steps: {x}', x=steps)
    # # jax.debug.print('--------------applied action {x}', x=action)  
    _, normalized_piano_qpos, key_activation = get_piano_state(mjx_state.qpos, self.piano_joint_range)

    current_goals = jax.lax.dynamic_slice_in_dim(self._goal_state, steps, self._lookahead_steps)
    # # jax.debug.print('---------------goals: {x}', x=current_goals[0])

    hands_state = get_hands_state(mjx_state.qpos)
    fingertip_xpos = get_fingertip_xpos(mjx_state.site_xpos)

    reward = self._get_reward(mjx_state, current_goals[0], fingertip_xpos, normalized_piano_qpos, key_activation)

    # get observation and done
    steps = state.info['steps'].astype(int) + 1 
    next_goals = jax.lax.dynamic_slice_in_dim(self._goal_state, steps, self._lookahead_steps)
    obs = self._get_obs(next_goals, 
                        normalized_piano_qpos, 
                        hands_state, 
                        fingertip_xpos,)

    done = jax.lax.cond((steps - 1) == (self.notes_length - 1), 
                        lambda x: jp.ones(1), 
                        lambda x: jp.zeros(1), None)
    if self._return_substeps:
      state.info['substeps'] = substeps
    return state.replace(
        mjx_state=mjx_state, obs=obs, reward=reward, done=done,
    )
   
  def _get_obs(self,
      goals, 
      piano_qpos, 
      hands_state, 
      fingertip_xpos, 
  ) -> jp.ndarray:
    """Observes piano key site position, finger states and (multi-step) goal states."""

    return jp.concatenate([goals.ravel(), piano_qpos, hands_state,
                          fingertip_xpos.ravel(),], axis=-1)

  ################## Reward terms ##################
  def _get_reward(self, mjx_state, goal, fingertip_xpos, normalized_piano_qpos, key_activation) -> float:
    fingering_reward = self._compute_fingering_reward(mjx_state, fingertip_xpos, goal)
    key_press_reward = self._compute_key_press_reward(goal, normalized_piano_qpos, key_activation)
    # TODO: Note that, to get actuator force from qfrc_actuator, 
    # the order of autuators have to be the same as the order of the joints in mjcf file.
    # A better way should be mjx_state.actuator_force (not implemented in mjx yet)
    actuator_force = get_finger_force(mjx_state.qfrc_actuator)
    actuator_velocity = mjx_state.actuator_velocity
    # jax.debug.print('actuator force: {x}, velocity: {y}', x=actuator_force, y=actuator_velocity)
    energy_reward = self._compute_energy_reward(actuator_force, actuator_velocity)
    jax.debug.print('fingering_reward: {x}, key_press_reward: {y}, energy_reward: {z}', x=fingering_reward, y=key_press_reward, z=_ENERGY_PENALTY_COEF * energy_reward)
    # print(f'energy_reward: {energy_reward}')
    # TODO: defing the reward weights
    return fingering_reward + key_press_reward + _ENERGY_PENALTY_COEF * energy_reward
      
  def _compute_fingering_reward(self, mjx_state, fingertip_xpos, goal) -> float:
    """Reward for distance between keys to press and its nearby fingertips.
        When fingering info is available, directly use self._compute_fingering_reward.
    """
    # calucate the positions of pressed keys
    keys_to_press = jp.flatnonzero(goal, size=_NUM_FINGERS, fill_value=-1) # keys to press
    # to jit the flatnonzero function, we need to make sure the returned shape is not data dependent
    # we can use fill_value to pad the array to a fixed size and use mask to filter out the padded values
    num_keys_to_press = jp.sum(jp.where(keys_to_press > -1, 1, 0))
    # # jax.debug.print('num_keys_to_press: {x}, {y}', x=num_keys_to_press, y=keys_to_press)
    key_mask = jp.where(keys_to_press > -1, 1, 0) # assign a large value to the padded values
    # import ipdb; ipdb.set_trace()
    def _cal_rew():
      # calculate the positions of keys that need to be pressed
      # TODO: double check the site pos of both fingertip and piano key

      key_xpos = get_piano_geom_xpos(mjx_state.geom_xpos, keys_to_press) # shape (num_pressed_keys, 3)
      # jax.debug.print('------- Fingering reward --------')
      # jax.debug.print('fingertip_xpos: {x}', x=fingertip_xpos)
      # jax.debug.print('key_xpos: {x}', x=key_xpos)
      key_geom_size = self.piano_key_geom_size[keys_to_press]
      # jax.debug.print('piano key geom size: {x}', x=key_geom_size)
      key_xpos_z = key_xpos[:, 2] + 0.35 * key_geom_size[:, 2] # slightly put the target lower than the key surface to encourage the finger to press the key
      key_xpos_x = key_xpos[:, 0] + 0.35 * key_geom_size[:, 0] 
      key_xpos = jp.stack([key_xpos_x, key_xpos[:, 1], key_xpos_z], axis=-1)
      # jax.debug.print('piano updated key xpos: {x}', x=key_xpos) 
      # apply mask to remove padded values
      # key_xpos = key_mask[:, jp.newaxis] * key_xpos
      # # # jax.debug.print('masked_key_xpos: {x}', x=key_xpos)
      # calcualte the distance between keys and fingers
      # # jax.debug.print('fingertip_xpos: {x}', x=fingertip_xpos)
      fingertip_xpos_tiled = jp.tile(fingertip_xpos[:, jp.newaxis, :], (1, keys_to_press.shape[0], 1))
      key_pos_tiled = jp.tile(key_xpos[jp.newaxis, :, :], (_NUM_FINGERS, 1, 1))
      dist = jp.linalg.norm(fingertip_xpos_tiled - key_pos_tiled, axis=-1)
      # # jax.debug.print('dist matrix: {x}, {y}, {z}', x=dist, y=key_mask[jp.newaxis, :], z=key_mask[jp.newaxis, :]*dist)
      # apply mask
      masked_dist = key_mask[jp.newaxis, :] * dist
      masked_dist = jp.where(masked_dist == 0, 10, masked_dist)

      # jax.debug.print('masked_dist matrix: {x}', x=masked_dist)
      # calculate the shortest distance between keys and fingers by solving a linear assignment problem
      row_ind, col_ind = linear_sum_assignment(masked_dist)
      distance = masked_dist[row_ind, col_ind]
      # jax.debug.print('selected_key_xpos: {x},{y} shortest dist {z}', x=row_ind, y=col_ind, z=distance)
      rews = tolerance(
        distance,
        bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
        margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
        sigmoid="gaussian",
      )
      # jax.debug.print('raw rewards : {x}', x=rews)
      # jax.debug.print('rewards: {x}', x=jp.sum(rews)/num_keys_to_press)
      return jp.sum(rews, keepdims=True) / num_keys_to_press

    # if no key is pressed
    rew = jax.lax.cond(num_keys_to_press == 0, lambda: jp.ones(1), _cal_rew)
    # jax.debug.print('fingering reward: {x}', x=rew)
    return rew


  def _compute_key_press_reward(self, goal, normalized_piano_qpos, key_activation) -> float:
    """Reward for pressing the right keys at the right time."""
    keys_to_press = jp.flatnonzero(goal, size=_NUM_FINGERS, fill_value=-1)
    key_mask = jp.where(keys_to_press > -1, 1, 0)
    num_keys_to_press = jp.sum(key_mask)
    # It's possible we have no keys to press at this timestep, so we need to check
    # that `on` is not empty.
    def cal_rew():
      rews = key_mask * tolerance(
          goal[keys_to_press] - normalized_piano_qpos[keys_to_press], # TODO: check this, is it the correct term to calculate?
          bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
          margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
          sigmoid="gaussian",
      )
      # jax.debug.print('------ key press reward,keys {x}', x=keys_to_press)
      # jax.debug.print('goal: {x}, actual {y}', x=goal[keys_to_press], y=normalized_piano_qpos[keys_to_press])  
      # # jax.debug.print('key press rewards: {x}', x=rews)
      
      rew = 0.5 * (jp.sum(rews, keepdims=True) / num_keys_to_press)
      return rew
    
    rew1 = jax.lax.cond(num_keys_to_press > 0, cal_rew, lambda: jp.zeros(1))
    
    # If there are any false positives, the remaining 0.5 reward is lost.
    # goal:    0 0 0 1 0 1 0 0 0 0
    # key :    0 0 1 1 0 1 0 0 0 0
    # 1-goal:  1 1 1 0 1 0 1 1 1 1
    # result:  0 0 1 0 0 0 0 0 0 0 . -> flase positive
    rew2 = 0.5 * (1. - ((1. - goal) * key_activation).any(keepdims=True))
    # jax.debug.print('key activation: {x}', x=key_activation)
    # jax.debug.print('key goal: {x}', x=goal)
    jax.debug.print('key press reward: r1 {x}, r2 {y}', x=rew1, y=rew2)
    return rew1 + rew2
    # return jp.zeros(1)

  def _compute_energy_reward(self, force, velocity) -> float:
    """Reward for minimizing energy."""
    reward = - jp.sum(jp.abs(force) * jp.abs(velocity), keepdims=True)
    return reward
    # return jp.zeros(1)

# %%
# for debug
# from mjx_env import VmapWrapper, EpisodeWrapper
# midi = Path('robopianist/music/data/musescore/Flight_of_the_Bumblebee.mid')
# env = RoboPianist(midi)

# env_key = jax.random.PRNGKey(42)

# env_state = env.reset(env_key)
# for i in range(100):
#   env_key, action_key = jax.random.split(env_key, 2)

#   action = jax.random.uniform(action_key, (env.action_size,))
#   env_state = env.step(env_state, action)

  # print(env_state.reward.shape)

# # %%
# venv
# from mjx_wrappers import VmapWrapper, EpisodeWrapper, CanonicalSpecWrapper, ObservationActionRewardWrapper
# env = RoboPianist(midi)
# env = VmapWrapper(env)
# env = ObservationActionRewardWrapper(env)
# env = CanonicalSpecWrapper(env, clip=True)
# env = EpisodeWrapper(env, episode_length=200, action_repeat=1)

# env_key = jax.random.PRNGKey(42)
# num_envs = 1024
# local_devices_to_use = jax.local_device_count()
# env_keys = jax.random.split(env_key, num_envs // jax.process_count())
# #   env_keys = jnp.reshape(env_keys,
# #                          (local_devices_to_use, -1) + env_keys.shape[1:])
# #   env_state = jax.pmap(env.reset)(env_keys)
# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)
# # jit_reset = env.reset
# # jit_step = env.step
# env_state = jit_reset(env_keys)
# # rollout = [env_state.mjx_state]
# for i in range(200):
#   env_key, action_key = jax.random.split(env_key, 2)

#   action = jax.random.uniform(action_key, (*env_keys.shape[:1], env.action_size))
#   action = jp.zeros_like(action)
#   env_state = jit_step(env_state, action)
#   # rollout.append(env_state.mjx_state)
#   print(env_state.reward.shape)



  # if jnp.all(env_state.done):
  #   env_state = jax.pmap(env.reset)(env_keys) # TODO: check how to reset only the done envs
# import ipdb; ipdb.set_trace()
# video = env.render(rollout, camera='side')
  


# # # %%
# # # test the naive mujoco-mjx memory usage
# # path = _HERE/ 'third_party' / 'shadow_hand' / 'robopianist.xml'

# # mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
# # mj_model.opt.timestep = _PHYSICS_TIMESTEP

# # mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
# # mj_data = mujoco.MjData(mj_model)
# # renderer = mujoco.Renderer(mj_model, height=240, width=320)

# # mjx_model = mjx.put_model(mj_model)
# # mjx_data = mjx.put_data(mj_model, mj_data)

# # duration = 3.8
# # framerate = 60

# # rng = jax.random.PRNGKey(0)
# # rng = jax.random.split(rng, 100)
# # batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (31,))))(rng)
# # jit_step = jax.vmap(mjx.step, in_axes=(None, 0))
# # batch = jit_step(mjx_model, batch)
# # print(batch.qpos.shape)
# # %%
