import abc
from typing import Any, Dict, List, Optional, Sequence, Union

import robopianist_mjx.base as base
# from brax.io import image
from flax import struct
import jax
import mujoco
from mujoco import mjx
import numpy as np

import jax
from jax import numpy as jp
from mujoco import mjx

from robopianist_mjx.base import MjxState, Contact, Motion, System, Transform
import jax
from jax import numpy as jp
from mujoco import mjx


def _reformat_contact(sys: System, data: MjxState) -> MjxState:
  """Reformats the mjx.Contact into a brax.base.Contact."""
  if data.contact is None:
    return data

  elasticity = jp.zeros(data.contact.pos.shape[0])
  body1 = jp.array(sys.geom_bodyid)[data.contact.geom1] - 1
  body2 = jp.array(sys.geom_bodyid)[data.contact.geom2] - 1
  link_idx = (body1, body2)
  data = data.replace(
      contact=Contact(
          link_idx=link_idx, elasticity=elasticity, **data.contact.__dict__
      )
  )
  return data


def init(
    sys: System, q: jax.Array, qd: jax.Array, unused_debug: bool = False
) -> MjxState:
  """Initializes physics data.

  Args:
    sys: a brax System
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    unused_debug: ignored

  Returns:
    data: initial physics data
  """

  data = mjx.make_data(sys)
  data = data.replace(qpos=q, qvel=qd)
  data = mjx.forward(sys, data)

  q, qd = data.qpos, data.qvel
  x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
  cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
  offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]

  offset = Transform.create(pos=offset)
  xd = offset.vmap().do(cvel)

  data = _reformat_contact(sys, data)
  return MjxState(q=q, qd=qd, x=x, xd=xd, **data.__dict__)


def step(
    sys: System, state: MjxState, act: jax.Array, unused_debug: bool = False
) -> MjxState:
  """Performs a single physics step using position-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    sys: a brax System
    state: physics data prior to step
    act: (act_size,) actuator input vector
    unused_debug: ignored

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  data = state.replace(ctrl=act)
  data = mjx.step(sys, data)

  q, qd = data.qpos, data.qvel
  x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
  cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
  offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
  offset = Transform.create(pos=offset)
  xd = offset.vmap().do(cvel)

  data = _reformat_contact(sys, data)
  return data.replace(q=q, qd=qd, x=x, xd=xd)


########### ENV ###########

@struct.dataclass
class EnvState(base.Base):
  """Environment state for training and inference."""

  mjx_state: Optional[MjxState]
  obs: jax.Array
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)


class Env(abc.ABC):
  """Interface for driving training and inference."""

  @abc.abstractmethod
  def reset(self, rng: jax.Array) -> EnvState:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    """Run one timestep of the environment's dynamics."""

  @property
  @abc.abstractmethod
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""

  @property
  @abc.abstractmethod
  def action_size(self) -> int:
    """The size of the action vector expected by step."""

  @property
  def unwrapped(self) -> 'Env':
    return self


class MJXEnv(Env):
  """API for driving a brax system for training and inference."""
  def __init__(
      self,
      sys: base.System,
      n_frames: int = 1,
      debug: bool = False,
      return_substeps: bool = False,
  ):
    """Initializes MjxEnv.

    Args:
      sys: system defining the kinematic tree and other properties
      n_frames: the number of times to step the physics step for each
        environment step
      debug: whether to get debug info from the mjx init/step
    """
    self.sys = sys
    self._n_frames = n_frames
    self._debug = debug
    self._return_substeps = return_substeps

  def mjx_init(self, q: jax.Array, qd: jax.Array) -> MjxState:
    """Initializes the pipeline state."""
    return init(self.sys, q, qd, self._debug)

  def mjx_step(self, mjx_state: Any, action: jax.Array):
    """Takes a physics step using the physics pipeline."""

    def f(state, _):
      next_step = step(self.sys, state, action, self._debug)
      if self._return_substeps:
        return (next_step, next_step)
      else:
        return (next_step, None)

    carry, states = jax.lax.scan(f, mjx_state, (), self._n_frames)
    if self._return_substeps:
      return carry, states
    else:
      return carry

  @property
  def dt(self) -> jax.Array:
    """The timestep used for each env step."""
    return self.sys.dt * self._n_frames

  @property
  def observation_size(self) -> int:
    rng = jax.random.PRNGKey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def action_size(self) -> int:
    return self.sys.act_size()

  def render(
      self,
      trajectory: List[MjxState],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
  ) -> Sequence[np.ndarray]:
    """Renders *a trajectory* using the MuJoCo renderer."""
    mj_model = self.sys.mj_model
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    # camera = camera or -1

    def get_image(state: MjxState):
      d = mujoco.MjData(mj_model)
      d.qpos, d.qvel = state.q, state.qd
      mujoco.mj_forward(mj_model, d)
      # renderer.update_scene(d, camera=camera)
      # TODO:
      renderer.update_scene(d)
      return renderer.render()

    if isinstance(trajectory, list):
      return [get_image(s) for s in trajectory]

    return get_image(trajectory)