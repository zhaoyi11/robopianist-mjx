import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".45"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from mjx_env import MjxEnv, EnvState
import mjcf
from pathlib import Path
import mujoco
import jax
import sim_math as utils
from jax import numpy as jp
from mujoco import mjx

_HERE = Path(__file__).resolve().parent
_SHADOW_HAND_DIR = _HERE / "third_party" / "shadow_hand"

################## Constants ##################
# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005
# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz. # TODO: whether to increase control timesteps
# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05
# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3
# Transparency of fingertip geoms.
_FINGERTIP_ALPHA = 1.0
# Bounds for the uniform distribution from which initial hand offset is sampled.
_POSITION_OFFSET = 0.05
###############################################

path = _HERE/ 'third_party' / 'shadow_hand' / 'robopianist.xml'

mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
# mj_model.opt.timestep = _PHYSICS_TIMESTEP

# from etils import epath
# path = epath.Path(epath.resource_path('mujoco')) / (
#         'mjx/test_data/shadow_hand'
#     )
# mj_model = mujoco.MjModel.from_xml_path(
#         (path / 'scene_right.xml').as_posix())

# mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
mj_data = mujoco.MjData(mj_model)
# renderer = mujoco.Renderer(mj_model, height=240, width=320)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

duration = 3.8
framerate = 60

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 4096)
batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (134,))))(rng)
jit_step = jax.vmap(mjx.step, in_axes=(None, 0))
batch = jit_step(mjx_model, batch)
print(batch.qpos.shape)