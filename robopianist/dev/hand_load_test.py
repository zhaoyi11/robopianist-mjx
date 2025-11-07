# %%
import os
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
path = Path().cwd()/ 'third_party' / 'shadow_hand'
# path = Path('/ssd/rl/robopianist')/ 'robopianist' / 'models' / 'hands' / 'third_party' / 'shadow_hand'
import jax
import mediapy as media
import mujoco
# mj_model = mujoco.MjModel.from_xml_path((path/'right_hand.xml').as_posix())
mj_model = mujoco.MjModel.from_xml_path((path/'robopianist.xml').as_posix())
# mj_model = mujoco.MjModel.from_xml_path((path/'humanoid.xml').as_posix())
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
# mjx_data = mjx.put_data(mj_model, mj_data)
while mj_data.time < duration:
  # mjx_data = jit_step(mjx_model, mjx_data)
  mujoco.mj_step(mj_model, mj_data)
  if len(frames) < mj_data.time * framerate:
    # mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

media.show_video(frames, fps=framerate)

# %%
# use mjx
from mujoco import mjx
mj_model = mujoco.MjModel.from_xml_path((Path().cwd()/'third_party'/'shadow_hand'/'robopianist.xml').as_posix())
# from etils import epath
# path = epath.Path(epath.resource_path('mujoco')) / (
        # 'mjx/test_data/shadow_hand')
# mj_model = mujoco.MjModel.from_xml_path(
        # (path / 'scene_right.xml').as_posix())
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)
# # put on the GPU devices
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
  mjx_data = jit_step(mjx_model, mjx_data)
  if len(frames) < mjx_data.time * framerate:
    mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

media.show_video(frames, fps=framerate)
# %%
