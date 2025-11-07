import abc
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union, NamedTuple
import jax
from jax import numpy as jp
import numpy as np
import dm_env
from dm_env import specs
import mujoco
import imageio
import subprocess
import shutil
from pathlib import Path
import wave

from robopianist_mjx.robopianist_env import _NUM_PIANO_KEYS
from robopianist import SF2_PATH
from robopianist_mjx.mjx_env import Env, EnvState, MjxState
from robopianist_mjx.mjx_wrappers import Wrapper
from robopianist_mjx.robopianist_env import get_piano_state
from robopianist_mjx.music import midi_message
from robopianist_mjx.music import synthesizer
from robopianist_mjx.music import constants as consts
from robopianist_mjx.music import midi_module

from sklearn.metrics import precision_recall_fscore_support

class SoundVideoWrapper(Wrapper):
    """Wraps an environment to store video together with sound.
       !!! Note that this wrapper is not jitable. 
       It receives a trajectory and stores the corresponding video and sound.
    """

    def __init__(self, env: Env, 
                 video_path, frame_rate=20, 
                 sf2_path=SF2_PATH,
                 sample_rate=consts.SAMPLING_RATE):
        self.env = env  
        self.sys = env.sys
        self._midi_module = midi_module.MidiModule() # take piano state and generate midi message used with a synthesizer to generate sound
        
        self._playback_speed: float = 1.0
        self._piano_joint_range = self.env.piano_joint_range
        self._sample_rate = sample_rate
        self._video_path = video_path
        self._frame_rate = frame_rate
        self._synth = synthesizer.Synthesizer(sf2_path, sample_rate)

    def store_sound_video(self, trajectory: List[EnvState],  
                          height: int = 480, width: int = 640, 
                          video_name: str = 'default', camera: Optional[str] = None,) -> Any:
        """Renders the environment with sound given one trajectory."""
        assert isinstance(trajectory, list) and isinstance(trajectory[0], EnvState), \
            "To generate sound, trajectory should be a list of EnvState."

        mj_model = self.sys.mj_model
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        camera = camera or -1

        def get_image(state: MjxState):
            data = mujoco.MjData(mj_model)
            data.qpos, data.qvel = state.q, state.qd

            mujoco.mj_forward(mj_model, data)
   
            # update key color for active keys
            _, _, activation = get_piano_state(data.qpos, self.piano_joint_range) 
            mj_model.geom_rgba[-_NUM_PIANO_KEYS:] = np.where(activation[:, None], 
                                                            np.array([0.2, 0.8, 0.2, 1.0]), # activation color
                                                            np.array([0.5, 0.5, 0.5, 1.0]))
            renderer.update_scene(data, camera=camera)
            return renderer.render()

        # get substeps of a trajectory
        sub_trajectory, sustain_state = self._parse_substeps(trajectory, self.env.sustain_state)
        
        video = [get_image(state.mjx_state) for state in trajectory]
        # save video
        filename = self._video_path / f"{video_name}.mp4"
        imageio.mimsave(
                str(filename), video, fps=self._frame_rate  # type: ignore
            )    
        
        #### add sound ###
        # get midi events       
        self._midi_module.init()
        for i, mjx_state in enumerate(sub_trajectory):
            _, _, activation = get_piano_state(mjx_state.qpos, self._piano_joint_range)
            self._midi_module.append(mjx_state, 
                                    activation=activation, 
                                    sustain_activation=sustain_state[i])
            
        midi_events = self._midi_module.get_all_midi_messages()
        self._add_sound(midi_events, filename)
        return video

    def _parse_substeps(self, trajectory: List[EnvState], sustain_state: np.ndarray) -> List[MjxState]:
        """Parse the substeps from the trajectory."""
        if 'substeps' not in trajectory[0].info:
            warnings.warn("No substeps in trajectory. Sound and video will be generated with lower frequence. \
                         To have higher frequency, try to set the return_substeps as True when initializing the environment.")
            sub_trajectory = [trajectory.mjx_state]
            sustain_state = sustain_state
        else:
            # process substeps
            num_substep = trajectory[0].info['substeps'].qpos.shape[0]
            sub_trajectory = []
            sub_sustatin_state = []
            # for the initial step, take the mjx_step
            sub_trajectory.append(trajectory[0].mjx_state)
            sub_sustatin_state.append(sustain_state[0])
            for i, state in enumerate(trajectory[1:]):
                for j in range(num_substep):
                    substep_state = jax.tree_map(lambda x: x[j], state.info['substeps'])
                    sub_trajectory.append(substep_state)
                    sub_sustatin_state.append(sustain_state[i])
            sub_sustatin_state = np.array(sub_sustatin_state)
        return sub_trajectory, sub_sustatin_state

    def _add_sound(self, midi_events, filename):
        # Exit if there are no MIDI events or if all events are sustain events.
        # Sustain only events cause white noise in the audio (which has shattered my
        # eardrums on more than one occasion).
        no_events = len(midi_events) == 0
        are_events_sustains = [
            isinstance(event, (midi_message.SustainOn, midi_message.SustainOff))
            for event in midi_events
        ]
        only_sustain = all(are_events_sustains) and len(midi_events) > 0
        if no_events or only_sustain:
            return

        # Synthesize waveform.
        waveform = self._synth.get_samples(midi_events)

        # Save waveform as mp3.
        waveform_name = Path(str(filename).replace('.mp4', '.mp3'))
        wf = wave.open(str(waveform_name), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self._sample_rate * self._playback_speed)
        wf.writeframes(waveform)  # type: ignore
        wf.close()

        # Make a copy of the MP4 so that FFMPEG can overwrite it.
        temp_filename = Path(str(filename).replace('.mp4', '_temp.mp4'))
        # temp_filename = self._record_dir / "temp.mp4"
        shutil.copyfile(filename, temp_filename)
        filename.unlink()

        # Add the sound to the MP4 using FFMPEG, suppressing the output.
        # Reference: https://stackoverflow.com/a/11783474
        ret = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(temp_filename),
                "-i",
                str(waveform_name),
                "-map",
                "0",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-shortest",
                str(filename),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
        if ret.returncode != 0:
            print(f"FFMPEG failed to add sound to video {temp_filename}.")

        # Remove temporary files.
        temp_filename.unlink()
        waveform_name.unlink()


class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class MusicMetricsWrapper(Wrapper):
    """Wraps an environment to compute music metrics.
       !!! Note that this wrapper is not jitable. 
       It receives a trajectory and computes the music metrics.
    """

    def __init__(self, env: Env):
        self.env = env

    def get_musical_metrics(self, trajectory: List[MjxState | EnvState]) -> Dict[str, Any]:
        # convert trajectory to a list of mjx_state
        if isinstance(trajectory[0], EnvState):
            trajectory = [state.mjx_state for state in trajectory]

        goal_state = self.env.goal_state

        precisions, recalls, f1s = [], [], []
        for i, mjx_state in enumerate(trajectory[1:]): # the first state is the initial state (no keys are pressed)
            _, _, activation = get_piano_state(mjx_state.qpos, self._piano_joint_range) 
            ground_truth = goal_state[i]
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=ground_truth, y_pred=activation, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return EpisodeMetrics(np.mean(precisions), 
                              np.mean(recalls), 
                              np.mean(f1s))
