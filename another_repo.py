import argparse
import string
import numpy as np
import jax
from jax import Array
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
import optax
from flax.core import FrozenDict
from flax.struct import dataclass
from jax import jit
from typing import Callable
from flax.training.train_state import TrainState
from flax import struct
from numpy import ndarray
import tensorflow_probability.substrates.jax.distributions as tfp
from tensorboardX import SummaryWriter
from jax import device_get
import time
from tqdm import tqdm
from termcolor import colored
import signal
from flax.training import orbax_utils
import orbax.checkpoint
from jax.lax import stop_gradient
from jax import value_and_grad

from pathlib import Path


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)
        x = nn.elu(x)
        x = nn.Dense(features=256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)
        x = nn.elu(x)
        x = nn.Dense(features=1, kernel_init=nn.initializers.orthogonal(1.0),
                     bias_init=nn.initializers.constant(0.0))(x)
        return x

class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)
        x = nn.elu(x)
        x = nn.Dense(features=256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)
        x = nn.elu(x) 
        action_mean = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)

        action_logstd = self.param('logstd', nn.initializers.zeros, (1, self.action_dim))
        # action_logstd = 0.2 * jnp.ones((1, self.action_dim))
        action_logstd = jnp.broadcast_to(action_logstd, action_mean.shape) # Make logstd the same shape as actions
        return action_mean, action_logstd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=50000000, help='total timesteps of the experiment')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='the learning rate of the optimizer')
    parser.add_argument('--num-envs', type=int, default=1024, help='the number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=32,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--gamma', type=float, default=0.9, help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.85,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=8, help='the number of mini batches')
    parser.add_argument('--update-epochs', type=int, default=8, help='the K epochs to update the policy')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='the surrogate clipping coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.0000, help='coefficient of the entropy')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient of the value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='the maximum norm for the gradient clipping')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducible benchmarks')
    parser.add_argument('--exp-name', type=str, default='PPO_continuous_action', help='unique experiment name')
    parser.add_argument('--env-id', type=str, default='Pusher-v4', help='id of the environment')
    parser.add_argument('--capture-video', type=bool, default=False, help='whether to save video of agent gameplay')
    parser.add_argument('--track', type=bool, default=False, help='whether to track project with W&B')
    parser.add_argument("--wandb-project-name", type=str, default="RL-Flax", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--anneal-lr", type=bool, default=True, help="linear anneal learning rate")
    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps  # size of the batch after one rollout
    args.minibatch_size = args.batch_size // args.num_minibatches  # size of the mini batch
    args.num_updates = args.total_timesteps // args.batch_size  # the number of learning cycle

    return args


# Anneal learning rate over time
def linear_schedule(count):
    frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
    return args.learning_rate * frac


@dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@jit
def get_action_and_value(actor: TrainState, critic: TrainState,
                         next_obs: ndarray, next_done: ndarray, 
                         storage: Storage, 
                         step: int,
                         key: random.PRNGKeyArray):
    action_mean, action_logstd = actor.apply_fn({"params": actor.params}, next_obs)
    value = critic.apply_fn({"params": critic.params}, next_obs)
    action_std = jnp.exp(action_logstd)

    # Sample continuous actions from Normal distribution
    probs = tfp.Normal(action_mean, action_std)
    key, subkey = random.split(key)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action).sum(1)
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done.squeeze()),
        actions=storage.actions.at[step].set(action),
        logprobs=storage.logprobs.at[step].set(logprob.squeeze()),
        values=storage.values.at[step].set(value.squeeze()),
    )
    return storage, action, key


def rollout(
        actor: TrainState,
        critic: TrainState,
        env_state,
        storage: Storage,
        key: random.PRNGKeyArray,
        global_step: int,
        writer: SummaryWriter,
):
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        storage, action, key = get_action_and_value(actor, critic, env_state.obs, env_state.done, storage, step, key)

        env_state = envs.step(env_state, action)
        storage = storage.replace(rewards=storage.rewards.at[step].set(env_state.reward.squeeze()))

        if env_state.done.any():
            # in our task, each env has fixed episode length
            writer.add_scalar("charts/episodic_return", env_state.info["episodic_return"].mean(), global_step)

            key, sub_key = random.split(key)
            new_env_state = envs.reset(sub_key)

            # replace new_env_state's done
            new_env_state.replace(done=env_state.done)
            env_state = new_env_state
    return env_state, storage, key, global_step


@jit
def compute_gae(
        critic: TrainState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage
):
    # Reset advantages values
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = critic.apply_fn({"params": critic.params}, next_obs).squeeze()
    # Compute advantage using generalized advantage estimate
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - storage.dones[t + 1]
            nextvalues = storage.values[t + 1]
        delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
    # Save returns as advantages + values
    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage


@jit
def update_actor(actor, obs, act, logp, advantage):
    # compute policy oss
    def _actor_loss(params):
        # get action and value
        action_mean, action_logstd = actor.apply_fn({"params": params}, obs)
        action_std = jnp.exp(action_logstd)
        probs = tfp.Normal(action_mean, action_std)

        newlogp = probs.log_prob(act).sum(1)
        entropy = probs.entropy().sum(1)

        logratio = newlogp - logp
        ratio = jnp.exp(logratio)
        # Calculate how much policy is changing
        approx_kl = ((ratio - 1) - logratio).mean()

        # Advantage normalization
        norm_adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy loss
        pg_loss1 = -norm_adv * ratio
        pg_loss2 = -norm_adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Entropy loss
        entropy_loss = entropy.mean() 

        loss = pg_loss - args.ent_coef * entropy_loss

        return loss, (entropy_loss, approx_kl, action_std)
    
    (actor_loss, (entropy_loss, approx_kl, action_std)), grads = jax.value_and_grad(_actor_loss, has_aux=True)(actor.params)
    actor = actor.apply_gradients(grads=grads)
    return actor, (actor_loss, entropy_loss, approx_kl, action_std.mean())


@jit
def update_critic(critic, obs, ret, val):
    # compute critic loss
    def _critic_loss(params):
        newval = critic.apply_fn({"params": params}, obs) 
        v_loss_unclipped = (newval - ret) ** 2
        v_clipped = val + jnp.clip(
            newval - val,
            -args.clip_coef,
            args.clip_coef,
        )

        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        return v_loss

    v_loss, grads = jax.value_and_grad(_critic_loss, has_aux=False)(critic.params)
    critic = critic.apply_gradients(grads=grads)
    return critic, v_loss


def update_ppo(
        actor: TrainState,
        critic: TrainState,
        storage: Storage,
        key: random.PRNGKeyArray
):
    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1, obs_size))
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1, action_size))
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    for epoch in range(args.update_epochs):
        key, subkey = random.split(key)
        b_inds = random.permutation(subkey, args.batch_size, independent=True)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            # update actor and critic
            actor, (actor_loss, entropy_loss, approx_kl, stddev) = update_actor(actor, b_obs[mb_inds], 
                                                                               b_actions[mb_inds], 
                                                                               b_logprobs[mb_inds], 
                                                                               b_advantages[mb_inds])

            critic, v_loss = update_critic(critic, b_obs[mb_inds], 
                                                   b_returns[mb_inds], 
                                                   b_values[mb_inds])

    # Calculate how good an approximation of the return is the value function
    y_pred, y_true = b_values, b_returns
    var_y = jnp.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return actor, critic, actor_loss, v_loss, entropy_loss, approx_kl, explained_var, stddev, key

def init_envs():
    # midi = Path('robopianist/music/data/musescore/Flight_of_the_Bumblebee.mid')
    # midi = Path('robopianist/music/data/musescore/South_Pole_Expedition_Nicoline_van_Goor.mid') 
    midi = Path('robopianist/music/data/rousseau/twinkle-twinkle-trimmed.mid')
    print('>>>>>>', midi)
    from robopianist_mjx_origin.robopianist_env import RoboPianist
    from robopianist_mjx_origin.mjx_wrappers import VmapWrapper, EpisodeWrapper, CanonicalSpecWrapper, ObservationActionRewardWrapper
    from robopianist_mjx_origin.music_wrappers import MusicMetricsWrapper, SoundVideoWrapper
    env = RoboPianist(midi)
    env = ObservationActionRewardWrapper(env)
    env = EpisodeWrapper(env, episode_length=200, action_repeat=1) 
    env = CanonicalSpecWrapper(env, clip=True)
    envs = VmapWrapper(env, num_envs=args.num_envs)

    # Initialize eval env
    eval_env = RoboPianist(midi)
    eval_env = ObservationActionRewardWrapper(eval_env)
    eval_env = EpisodeWrapper(eval_env, episode_length=200, action_repeat=1) 
    eval_env = CanonicalSpecWrapper(eval_env, clip=True) 
    eval_env = SoundVideoWrapper(eval_env, video_path=Path("/ssd/rl/robopianist"), frame_rate=20)
    eval_env = MusicMetricsWrapper(eval_env)
    return envs, eval_env

def env_jit_wrapper(env):
    """ Wraps the env reset and step method to jitted functions. """
    env.reset = jax.jit(env.reset)
    env.step = jax.jit(env.step)
    return env

def eval(env, actor, key, global_step):
    env_state = env.reset(key)
    ep_reward = 0
    rollout = [env_state.mjx_state]
    while not env_state.done.squeeze():
        action_mean, action_logstd = actor.apply_fn({"params": actor.params}, env_state.obs[None])
        env_state = env.step(env_state, action_mean.squeeze())
        ep_reward += np.array(env_state.reward)
        rollout.append(env_state.mjx_state)

    # save video
    video = env.store_sound_video(rollout, video_name=f"eval_{global_step}")

    # get music metrics
    metrics = env.get_musical_metrics(rollout)
    print(metrics)
    return ep_reward, metrics


if __name__ == '__main__':
    # Make kernel interrupt be handled as normal python error
    signal.signal(signal.SIGINT, signal.default_int_handler)

    args = parse_args()
    # seed
    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    key, env_key, actor_key, critic_key, action_key, permutation_key = random.split(key, num=6)

    envs, eval_env = init_envs()
    envs = env_jit_wrapper(envs)
    eval_env = env_jit_wrapper(eval_env)

    env_state = envs.reset(env_key)
    obs = env_state.obs
    obs_size = obs.shape[1]
    action_size = envs.action_size

    actor_def = Actor(action_dim=action_size)
    actor_params = actor_def.init(actor_key, obs)['params']
    actor = TrainState.create(
        apply_fn=actor_def.apply,
        params=actor_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=args.learning_rate, eps=1e-5 # TODO: add lr anneal
            ),
        ),
    )

    critic_def = Critic()
    critic_params = critic_def.init(critic_key, obs)['params']
    critic = TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=args.learning_rate, eps=1e-5 # TODO: add lr anneal
            ),
        ),
    )

    run_name = f"{args.exp_name}_{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
            config=vars(args)
        )

    writer = SummaryWriter(f'runs/{args.env_id}/{run_name}')

    # Initialize the storage
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs, obs_size)),
        actions=jnp.zeros((args.num_steps, args.num_envs, action_size)),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )
    global_step = 0
    start_time = time.time()
    key, sub_key = random.split(key)
    env_state = envs.reset(sub_key)

    try:
        for update in tqdm(range(1, args.num_updates + 1)):
            # collect data
            env_state, storage, action_key, global_step = rollout(actor, critic, env_state, storage,
                                                                  action_key, global_step, writer)
            storage = compute_gae(critic, env_state.obs, env_state.done, storage)

            # update agent
            actor, critic, actor_loss, v_loss, entropy_loss, approx_kl, explained_var, stddev, permutation_key = update_ppo(
                actor, critic, storage, permutation_key)

            writer.add_scalar("losses/actor_stddev", stddev, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # eval
            if update % 20 == 0:
                key, sub_key = random.split(key)
                eval_ep_reward, eval_metrics = eval(eval_env, actor, sub_key, global_step)
                writer.add_scalar("eval/eval_episodic_return", eval_ep_reward, global_step)
                # add music metrics
                writer.add_scalar("eval/precision", eval_metrics.precision, global_step)
                writer.add_scalar("eval/recall", eval_metrics.recall, global_step)
                writer.add_scalar("eval/f1", eval_metrics.f1, global_step)


        print(colored('Training complete!', 'green'))
    except KeyboardInterrupt:
        print(colored('Training interrupted!', 'red'))
    finally:
        writer.close()

        ckpt = {'actor': actor, 'critic': critic,}
        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        path = Path.cwd() / f"model_{global_step}"
        orbax_checkpointer.save(path, ckpt)