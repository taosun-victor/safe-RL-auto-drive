import cvxpy as cvx
from qcqp import *

import math
import time
from collections import deque

import numpy as np
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.common import TensorboardWriter

import tensorflow as tf

class SACWithVAE_safe(SAC):
    """
    Custom version of Soft Actor-Critic (SAC) to use it with donkey car env.
    It is adapted from the stable-baselines version.

    Notable changes:
    - optimization is done after each episode and not at every step
    - this version is integrated with teleoperation

    """
    def optimize(self, step, writer, current_lr):
        """
        Do several optimization steps to update the different networks.
        
        :param step: (int) current timestep
        :param writer: (TensorboardWriter object)
        :param current_lr: (float) Current learning rate
        :return: ([np.ndarray]) values used for monitoring
        """
        train_start = time.time()
        mb_infos_vals = []
        for grad_step in range(self.gradient_steps):
            if step < self.batch_size or step < self.learning_starts:
                break
            self.n_updates += 1
            # Update policy and critics (q functions)
            mb_infos_vals.append(self._train_step(step, writer, current_lr))

            if (step + grad_step) % self.target_update_interval == 0:
                # Update target network
                self.sess.run(self.target_update_op)
        if self.n_updates > 0:
            print("SAC training duration: {:.2f}s".format(time.time() - train_start))
        return mb_infos_vals



    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=1, tb_log_name="SAC", print_freq=100):

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)

            start_time = time.time()
            episode_rewards = [0.0]
            is_teleop_env = hasattr(self.env, "wait_for_teleop_reset")
            # TeleopEnv
            if is_teleop_env:
                print("Waiting for teleop")
                obs = self.env.wait_for_teleop_reset()
                info = {"cte": 0.0}
            else:
                obs = self.env.reset()
                info = {"cte": 0.0}

            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            ep_len = 0
            self.n_updates = 0
            infos_values = []
            mb_infos_vals = []

            # ---------------------load the trained NN for safety signal

            tf_obs = tf.placeholder(tf.float32, shape=(1, 104))

            hidden1 = tf.layers.dense(tf_obs, 64, tf.nn.relu)
            hidden2 = tf.layers.dense(hidden1, 16, tf.nn.relu)
            output1 = tf.layers.dense(hidden2, 2)

            hidden3 = tf.layers.dense(tf_obs, 64, tf.nn.relu)
            hidden4 = tf.layers.dense(hidden3, 16, tf.nn.relu)
            output2 = tf.layers.dense(hidden4, 3)

            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, "./saved_params/param03-level1-quad/safe_layer")

# --------------------------------------------------------

            fr = open("dump_reward.txt", "w")
            fv = open("dump_violation.txt", "w")
            cum_reward = []
            num_vio = 0

            for step in range(total_timesteps):
                # Compute current learning_rate
                frac = 1.0 - step / total_timesteps
                current_lr = self.learning_rate(frac)

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if step < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                # ---------- use trained NN to revise the action
                if action[1] < 0:
                    action[1] *= -1
                print("h1, action ", action)
                proposed_action = action.copy()
                action_take = action.copy()
                proposed_action = np.asarray(proposed_action).reshape((1, 2))
                #print ("h2, proposed_action", proposed_action)
             
                #print("obs shape", obs.shape)
                v1 = sess.run(output1, {tf_obs: obs.reshape((1, 104))})
             
                v2 = sess.run(output2, {tf_obs: obs.reshape((1, 104))})
                q = [v2[0][0], 0.5*v2[0][1], 0.5*v2[0][1], v2[0][2]]
                q = np.reshape(q, (2,2))
             
                x = cvx.Variable(1, 2)
                obj = cvx.sum_squares(x - proposed_action)
                cons = [info["cte"] + v1 * x.T + x * q * x.T <= 4.8, x[1] > 0]
                prob = cvx.Problem(cvx.Minimize(obj), cons)
             
                try:
                    qcqp = QCQP(prob)
                    qcqp.suggest(SDR)
                    f_cd, v_cd = qcqp.improve(COORD_DESCENT)
                    print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))

                    if v_cd == 0:
                        new_action = x.value
                        new_action = np.asarray(new_action).reshape((1,2))
                        print("h5, action ", new_action)
                        action_take[0] = new_action[0][0]
                        action_take[1] = new_action[0][1]
                        new_obs, reward, done, new_info = self.env.step(action_take)
                        action = action_take
                    else:
                        new_obs, reward, done, new_info = self.env.step(action)    

                except:
                    new_obs, reward, done, new_info = self.env.step(action)
#             -----------------------------------------

                ep_len += 1
                
                if (len(cum_reward) == 10):
                    cum_reward.pop(0)
                cum_reward.append(reward)
                curr = 0.0
                for i in range(len(cum_reward)):
                    idx = len(cum_reward) - i - 1
                    curr += cum_reward[idx] * (0.99**i)
                fr.write("%f \n" %(curr))
                fv.write("%d \n" %(num_vio))



                if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
                    print("{} steps".format(ep_len))

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs
                info = new_info

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, step)

                if ep_len > self.train_freq:
                    print("Additional training")
                    self.env.reset()
                    mb_infos_vals = self.optimize(step, writer, current_lr)
                    done = True

                episode_rewards[-1] += reward
                if done:
                    num_vio += 1
                    if not (isinstance(self.env, VecEnv) or is_teleop_env):
                        obs = self.env.reset()

                    print("Episode finished. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))
                    episode_rewards.append(0.0)
                    ep_len = 0
                    mb_infos_vals = self.optimize(step, writer, current_lr)

                    # Refresh obs when using TeleopEnv
                    if is_teleop_env:
                        print("Waiting for teleop")
                        obs = self.env.wait_for_teleop_reset()

                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", self.n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", step)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            if is_teleop_env:
                self.env.is_training = False
            # Use last batch
            print("Final optimization before saving")
            self.env.reset()
            mb_infos_vals = self.optimize(step, writer, current_lr)
        return self
