import time
import math
from collections import deque

import numpy as np
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnv
#from stable_baselines.a2c.utils import total_episode_reward_logger
#from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.common import TensorboardWriter


class SampleWithVAE(SAC):
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=1, tb_log_name="SAC", print_freq=100):

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            is_teleop_env = hasattr(self.env, "wait_for_teleop_reset")
            # TeleopEnv
            if is_teleop_env:
                print("Waiting for teleop")
                obs = self.env.wait_for_teleop_reset()
                info = {"cte": 0.0}
            else:
                obs = self.env.reset()
                info = {"cte": 0.0}
           
            file1 = open("cte.log", "a")
            file2 = open("obs.log", "a")
            for step in range(total_timesteps):

                if callback is not None:
                    if callback(locals(), globals()) is False:
                        break

                action = self.env.action_space.sample()

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, new_info = self.env.step(action)
                print (obs)
                print (info["cte"], action, new_info["cte"])
                if math.fabs(info["cte"] - new_info["cte"]) < 1.5:
                    file1.write("%f\t %f\t %f\t %f\n" %(info["cte"], action[0], action[1], new_info["cte"]))
                    obs_flatten = obs.flatten()
                    file2.write(' '.join(str(item) for item in obs_flatten) + "\n")
                    #file2.write("\n")

                obs = new_obs
                info = new_info

                if done:
                    if not (isinstance(self.env, VecEnv) or is_teleop_env):
                        obs = self.env.reset()

                    if is_teleop_env:
                        print("Waiting for teleop")
                        obs = self.env.wait_for_teleop_reset()


            if is_teleop_env:
                self.env.is_training = False
            print("Final optimization before saving")
            self.env.reset()
            file1.close()
            file2.close()
        return self
