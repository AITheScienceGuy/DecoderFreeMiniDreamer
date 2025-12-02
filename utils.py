#!/usr/bin/env python

import gymnasium as gym
import ale_py
import cv2
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

class PreprocessAtari(gym.ObservationWrapper):
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, size[0], size[1]), dtype=np.float32
        )

    def observation(self, obs):
        obs = np.array(obs)
        processed_frames = []
        for i in range(obs.shape[0]):
            frame = obs[i] 
            img = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            processed_frames.append(img)
        stacked_obs = np.stack(processed_frames, axis=0)
        return stacked_obs.astype(np.float32) / 255.0

class ClipReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        return np.sign(reward)

def make_single_env():
    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, 
        terminal_on_life_loss=True, 
        grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = PreprocessAtari(env, size=(64, 64))
    env = ClipReward(env)
    return env

def make_vector_env(num_envs=8):
    return AsyncVectorEnv([make_single_env for _ in range(num_envs)])

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
