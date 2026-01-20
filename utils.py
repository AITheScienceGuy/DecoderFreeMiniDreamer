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

def make_single_env(env_name, mode='train'):
    gym.register_envs(ale_py)
    env = gym.make(env_name, frameskip=1)
    
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, 
        terminal_on_life_loss=False, 
        grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = PreprocessAtari(env, size=(64, 64))
    return env

def make_vector_env(num_envs, env_name, mode='train'):
    return AsyncVectorEnv([lambda: make_single_env(env_name, mode) for _ in range(num_envs)])

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
