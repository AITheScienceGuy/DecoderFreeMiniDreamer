#!/usr/bin/env python

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=5000, num_envs=16, seq_len=50):
        """
        capacity: Number of time steps PER environment.
        Total frames stored = capacity * num_envs.
        """
        self.capacity = capacity
        self.num_envs = num_envs
        self.seq_len = seq_len
        self.pos = 0
        self.full = False
        
        # Storage: (Time, Env, Channels, H, W)
        # Using uint8 to save RAM (converted to float in sample)
        self.obs = np.zeros((capacity, num_envs, 4, 64, 64), dtype=np.uint8)
        self.actions = np.zeros((capacity, num_envs), dtype=np.int64)
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs), dtype=np.float32)

    def add_batch(self, obs_batch, act_batch, rew_batch, done_batch):
        """
        Expects batches of shape (Num_Envs, ...)
        We write to the current time 'pos' across all envs at once.
        """
        if obs_batch.dtype != np.uint8:
            obs_batch = (obs_batch * 255).astype(np.uint8)
            
        self.obs[self.pos] = obs_batch
        self.actions[self.pos] = act_batch
        self.rewards[self.pos] = rew_batch
        self.dones[self.pos] = done_batch
        
        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True
            self.pos = 0

    def sample_sequence(self, batch_size, recent_only_pct=1.0):
        """
        Samples (B, T, ...) valid sequences.
        recent_only_pct: If < 1.0, restricts sampling to the most recent % of data.
        """
        valid_obs = []
        valid_act = []
        valid_rew = []
        valid_don = []
        
        curr_size = self.capacity if self.full else self.pos
        
        if curr_size < self.seq_len:
            return None

        start_bound = 0
        if recent_only_pct < 1.0:
            start_bound = int(curr_size * (1.0 - recent_only_pct))
        
        count = 0
        attempts = 0
        max_attempts = batch_size * 50 # Safety break
        
        while count < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Pick random Environment
            env_idx = np.random.randint(0, self.num_envs)
            
            # Pick random logical index (virtual time)
            # We want a sequence of length seq_len ending at 'end_logical'
            # So start_logical must be in [start_bound, curr_size - seq_len]
            if start_bound >= curr_size - self.seq_len:
                # Not enough recent data yet, fall back to full range
                start_logical = np.random.randint(0, curr_size - self.seq_len)
            else:
                start_logical = np.random.randint(start_bound, curr_size - self.seq_len)
            
            if self.full:
                start_t = (self.pos + start_logical) % self.capacity
            else:
                # If not full, 0 is 0.
                start_t = start_logical
            
            if start_t + self.seq_len > self.capacity:
                continue
                
            end_t = start_t + self.seq_len
            
            # Check Dones
            dones_seq = self.dones[start_t : end_t, env_idx]
            
            # We reject if a done occurs in the middle [0 ... T-2]
            # We allow done at [T-1] (end of sequence)
            if np.any(dones_seq[:-1] > 0):
                continue

            # 5. Retrieve Data
            seq_obs = self.obs[start_t : end_t, env_idx]
            seq_act = self.actions[start_t : end_t, env_idx]
            seq_rew = self.rewards[start_t : end_t, env_idx]
            seq_don = self.dones[start_t : end_t, env_idx]

            valid_obs.append(seq_obs)
            valid_act.append(seq_act)
            valid_rew.append(seq_rew)
            valid_don.append(seq_don)
            count += 1
            
        if count < batch_size:
            return None # Should not happen often once buffer is filled

        # Stack into (Batch, Time, ...)
        batch_obs = np.stack(valid_obs).astype(np.float32) / 255.0
        batch_act = np.stack(valid_act)
        batch_rew = np.stack(valid_rew)
        batch_don = np.stack(valid_don)

        return (torch.tensor(batch_obs, dtype=torch.float32),
                torch.tensor(batch_act, dtype=torch.long),
                torch.tensor(batch_rew, dtype=torch.float32).unsqueeze(2),
                torch.tensor(batch_don, dtype=torch.float32).unsqueeze(2))

    def __len__(self):
        return (self.capacity if self.full else self.pos) * self.num_envs
