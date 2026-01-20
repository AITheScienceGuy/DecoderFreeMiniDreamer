#!/usr/bin/env python

import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000, num_envs=8, seq_len=50):
        self.capacity = capacity
        self.num_envs = num_envs
        self.seq_len = seq_len
        self.pos = 0
        self.full = False
        
        # 1. Image Observation (Current)
        self.obs = np.zeros((capacity, num_envs, 4, 64, 64), dtype=np.uint8)
        # 2. Image Observation (Next/Target) - Explicit storage for correctness
        self.next_obs = np.zeros((capacity, num_envs, 4, 64, 64), dtype=np.uint8)
        # 3. Action taken
        self.actions = np.zeros((capacity, num_envs), dtype=np.int64)
        # 4. Reward received
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        # 5. Episode Boundary (Terminated OR Truncated OR Life Loss)
        self.dones = np.zeros((capacity, num_envs), dtype=np.float32)
        # 6. Real Terminal (Terminated ONLY)
        self.terminals = np.zeros((capacity, num_envs), dtype=np.float32)

        # Event Anchors: Deques of (buffer_index, env_index)
        self.term_indices = deque()
        self.rew_indices = deque()

    def add_batch(self, obs_batch, next_obs_batch, act_batch, rew_batch, done_batch, term_batch, rew_eps=0.01):
        if obs_batch.dtype != np.uint8:
            obs_batch = (obs_batch * 255).astype(np.uint8)
        if next_obs_batch.dtype != np.uint8:
            next_obs_batch = (next_obs_batch * 255).astype(np.uint8)
            
        # FIX: Instead of clearing everything on wrap, remove only overwritten anchors
        if self.full:
            # We are about to overwrite self.pos. Remove any anchors pointing to this index.
            # Since we add sequentially, the oldest anchors (matching self.pos) are at the left.
            while len(self.term_indices) > 0 and self.term_indices[0][0] == self.pos:
                self.term_indices.popleft()
            
            while len(self.rew_indices) > 0 and self.rew_indices[0][0] == self.pos:
                self.rew_indices.popleft()

        self.obs[self.pos] = obs_batch
        self.next_obs[self.pos] = next_obs_batch
        self.actions[self.pos] = act_batch
        self.rewards[self.pos] = rew_batch
        self.dones[self.pos] = done_batch
        self.terminals[self.pos] = term_batch
        
        # Update Anchors
        term_envs = np.where(term_batch > 0.5)[0]
        for env_idx in term_envs:
            self.term_indices.append((self.pos, env_idx))
            
        rew_envs = np.where(np.abs(rew_batch) > rew_eps)[0]
        for env_idx in rew_envs:
            self.rew_indices.append((self.pos, env_idx))

        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True
            self.pos = 0

    def sample_sequence(self, batch_size, recent_only_pct=1.0, 
                        force_terminal_rate=0.0, force_reward_rate=0.0):
        
        curr_size = self.capacity if self.full else self.pos
        if curr_size < self.seq_len: return None

        # Clamp rates to ensure sum <= 1.0 implicitly via counts
        target_terminals = int(batch_size * force_terminal_rate)
        target_rewards = int(batch_size * force_reward_rate)
        
        # Ensure total targets do not exceed batch_size
        if target_terminals + target_rewards > batch_size:
            target_rewards = batch_size - target_terminals
            if target_rewards < 0: target_rewards = 0

        valid_indices = []
        valid_envs = []
        
        # Helper to sample from anchors
        def sample_from_anchors(anchor_list, count):
            chosen_idxs = []
            chosen_envs = []
            if not anchor_list: return [], []
            
            n_anchors = len(anchor_list)
            attempts = 0
            while len(chosen_idxs) < count and attempts < count * 5:
                attempts += 1
                # Deque supports indexing, though O(N) worst case. 
                # For sampling a few items, this is acceptable.
                rand_idx = np.random.randint(n_anchors)
                idx, env = anchor_list[rand_idx]
                
                offset = np.random.randint(0, self.seq_len)
                start = idx - offset
                
                # Handle Wrap
                if start < 0:
                    if self.full: start += self.capacity
                    else: continue
                
                if not self.full:
                    if start + self.seq_len > self.pos: continue
                
                if self.full:
                    indices = (start + np.arange(self.seq_len)) % self.capacity
                    if np.any(indices == self.pos): continue
                
                chosen_idxs.append(start)
                chosen_envs.append(env)
            
            return chosen_idxs, chosen_envs

        # 1. Terminals
        t_idxs, t_envs = sample_from_anchors(self.term_indices, target_terminals)
        for i in range(len(t_idxs)):
            valid_indices.append(t_idxs[i])
            valid_envs.append(t_envs[i])
        
        # 2. Rewards
        r_idxs, r_envs = sample_from_anchors(self.rew_indices, target_rewards)
        for i in range(len(r_idxs)):
            valid_indices.append(r_idxs[i])
            valid_envs.append(r_envs[i])
            
        # 3. Fill Remainder
        needed = batch_size - len(valid_indices)
        if needed > 0:
            start_bound = 0
            if recent_only_pct < 1.0:
                start_bound = int(curr_size * (1.0 - recent_only_pct))
            
            attempts = 0
            while len(valid_indices) < batch_size and attempts < needed * 5:
                attempts += 1
                env = np.random.randint(0, self.num_envs)
                
                if self.full:
                    start = np.random.randint(start_bound, self.capacity)
                    idxs = (start + np.arange(self.seq_len)) % self.capacity
                    if np.any(idxs == self.pos): continue
                else:
                    max_start = self.pos - self.seq_len
                    if max_start < start_bound: max_start = start_bound
                    start = np.random.randint(start_bound, max_start + 1)
                
                valid_indices.append(start)
                valid_envs.append(env)

        final_indices = np.array(valid_indices[:batch_size])
        final_envs = np.array(valid_envs[:batch_size])
        
        if self.full:
            seq_idxs = (final_indices[:, None] + np.arange(self.seq_len)) % self.capacity
        else:
            seq_idxs = final_indices[:, None] + np.arange(self.seq_len)
            
        final_envs_expanded = final_envs[:, None]
        
        batch_obs = self.obs[seq_idxs, final_envs_expanded]
        batch_next_obs = self.next_obs[seq_idxs, final_envs_expanded]
        batch_act = self.actions[seq_idxs, final_envs_expanded]
        batch_rew = self.rewards[seq_idxs, final_envs_expanded]
        batch_boundary = self.dones[seq_idxs, final_envs_expanded]
        batch_terminal = self.terminals[seq_idxs, final_envs_expanded]

        batch_obs = batch_obs.astype(np.float32) / 255.0
        batch_next_obs = batch_next_obs.astype(np.float32) / 255.0

        return (torch.tensor(batch_obs, dtype=torch.float32),
                torch.tensor(batch_next_obs, dtype=torch.float32),
                torch.tensor(batch_act, dtype=torch.long),
                torch.tensor(batch_rew, dtype=torch.float32).unsqueeze(2),
                torch.tensor(batch_boundary, dtype=torch.float32).unsqueeze(2),
                torch.tensor(batch_terminal, dtype=torch.float32).unsqueeze(2)) 

    def __len__(self):
        return (self.capacity if self.full else self.pos) * self.num_envs
