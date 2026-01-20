#!/usr/bin/env python

import gymnasium as gym
import ale_py
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from models import WorldModel, ActorCritic, EMBED_DIM, STOCH_DIM, CLASS_DIM
from utils import get_device, PreprocessAtari

def save_gif(frames, filename="gameplay.gif"):
    print(f"Saving {filename}...")
    imgs = [Image.fromarray(frame) for frame in frames]
    # Duration in ms (33ms ~= 30fps)
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], optimize=False, duration=33, loop=0)
    print("Saved!")

def watch_agent():
    device = get_device()

    # Register ALE environments
    gym.register_envs(ale_py)

    # 1. Change render_mode to "rgb_array" to capture frames
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=1)
    
    # Apply wrappers manually to match training
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, 
        terminal_on_life_loss=False, grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = PreprocessAtari(env, size=(64, 64))
    
    print(f"Loading models (Embed: {EMBED_DIM}, Stoch: {STOCH_DIM*CLASS_DIM})...")
    wm = WorldModel(action_dim=env.action_space.n).to(device)
    try:
        wm.load_state_dict(torch.load("best_world_model.pth", map_location=device))
    except FileNotFoundError:
        print("world_model.pth not found! Run training first.")
        return
    wm.eval()

    agent = ActorCritic(action_dim=env.action_space.n).to(device)
    try:
        agent.load_state_dict(torch.load("best_policy.pth", map_location=device))
    except FileNotFoundError:
        print("policy.pth not found! Run training first.")
        return
    agent.eval()

    print("Starting Recording...")
    
    # Record 3 episodes
    for episode in range(3): 
        # 1. Capture info on reset
        obs, info = env.reset()
        current_lives = info["lives"]
        
        done = False
        total_reward = 0
        
        h = torch.zeros(1, EMBED_DIM, device=device)
        z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device) 
        
        # Start with a dummy action (index 0)
        action_idx = 0
        
        frames = [] 
        
        while not done:
            # Capture the high-res frame for GIF
            frame = env.render() 
            frames.append(frame)

            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            
            with torch.no_grad():
                embed = wm.encoder(obs_tensor)
                
                # Convert integer action to one-hot for RSSM
                action_t = torch.tensor([action_idx], device=device)
                action_one_hot = F.one_hot(action_t, env.action_space.n).float()
                
                h, logits = wm.rssm.step(h, z_flat, action_one_hot, embed=embed)
                z, z_flat = wm.get_stochastic_state(logits, hard=True)
                
                feat = wm.rssm.get_feat(h, z_flat)
                
                # Low temperature for deterministic play
                action_probs = agent.get_action(feat, temperature=0.1)
                action_idx = torch.distributions.Categorical(probs=action_probs).sample().item()
            
            # 2. Capture info on step
            obs, reward, term, trunc, info = env.step(action_idx)
            done = term or trunc
            total_reward += reward
            
            # 3. Detect Life Loss & Reset Latent State
            if info["lives"] < current_lives and info["lives"] > 0:
                h = torch.zeros(1, EMBED_DIM, device=device)
                z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device)
                action_idx = 0
                
            current_lives = info["lives"]

        print(f"Episode {episode+1}: Total Reward {total_reward}")
        save_gif(frames, filename=f"episode_{episode+1}.gif")

    env.close()

if __name__ == "__main__":
    watch_agent()
